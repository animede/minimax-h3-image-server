"""MiniMax-H3-Image transformer の軽量化ローダー: AdaLN 事前計算テーブル + torchao int8wo。

動画 H3 の `core/pruned.py`(AdaLN-pruned + int8wo)の画像版。違いは AdaLN の扱い:
- 動画版は AdaLN 射影を rank-8 に構造リファクタした配布物(近似、ほぼロスレス)を読む。
- 画像モデルは **サンプリング時刻が固定**(`1 - sigma` の 6 値 + 参照行の `reference_noise_aug`)
  なので、各ブロックの射影 `Linear(2688 -> 96768)`(bf16 520MB × 50 = 26GB、全体の 39%)の
  出力をその時刻集合について事前計算し、テーブル `[T, 96768]`(T=7、計 68MB)に置き換える。
  元の forward と同じ `F.linear(silu(temb).to(bf16), W, b)` を同じ device/dtype で計算するため
  **bit 一致(近似なし)**。ブロック 0 で実際に照合する(`verify=True`)。
- 残りの Linear(blocks 50 × 6 + token_refiner 2 × 6 = 312 層)は torchao
  `Int8WeightOnlyConfig(version=2, PerRow)`(動画 H3 の本番 int8 と同じレシピ)で、
  シャードから 1 テンソルずつ GPU へ上げて量子化する(bf16 66GB を CPU/GPU に全展開しない)。
- 時刻集合が変わる構成(例: 2-step pDMD LoRA の別スケジュール)はテーブルを作り直す必要が
  ある。未知の時刻で forward されたら黙って誤らず例外にする。

キャッシュ: 量子化済み state_dict を torch.save(pickle)で保存する(torchao テンソル
サブクラスは safetensors に直列化できない、動画版と同じ理由)。2 回目以降は
`init_empty_weights` + `load_state_dict(assign=True)` で bf16 のランダム初期化を踏まずに読む。
"""

from __future__ import annotations

import functools
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

logger = logging.getLogger("h3image.pruned")

QUANT_SUFFIXES = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
STATE_FILENAME = "transformer_int8wo_adaln_table.pt"
MODALITY_NUM = 3


def _is_quant_target(name: str) -> bool:
    return (name.startswith("transformer_blocks.") or name.startswith("token_refiner.refiner_blocks.")) and any(
        name.endswith(s + ".weight") for s in QUANT_SUFFIXES
    )


def _int8wo_config():
    from torchao.quantization import Int8WeightOnlyConfig, PerRow

    return Int8WeightOnlyConfig(version=2, granularity=PerRow(), set_inductor_config=False)


class AdaLNTable(nn.Module):
    """`MiniMaxH3ImageAdaLayerNormModulation` の置き換え。temb ではなく、transformer.forward の
    ラッパーが timestep から解決した行番号 `rows` でテーブルを引く。"""

    def __init__(self, keys: torch.Tensor, table: torch.Tensor, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.register_buffer("keys", keys)      # [T] float32、時刻の値
        self.register_buffer("table", table)    # [T, 6 * hidden * MODALITY_NUM] bf16
        self.rows: torch.Tensor | None = None

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.rows is None:
            raise RuntimeError("AdaLNTable.rows が未設定です(transformer.forward のラッパー経由で呼ぶこと)")
        out = self.table.index_select(0, self.rows).view(-1, 6 * self.hidden_size)
        return out.chunk(6, dim=-1)


def timestep_keys(pipe) -> torch.Tensor:
    """before_denoise の `MiniMaxH3ImageSetTimestepsStep` と同じ手順で時刻集合を作る。"""
    sample_sigmas = pipe.config.sample_sigmas
    timesteps = torch.tensor([1.0 - s for s in sample_sigmas[:-1]], dtype=torch.float32)
    sigmas = torch.cat([1.0 - timesteps, torch.zeros(1)])
    pipe.scheduler.set_timesteps(sigmas=sigmas, device="cpu")
    keys = torch.cat([pipe.scheduler.timesteps.to(torch.float32).cpu(),
                      torch.tensor([float(pipe.reference_noise_aug)], dtype=torch.float32)])
    return torch.unique(keys, sorted=True)


def _resolve_rows(timestep: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    t = timestep.to(keys.device, torch.float32)
    hit = t[:, None] == keys[None, :]
    if not bool(hit.any(dim=1).all()):
        missing = t[~hit.any(dim=1)].tolist()
        raise RuntimeError(
            f"AdaLN テーブルに無い timestep が渡されました: {missing}(テーブルの時刻: {keys.tolist()})。"
            "スケジュールを変えた場合はキャッシュを作り直してください")
    return hit.float().argmax(dim=1)


def install_forward_wrapper(tr) -> None:
    """forward を包み、timestep → テーブル行を全 AdaLNTable に配る。
    denoise ブロックは `inspect.signature(transformer.forward)` で引数を選別するため
    functools.wraps でシグネチャを保つ。"""
    tables = [m for m in tr.modules() if isinstance(m, AdaLNTable)]
    keys = tables[0].keys
    orig = tr.forward

    @functools.wraps(orig)
    def forward(*args, **kwargs):
        timestep = kwargs["timestep"] if "timestep" in kwargs else args[2]
        rows = _resolve_rows(timestep, keys)
        for m in tables:
            m.rows = rows
        return orig(*args, **kwargs)

    tr.forward = forward


def _replace_adaln(tr, block_index: int, module: AdaLNTable) -> None:
    tr.transformer_blocks[block_index].adaln_proj = module


def _assign(tr, name: str, device: torch.device, value: torch.Tensor) -> None:
    """ファイルの dtype のまま実体化する。dtype を渡さないと accelerate が meta モデルの既定
    dtype(float32)へキャストし、norm 等が fp32 になって活性全体が fp32 で流れる(2026-11-11 実測:
    ブロックが 2 倍遅くなった原因)。"""
    from accelerate.utils import set_module_tensor_to_device

    set_module_tensor_to_device(tr, name, device, value=value, dtype=value.dtype)
    mod_name, _, attr = name.rpartition(".")
    p = getattr(dict(tr.named_modules())[mod_name], attr)
    if isinstance(p, nn.Parameter):
        p.requires_grad_(False)


def _bnb_int8_linear(weight: torch.Tensor, device: torch.device, threshold: float = 6.0):
    """bnb LLM.int8(from_pretrained の load_in_8bit と同じ threshold=6.0)。`.to(device)` で量子化される。"""
    import bitsandbytes as bnb

    out_f, in_f = weight.shape
    lin = bnb.nn.Linear8bitLt(in_f, out_f, bias=False, has_fp16_weights=False, threshold=threshold)
    lin.weight = bnb.nn.Int8Params(weight.to(torch.bfloat16), requires_grad=False, has_fp16_weights=False)
    return lin.to(device)


def _set_module(root: nn.Module, name: str, value: nn.Module) -> None:
    parent_name, _, attr = name.rpartition(".")
    parent = dict(root.named_modules())[parent_name] if parent_name else root
    setattr(parent, attr, value)


def load_lora_pairs(lora_path: str) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]:
    """diffusers レイアウト(`transformer.<module>.lora_A/B.weight`)の LoRA を {module: (A, B)} に読む。
    scale はメタデータ `lora_scale`(pDMD 2-step は 1.0、alpha=rank=128)。"""
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            name = k
            if name.startswith("transformer."):
                name = name[len("transformer."):]
            mod, _, tail = name.rpartition(".lora_")
            pairs.setdefault(mod, {})[tail.split(".")[0]] = f.get_tensor(k)
    scale = float(meta.get("lora_scale", 1.0))
    out = {m: (d["A"], d["B"]) for m, d in pairs.items() if "A" in d and "B" in d}
    logger.info("LoRA loaded: %d pairs, scale=%s, from %s", len(out), scale, lora_path)
    return out, scale


def build_fresh(repo: str, device: torch.device, keys: torch.Tensor, verify: bool = True,
                quant: str = "int8wo", lora_path: str | None = None):
    """bf16 シャードから量子化済み transformer を組み立てる(1 テンソルずつ GPU へ)。

    quant="int8wo": torchao Int8WeightOnly(キャッシュ可。eager では bf16 へ逆量子化してから
    GEMM するため bnb より遅い、2026-11-11 実測)。
    quant="bnb-int8": bitsandbytes LLM.int8(真の int8 GEMM、速い。キャッシュ非対応で毎回構築)。
    """
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers import MiniMaxH3ImageTransformer2DModel as Cls
    from torchao.quantization import quantize_

    sub = Path(repo) / "transformer_diffusers"
    config = Cls.load_config(str(sub))
    with init_empty_weights():
        tr = Cls.from_config(config)
    hidden = config["hidden_size"]
    index = json.load(open(sub / "diffusion_pytorch_model.safetensors.index.json"))["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for name, shard in index.items():
        by_shard.setdefault(shard, []).append(name)
    cfg = _int8wo_config()
    lora, lora_scale = load_lora_pairs(lora_path) if lora_path else ({}, 1.0)
    n_merged = 0
    t0 = time.time()

    # Pass 1: AdaLN 射影でも量子化対象でもない小さいテンソル(time_embedder 等)を先に実体化し、
    # 時刻集合の temb を得る。
    for shard, names in by_shard.items():
        with safe_open(str(sub / shard), framework="pt", device="cpu") as f:
            for name in names:
                if "adaln_proj" in name or _is_quant_target(name):
                    continue
                _assign(tr, name, device, f.get_tensor(name))
    keys = keys.to(device)
    with torch.no_grad():
        temb = tr.time_embedder(tr.time_proj(keys).to(tr.time_embedder.linear_1.weight.dtype))  # float32
    act = F.silu(temb).to(torch.bfloat16)
    logger.info("pass1 (small tensors + temb for %d timesteps): %.1fs", len(keys), time.time() - t0)

    # Pass 2: ブロックの重み。AdaLN はテーブル化、Linear は int8wo 化。
    pending_adaln: dict[int, dict[str, torch.Tensor]] = {}
    n_quant = 0
    modules = dict(tr.named_modules())
    verified = False
    for shard, names in by_shard.items():
        with safe_open(str(sub / shard), framework="pt", device="cpu") as f:
            for name in names:
                if "adaln_proj" in name:
                    bi = int(name.split(".")[1])
                    slot = pending_adaln.setdefault(bi, {})
                    slot[name.rsplit(".", 1)[-1]] = f.get_tensor(name).to(device)
                    if "weight" in slot and "bias" in slot:
                        with torch.no_grad():
                            table = F.linear(act, slot["weight"], slot["bias"])  # [T, 96768] bf16、元 forward と同一演算
                        if verify and not verified:
                            lin = nn.Linear(slot["weight"].shape[1], slot["weight"].shape[0], device=device, dtype=torch.bfloat16)
                            with torch.no_grad():
                                lin.weight.copy_(slot["weight"]); lin.bias.copy_(slot["bias"])
                                ref = lin(F.silu(temb).to(torch.bfloat16))
                            assert torch.equal(ref, table), "AdaLN テーブルが元の射影と bit 一致しません"
                            verified = True
                            del lin
                        _replace_adaln(tr, bi, AdaLNTable(keys.clone(), table, hidden))
                        del pending_adaln[bi], slot
                elif _is_quant_target(name):
                    mod_name = name.rsplit(".", 1)[0]
                    w = f.get_tensor(name)
                    if mod_name in lora:
                        # W += scale * (B @ A) を fp32 で加算してから bf16 へ(量子化はその後)。
                        a, b = lora[mod_name]
                        w = (w.to(device, torch.float32) + lora_scale * (b.to(device, torch.float32) @ a.to(device, torch.float32))).to(torch.bfloat16)
                        n_merged += 1
                    if quant == "bnb-int8":
                        _set_module(tr, mod_name, _bnb_int8_linear(w.cpu() if w.is_cuda else w, device))
                    else:
                        _assign(tr, name, device, w)
                        quantize_(modules[mod_name], cfg)
                    n_quant += 1
                else:
                    continue
    assert not pending_adaln, f"AdaLN の重みが揃わないブロック: {sorted(pending_adaln)}"
    # init_empty_weights はパラメータだけを meta にし、バッファ(rope.inv_freq)は CPU に残す。
    for name, buf in list(tr.named_buffers()):
        if buf.device != device:
            mod_name, _, attr = name.rpartition(".")
            dict(tr.named_modules())[mod_name].register_buffer(attr, buf.to(device), persistent=False)
    torch.cuda.empty_cache()
    install_forward_wrapper(tr)
    tr.eval()
    if lora_path and n_merged != len(lora):
        raise RuntimeError(f"LoRA の対象 {len(lora)} 件のうち {n_merged} 件しかマージできませんでした(キー不一致)")
    logger.info("built: %d linears %s (lora merged: %d), %d AdaLN tables, verify=%s, %.1fs, alloc=%.2fGB",
                n_quant, quant, n_merged, len(tr.transformer_blocks), verified, time.time() - t0,
                torch.cuda.memory_allocated(device) / 2**30)
    return tr


def save_cache(tr, cache_dir: Path) -> None:
    tmp = cache_dir.with_name(cache_dir.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    tr.save_config(str(tmp))
    sd = {k: v.detach().cpu() for k, v in tr.state_dict().items()}
    torch.save(sd, str(tmp / STATE_FILENAME))
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
    tmp.rename(cache_dir)
    size = sum(p.stat().st_size for p in cache_dir.rglob("*") if p.is_file()) / 2**30
    logger.info("cache saved: %s (%.1fGiB)", cache_dir, size)


def load_cache(cache_dir: Path, device: torch.device):
    from accelerate import init_empty_weights
    from diffusers import MiniMaxH3ImageTransformer2DModel as Cls

    t0 = time.time()
    config = Cls.load_config(str(cache_dir))
    with init_empty_weights():
        tr = Cls.from_config(config)
    sd = torch.load(str(cache_dir / STATE_FILENAME), map_location="cpu", weights_only=False)
    hidden = config["hidden_size"]
    for i in range(len(tr.transformer_blocks)):
        k = f"transformer_blocks.{i}.adaln_proj."
        _replace_adaln(tr, i, AdaLNTable(sd[k + "keys"], sd[k + "table"], hidden))
    # 量子化済み Linear は meta の nn.Parameter(bf16)に Int8Tensor を assign で差し替える。
    tr.load_state_dict(sd, assign=True)
    tr = tr.to(device)
    install_forward_wrapper(tr)
    tr.eval()
    logger.info("cache loaded: %s %.1fs alloc=%.2fGB", cache_dir, time.time() - t0,
                torch.cuda.memory_allocated(device) / 2**30)
    return tr


def load_transformer(repo: str, pipe, device: torch.device, cache_dir: Path, quant: str = "int8wo",
                     lora_path: str | None = None):
    """キャッシュがあれば読む、無ければ構築して保存する(bnb-int8 は毎回構築)。"""
    keys = timestep_keys(pipe)
    logger.info("AdaLN timestep keys: %s", keys.tolist())
    if quant == "bnb-int8":
        return build_fresh(repo, device, keys, verify=True, quant=quant, lora_path=lora_path)
    if (cache_dir / STATE_FILENAME).exists():
        tr = load_cache(cache_dir, device)
        cached = next(m for m in tr.modules() if isinstance(m, AdaLNTable)).keys.cpu()
        if not torch.equal(cached, keys):
            raise RuntimeError(f"キャッシュの時刻集合 {cached.tolist()} が現在の {keys.tolist()} と異なります。"
                               f"{cache_dir} を削除して作り直してください")
        return tr
    tr = build_fresh(repo, device, keys, verify=True, quant=quant, lora_path=lora_path)
    save_cache(tr, cache_dir)
    return tr


def tabulate_adaln(tr, repo: str, keys: torch.Tensor, device: torch.device) -> None:
    """ロード済み transformer(例: from_pretrained の bnb int8)の AdaLN 射影を、bf16 シャードから
    計算した事前計算テーブルに差し替え、元の射影(int8 なら ~13GB)を解放する。
    テーブルは bf16 シャードの重みから計算するため量子化誤差を含まない(bit 一致)。"""
    sub = Path(repo) / "transformer_diffusers"
    index = json.load(open(sub / "diffusion_pytorch_model.safetensors.index.json"))["weight_map"]
    hidden = tr.config.hidden_size
    keys = keys.to(device)
    with torch.no_grad():
        temb = tr.time_embedder(tr.time_proj(keys).to(tr.time_embedder.linear_1.weight.dtype))
    act = F.silu(temb).to(torch.bfloat16)
    names = [n for n in index if "adaln_proj" in n]
    by_shard: dict[str, list[str]] = {}
    for n in names:
        by_shard.setdefault(index[n], []).append(n)
    pending: dict[int, dict[str, torch.Tensor]] = {}
    t0 = time.time()
    for shard, ns in by_shard.items():
        with safe_open(str(sub / shard), framework="pt", device="cpu") as f:
            for n in ns:
                bi = int(n.split(".")[1])
                slot = pending.setdefault(bi, {})
                slot[n.rsplit(".", 1)[-1]] = f.get_tensor(n).to(device)
                if "weight" in slot and "bias" in slot:
                    with torch.no_grad():
                        table = F.linear(act, slot["weight"], slot["bias"])
                    _replace_adaln(tr, bi, AdaLNTable(keys.clone(), table, hidden))
                    del pending[bi], slot
    assert not pending
    torch.cuda.empty_cache()
    install_forward_wrapper(tr)
    logger.info("AdaLN tabulated from bf16 shards: %d blocks, %.1fs, alloc=%.2fGB",
                len(tr.transformer_blocks), time.time() - t0, torch.cuda.memory_allocated(device) / 2**30)
