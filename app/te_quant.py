"""text_encoder(Qwen3-VL-32B)の nf4 量子化を、テンソル 1 つずつ・層ごとのシャードに書き出して行う(低 VRAM / 低 RAM)。

transformers の `from_pretrained(..., quantization_config=BitsAndBytesConfig(...))` はモデル全体を GPU に載せてから
量子化するので約 17GB の VRAM が要る。ここでは bf16 のシャードを safetensors の mmap から 1 テンソルずつ読み、
bitsandbytes の `Params4bit` で GPU 上で量子化して(その場で CPU へ戻し)、層ごとに 1 ファイルの safetensors へ書く。
VRAM は最大のテンソル 1 つ分(約 0.2GB)、RAM は 1 シャード分(約 0.3GB)しか使わない。

出力は transformers がそのまま読める形式(`model.safetensors.index.json` + シャード、config.json に
`quantization_config`)。量子化の規則(全 nn.Linear を nf4 化、lm_head は bf16 のまま、二重量子化なし、
blocksize 64)は `from_pretrained` 版と同じで、テンソルの値はビット一致する(1 シャードで検証済み)。
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from pathlib import Path

import torch

logger = logging.getLogger("h3image.te_quant")

_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
_REST_SHARD_MAX_BYTES = 2 << 30     # 層以外(埋め込み表・視覚塔・lm_head)は 2GB ごとに分ける


def _shard_name(key: str, n_layers: int) -> str:
    m = _LAYER_RE.match(key)
    if m:
        return f"layer-{int(m.group(1)):03d}"
    return "rest"


def quantize_tensors(src_dir: Path, n_layers: int, device: torch.device, linear_weights: set[str],
                     wanted: set[str]):
    """bf16 シャードを順に読み、(キー, テンソル群 dict) を yield する。
    linear_weights のキーは nf4 化して {key: packed uint8, key.absmax, key.quant_map, key.quant_state.bitsandbytes__nf4}、
    それ以外は bf16 のまま {key: tensor}。wanted に無いキー(52 層目以降など)は読まない。"""
    import bitsandbytes as bnb
    from safetensors import safe_open

    index = json.load(open(src_dir / "model.safetensors.index.json"))["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for key, shard in index.items():
        if key in wanted:
            by_shard.setdefault(shard, []).append(key)
    for shard in sorted(by_shard):
        with safe_open(str(src_dir / shard), framework="pt", device="cpu") as f:
            for key in by_shard[shard]:
                t = f.get_tensor(key)
                if key in linear_weights:
                    p = bnb.nn.Params4bit(t.to(torch.bfloat16), requires_grad=False, quant_type="nf4",
                                          compress_statistics=False, blocksize=64).to(device)
                    out = {key: p.data.cpu()}
                    for k, v in p.quant_state.as_dict(packed=True).items():
                        out[f"{key}.{k}"] = v.cpu()
                    del p
                else:
                    out = {key: t.to(torch.bfloat16).contiguous()}
                yield key, out


def build(src_dir: Path, dst_dir: Path, n_layers: int, device: torch.device, log=logger.info, source=None) -> None:
    """src_dir(本家 text_encoder、bf16 シャード)→ dst_dir(nf4、層ごとのシャード)。dst_dir.tmp に書いてから rename。
    source: テスト用に (キー, {名前: テンソル}) の反復子を差し替える(既定は quantize_tensors)。"""
    from accelerate import init_empty_weights
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForImageTextToText, BitsAndBytesConfig

    cfg = AutoConfig.from_pretrained(str(src_dir))
    cfg.text_config.num_hidden_layers = n_layers
    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(cfg)
    wanted = set(model.state_dict().keys())
    linear_weights = {n + ".weight" for n, m in model.named_modules()
                      if isinstance(m, torch.nn.Linear) and n != "lm_head"}
    del model

    tmp = dst_dir.with_name(dst_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    weight_map: dict[str, str] = {}
    total = 0
    t0 = time.time()
    pending: dict[str, dict[str, torch.Tensor]] = {}     # シャード名 → テンソル
    pending_bytes: dict[str, int] = {}
    expected: dict[str, int] = {}                        # シャード名 → そこに入る元テンソルの数
    for key in wanted:
        nm = _shard_name(key, n_layers)
        expected[nm] = expected.get(nm, 0) + 1
    seen: dict[str, int] = {}
    rest_no = 0
    n_done = 0

    def flush(name: str) -> None:
        nonlocal total, rest_no
        tensors = pending.pop(name)
        pending_bytes.pop(name, None)
        fname = f"model-{name}.safetensors" if name != "rest" else f"model-rest-{rest_no:02d}.safetensors"
        if name == "rest":
            rest_no += 1
        save_file(tensors, str(tmp / fname), metadata={"format": "pt"})
        for k, v in tensors.items():
            weight_map[k] = fname
            total += v.numel() * v.element_size()

    if source is None:
        source = quantize_tensors(src_dir, n_layers, device, linear_weights, wanted)
    for key, tensors in source:
        name = _shard_name(key, n_layers)
        pending.setdefault(name, {}).update(tensors)
        pending_bytes[name] = pending_bytes.get(name, 0) + sum(v.numel() * v.element_size() for v in tensors.values())
        seen[name] = seen.get(name, 0) + 1
        n_done += 1
        if name == "rest":
            if pending_bytes[name] >= _REST_SHARD_MAX_BYTES:
                flush("rest")
        elif seen[name] == expected[name]:       # 層のテンソルが揃ったら 1 ファイルに書く(元シャードの並び順に依存しない)
            flush(name)
        if n_done % 100 == 0:
            log(f"  {n_done} tensors, {time.time() - t0:.0f}s, VRAM {torch.cuda.max_memory_allocated(device) / 2**30:.2f}GB")
    for name in list(pending):
        flush(name)
    with open(tmp / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
    cfg.quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                 bnb_4bit_compute_dtype=torch.bfloat16).to_dict()
    cfg.save_pretrained(str(tmp))
    for extra in ("generation_config.json",):
        if (src_dir / extra).exists():
            shutil.copy(src_dir / extra, tmp / extra)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    tmp.rename(dst_dir)
    log(f"text_encoder nf4: {n_done} tensors -> {len(set(weight_map.values()))} shards, {total / 2**30:.1f}GiB, "
        f"{time.time() - t0:.0f}s, peak VRAM {torch.cuda.max_memory_allocated(device) / 2**30:.2f}GB")
