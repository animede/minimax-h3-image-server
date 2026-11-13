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
- quant="ck-w4a8": comfy-kitchen の W4A8(重み 4bit group16 + codebook、活性 int8 回転)。
  int8wo の 19GiB → 約 11GiB。GEMM は bf16 比 1.5 倍速(torch 2.11 + sm_120 実測)。
  `CKW4A8Linear` は forward を全面差し替えた推論専用 Linear(動画 H3 の `core/pruned.py` と同型)。
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
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

logger = logging.getLogger("h3image.pruned")

QUANT_SUFFIXES = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
STATE_FILENAME = "transformer_int8wo_adaln_table.pt"               # torchao の Int8Tensor を含む(int8wo)
STATE_FILENAME_ST = "transformer_adaln_table.safetensors"          # 素のテンソルだけのとき(ck-w4a8)
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
        out = self.table.index_select(0, self.rows.to(self.table.device)).view(-1, 6 * self.hidden_size)
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


class CKW4A8Linear(nn.Linear):
    """comfy-kitchen W4A8 の推論専用 Linear(重みは 4bit パック + スケール群のバッファ、bias なし)。

    バッファ形状は in/out_features から決まるので、meta で組んでから
    `load_state_dict(assign=True)` するキャッシュ再ロードがそのまま通る。
    注意: 親モデルに `.to(dtype=...)` を掛けると fp8/f32 のスケールが壊れる(device 移動のみ可)。
    """

    GROUP_SIZE = 16

    @staticmethod
    def convrot_groupsize_for(in_features: int) -> int:
        for g in (256, 128, 64, 32, 16):
            if in_features % g == 0:
                return g
        raise RuntimeError(f"in_features={in_features} は 16 で割り切れず W4A8 化できません")

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False, device="meta")
        self.convrot_groupsize = self.convrot_groupsize_for(in_features)
        del self._parameters["weight"]
        self.register_buffer("qdata", torch.empty(out_features, in_features // 2, dtype=torch.int8, device="meta"))
        self.register_buffer("s_rel", torch.empty(out_features, in_features // self.GROUP_SIZE,
                                                  dtype=torch.float8_e4m3fn, device="meta"))
        self.register_buffer("s_channel", torch.empty(out_features, dtype=torch.float32, device="meta"))
        self.register_buffer("codebook", torch.empty(16, dtype=torch.float32, device="meta"))

    @classmethod
    def from_weight(cls, w: torch.Tensor, device: torch.device) -> "CKW4A8Linear":
        import comfy_kitchen as ck

        out_f, in_f = w.shape
        qdata, s_rel, s_channel, correction, codebook = ck.quantize_w4a8_int8_weight(
            w.to(device, torch.bfloat16), group_size=cls.GROUP_SIZE,
            convrot_groupsize=cls.convrot_groupsize_for(in_f))
        if correction is not None:
            raise RuntimeError("quantize_w4a8_int8_weight が correction を返しました(未対応)")
        new = cls(in_f, out_f)
        new.qdata, new.s_rel, new.s_channel, new.codebook = qdata, s_rel, s_channel, codebook
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import comfy_kitchen as ck

        shape = x.shape
        y = ck.w4a8_int8_linear(x.reshape(-1, shape[-1]), self.qdata, self.s_rel, self.s_channel,
                                codebook=self.codebook, correction=None, bias=None,
                                group_size=self.GROUP_SIZE, convrot_groupsize=self.convrot_groupsize,
                                out_dtype=x.dtype)
        return y.reshape(*shape[:-1], y.shape[-1])


# --- ホスト側にピン留めした原本から GPU へ流す仕組み(TE の層・transformer ブロック・VAE で共通) ---------------
#
# 1 モジュール(層・ブロック)の全テンソルを 1 本の匿名メモリにまとめ、cudaHostRegister でその場でピン留めする。
# `Tensor.pin_memory()` は PyTorch のピン留めアロケータが確保サイズを切り上げるため、実体より大きく取られる
# (実測: 8gb 構成で実体 24.6GB に対し 33.1GB)。cudaHostRegister なら切り上げは起きない。
# 登録したバッファは owner ごとに覚えておき、解放前に release_pinned(owner) で登録を解除する。

_PINNED: dict[str, list[torch.Tensor]] = {}
_ALIGN = 256


_FILE_STREAMERS: dict[str, list] = {}


def release_file_streamers(owner: str) -> None:
    for fs in _FILE_STREAMERS.pop(owner, []):
        fs.close()


def release_pinned(owner: str) -> float:
    """owner のピン留めを解除して参照を手放す。解除した量(GB)を返す。"""
    bufs = _PINNED.pop(owner, [])
    cudart = torch.cuda.cudart()
    for b in bufs:
        cudart.cudaHostUnregister(b.data_ptr())
    return sum(b.numel() for b in bufs) / 2**30


def pinned_gb() -> dict[str, float]:
    return {k: round(sum(b.numel() for b in v) / 2**30, 2) for k, v in _PINNED.items()}


def _quant_state_slots(qs, out: list) -> None:
    """bitsandbytes の QuantState が持つテンソル(absmax / code / offset、二重量子化なら state2 も)。"""
    for attr in ("absmax", "code", "offset"):
        if isinstance(getattr(qs, attr, None), torch.Tensor):
            out.append((qs, attr))
    if getattr(qs, "state2", None) is not None:
        _quant_state_slots(qs.state2, out)


def _slots(module: nn.Module) -> list:
    """モジュールが forward で使うテンソルの置き場所 (obj, attr) の一覧。attr == "data" はパラメータ/バッファ本体。"""
    out, seen = [], set()
    for t in list(module.parameters()) + list(module.buffers()):
        if id(t) in seen:
            continue
        seen.add(id(t))
        out.append((t, "data"))
        qs = getattr(t, "quant_state", None)  # bnb の Params4bit
        if qs is not None:
            _quant_state_slots(qs, out)
    return out


def _get(slot) -> torch.Tensor:
    obj, attr = slot
    return obj.data if attr == "data" else getattr(obj, attr)


def _set(slot, value: torch.Tensor) -> None:
    obj, attr = slot
    if attr == "data":
        obj.data = value
    else:
        setattr(obj, attr, value)


def pin_module(module: nn.Module, owner: str) -> list:
    """module の全テンソルを 1 本のピン留めバッファへ移し、各スロットをそのビューに差し替える。
    戻り値は [(slot, pinned_view)](GPU へ流すときの原本)。元の CPU/GPU テンソルは手放される。"""
    slots = _slots(module)
    sizes = [_get(sl).numel() * _get(sl).element_size() for sl in slots]
    offsets, total = [], 0
    for nb in sizes:
        offsets.append(total)
        total += (nb + _ALIGN - 1) // _ALIGN * _ALIGN
    buf = torch.empty(max(total, _ALIGN), dtype=torch.uint8)
    err = torch.cuda.cudart().cudaHostRegister(buf.data_ptr(), buf.numel(), 0)
    if int(err) != 0:
        raise RuntimeError(f"cudaHostRegister に失敗しました({err})")
    _PINNED.setdefault(owner, []).append(buf)
    masters = []
    for sl, off, nb in zip(slots, offsets, sizes):
        src = _get(sl)
        view = buf[off:off + nb].view(src.dtype).view(src.shape)
        view.copy_(src)
        _set(sl, view)
        masters.append((sl, view))
    return masters


def to_device(masters: list, device: torch.device, stream_to_record: torch.cuda.Stream | None = None) -> None:
    """原本から GPU へコピーしてスロットに差す(non_blocking)。stream_to_record があれば record_stream する。"""
    for sl, pinned in masters:
        g = pinned.to(device, non_blocking=True)
        if stream_to_record is not None:
            g.record_stream(stream_to_record)
        _set(sl, g)


def to_host(masters: list) -> None:
    """スロットをピン留めの原本へ戻す(重みは不変なので GPU からの書き戻しはしない)。"""
    for sl, pinned in masters:
        _set(sl, pinned)


def _libc_madvise():
    """Linux の madvise。Windows / macOS には無い(None を返し、呼び出し側は先読み・破棄を行わない)。"""
    import ctypes
    import sys

    if sys.platform != "linux":
        return None
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        return libc
    except (OSError, AttributeError):
        return None


def host_memory_gb() -> tuple[float, float]:
    """(搭載 RAM, 回収できるキャッシュ込みの空き RAM)を GB で返す。Linux は /proc/meminfo、Windows は
    GlobalMemoryStatusEx。読めなければ (1e9, 1e9)。"""
    import sys

    if sys.platform == "win32":
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                        ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                        ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                        ("ullAvailExtendedVirtual", ctypes.c_uint64)]
        ms = _MS(); ms.dwLength = ctypes.sizeof(_MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            return ms.ullTotalPhys / 2**30, ms.ullAvailPhys / 2**30
        return 1e9, 1e9
    total = avail = 1e9
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) / 2**20
            elif line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) / 2**20
    except OSError:
        pass
    return total, avail


def _process_anon_gb() -> float:
    """このプロセスの匿名メモリ(GB)。Linux は RssAnon、Windows は PrivateUsage。"""
    import sys

    if sys.platform == "win32":
        import ctypes

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t)]
        pmc = _PMC(); pmc.cb = ctypes.sizeof(_PMC)
        try:
            h = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                return pmc.PrivateUsage / 2**30
        except (AttributeError, OSError):
            pass
        return 0.0
    try:
        for line in open("/proc/self/status"):
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return 0.0


def _make_readahead(masters: list):
    """mmap 上の原本に対して madvise(MADV_WILLNEED) を出す関数を返す。ディスクからの読み込みを GPU の計算と
    重ねるためのもの(ページキャッシュから流す構成では、これが無いとブロックごとにページフォールトで
    読み込みを待つ)。madvise が無い OS(Windows)では何もしない関数を返す(drop も同様)。"""
    import ctypes
    import mmap as _mmap

    libc = _libc_madvise()
    if libc is None:
        noop = lambda i: None  # noqa: E731
        noop.drop = lambda i: None
        return noop
    page = _mmap.PAGESIZE
    ranges = []
    for entries in masters:
        rs = []
        for _, t in entries:
            addr, nb = t.data_ptr(), t.numel() * t.element_size()
            start = addr - addr % page
            rs.append((start, addr + nb - start))
        ranges.append(rs)

    def readahead(i: int) -> None:
        if 0 <= i < len(ranges):
            for start, length in ranges[i]:
                libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(length), 3)   # MADV_WILLNEED

    def drop(i: int) -> None:
        """使い終わった層のページをページキャッシュから追い出す(MADV_PAGEOUT、Linux 5.4+)。
        TE の層は 1 回しか読まないので、残しておくと transformer のページを押し出すだけ。"""
        if 0 <= i < len(ranges):
            for start, length in ranges[i]:
                libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(length), 21)  # MADV_PAGEOUT
    readahead.drop = drop
    return readahead


def host_cache_budget_gb() -> float:
    """この後ページキャッシュに使える見込みのホスト RAM(GB)。
    MemAvailable(回収できるキャッシュ込みの空き)と、cgroup の上限があれば「上限 − このプロセスの匿名メモリ」の小さい方。
    cgroup の memory.current はキャッシュ込みなので使わない(キャッシュは回収できる)。"""
    avail = host_memory_gb()[1]
    anon = _process_anon_gb()
    try:
        rel = open("/proc/self/cgroup").read().strip().split(":", 2)[-1]   # Linux のみ(無ければ OSError)
        node = Path("/sys/fs/cgroup" + rel)
        while True:
            mx = node / "memory.max"
            if mx.exists():
                v = mx.read_text().strip()
                if v != "max":
                    avail = min(avail, int(v) / 2**30 - anon)
            if node == Path("/sys/fs/cgroup") or node == node.parent:
                break
            node = node.parent
    except (OSError, ValueError):
        pass
    return avail


def pageout_tensors(tensors) -> None:
    """mmap 上のテンソルのページをページキャッシュから捨てる(GPU へコピー済みで、もう読まないもの用)。
    madvise が無い OS では何もしない。"""
    import ctypes
    import mmap as _mmap

    libc = _libc_madvise()
    if libc is None:
        return
    page = _mmap.PAGESIZE
    for t in tensors:
        addr, nb = t.data_ptr(), t.numel() * t.element_size()
        start = addr - addr % page
        libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(addr + nb - start), 21)   # MADV_PAGEOUT


def stream_layers(layers, device: torch.device, owner: str, prefetch: bool = True, pin: bool = True,
                  drop_after_use: bool = False, drop_last: int = 0, file_paths: list[Path] | None = None,
                  file_root: nn.Module | None = None) -> None:
    """layers(順に呼ばれる層の列)をピン留めした原本に置き、forward hook で 1 つずつ GPU へ流す。
    層 i の計算中に i+1 を別 stream で先読みする。

    先読みは copy stream 上で行うので、GPU 側テンソルに `record_stream(計算 stream)` を付け、計算が終わる前に
    アロケータがその領域を次の先読みに再利用しないようにする(これを欠くと出力が黒画像になる、実測)。

    file_paths(pin=False のとき): 原本の safetensors。drop_after_use なら全層を、drop_last>0 なら末尾 K 層を、
    ページキャッシュを経由せずファイルから直接読む(filestream、madvise の無い Windows 向け。Linux でも使える)。
    file_root: 層の safetensors 上の名前を引くためのルートモジュール(層のモジュールパス = ファイル上のキーの接頭辞)。
    """
    masters = []
    for layer in layers:
        layer.to("cpu")
        # pin=False: 今の CPU テンソル(safetensors の mmap ならページキャッシュ)をそのまま原本にする。
        # ピン留めしないのでコピーは同期的になるが、OS が回収できるメモリで済む。
        masters.append(pin_module(layer, owner) if pin else [(sl, _get(sl)) for sl in _slots(layer)])
    n = len(masters)
    readahead = (lambda i: None) if pin else _make_readahead(masters)
    drop_after_use = drop_after_use and not pin
    # drop_last: 末尾 K 層は使い終わったらページを捨てる(mmap で原本がキャッシュに収まりきらないとき、
    # 全部が LRU で押し出されるより、決まった K 層だけを読み直す方が安い)
    drop_last = min(max(int(drop_last), 0), len(masters)) if not pin else 0
    if file_paths and not pin and (drop_after_use or drop_last > 0):
        from . import filestream
        first = 0 if drop_after_use else n - drop_last
        mod_names = {id(m): nm for nm, m in file_root.named_modules()}
        prefixes = [mod_names[id(l)] + "." for l in list(layers)[first:]]
        fs = filestream.install(list(layers)[first:], masters[first:], [Path(p) for p in file_paths], device, owner,
                                prefixes)
        _FILE_STREAMERS.setdefault(owner, []).append(fs)
        if first == 0:
            return                       # 全層ファイル読み
        # 手前の層は従来どおり(ページキャッシュ)。末尾 K 層の最初の読み込みは、少し手前の層の pre hook で始める
        layers, masters, n = list(layers)[:first], masters[:first], first
        kick = max(0, n - 2)

        def kick_hook(mod, args, kwargs):
            fs.prefetch(0)

        layers[kick].register_forward_pre_hook(kick_hook, with_kwargs=True)
        drop_last = 0
    if not prefetch:
        # 先読みなし: 計算と同じ stream でコピーする。解放はストリーム順序で守られるので record_stream が要らず、
        # アロケータの予約が膨らまない(1 回しか通らない TE 向け)。mmap のときはディスクの先読みだけ出す。
        index = {id(m): k for k, m in enumerate(masters)}

        def pre_sync(mod, args, kwargs, _m=None):
            k = index[id(_m)]
            readahead(k + 1); readahead(k + 2); readahead(k + 3)
            to_device(_m, device)

        def post_sync(mod, args, out, _m=None):
            to_host(_m)
            if drop_after_use:
                readahead.drop(index[id(_m)])

        for layer, m in zip(layers, masters):
            layer.register_forward_pre_hook(functools.partial(pre_sync, _m=m), with_kwargs=True)
            layer.register_forward_hook(functools.partial(post_sync, _m=m))
        return
    copy_stream = torch.cuda.Stream(device)
    compute = torch.cuda.current_stream(device)
    events: list = [None] * n

    def prefetch(i: int) -> None:
        with torch.cuda.stream(copy_stream):
            to_device(masters[i], device, compute)
            ev = torch.cuda.Event()
            ev.record(copy_stream)
            events[i] = ev

    def make_pre(i: int):
        def pre(mod, args, kwargs):
            if events[i] is None:
                prefetch(i)
            compute.wait_event(events[i])
            events[i] = None
            if i + 1 < n:
                prefetch(i + 1)
            readahead(i + 2); readahead(i + 3); readahead(i + 4)   # mmap のとき: 数ブロック先をディスクから先読み(非同期)
        return pre

    def make_post(i: int):
        sacrificial = i >= n - drop_last

        def post(mod, args, out):
            to_host(masters[i])
            if sacrificial:
                readahead.drop(i)
        return post

    for i, layer in enumerate(layers):
        layer.register_forward_pre_hook(make_pre(i), with_kwargs=True)
        layer.register_forward_hook(make_post(i))


def install_block_streaming(tr, device: torch.device, pin: bool = True, drop_blocks: str = "auto",
                            file_paths: list[Path] | None = None) -> None:
    """transformer_blocks(50 個、ck-w4a8 で各 ~0.2GB)を先読みしながら流す(8gb 構成)。
    pin=False なら原本は量子化キャッシュの mmap(ページキャッシュ)のまま(RAM の少ない PC 向け)。

    drop_blocks: mmap のとき、使い終わったらページを捨てる末尾ブロック数。"auto" はページキャッシュに使える RAM
    (cgroup の上限も見る)と原本の大きさから、はみ出す分だけを捨てる(残りは毎ステップ読み直さずに済む)。"""
    blocks = tr.transformer_blocks
    k = 0
    if not pin:
        total_gb = sum(_get(sl).numel() * _get(sl).element_size() for b in blocks for sl in _slots(b)) / 2**30
        per_block = total_gb / len(blocks)
        if drop_blocks == "auto":
            margin = 0.5   # TE の通過中の層・埋め込み表の参照ページなど、キャッシュを共有するものの分
            if file_paths:
                margin += 2 * per_block   # file streaming の staging バッファ(ピン留め、匿名メモリ)の分
            avail = host_cache_budget_gb()
            k = int(math.ceil(max(0.0, total_gb + margin - avail) / per_block))
        else:
            k = int(drop_blocks)
        k = min(max(k, 0), len(blocks))
        logger.info("transformer blocks from page cache: %.1fGB, cache budget %.1fGB -> drop %d/%d blocks after use",
                    total_gb, host_cache_budget_gb(), k, len(blocks))
        if _libc_madvise() is None and not file_paths:
            logger.warning("この OS には madvise が無いため、ページキャッシュの先読み・破棄(drop)は行えません。"
                           "H3IMG_TRANSFORMER_HOST=file にすると、はみ出す分をファイルから直接読みます")
    stream_layers(blocks, device, owner="transformer", pin=pin, drop_last=k, file_paths=file_paths, file_root=tr)


def _replace_quant_targets_with_ck_shells(tr) -> None:
    """キャッシュ再ロード用: 量子化対象の nn.Linear(meta)を CKW4A8Linear(meta)へ置換する。"""
    for name, mod in list(tr.named_modules()):
        if isinstance(mod, nn.Linear) and _is_quant_target(name + ".weight"):
            _set_module(tr, name, CKW4A8Linear(mod.in_features, mod.out_features))


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
                quant: str = "int8wo", lora_path: str | None = None, stream_blocks: bool = False,
                pin_blocks: bool = True, drop_blocks: str = "auto"):
    """bf16 シャードから量子化済み transformer を組み立てる(1 テンソルずつ GPU へ)。

    quant="int8wo": torchao Int8WeightOnly(キャッシュ可。eager では bf16 へ逆量子化してから
    GEMM するため bnb より遅い、2026-11-11 実測)。
    quant="bnb-int8": bitsandbytes LLM.int8(真の int8 GEMM、速い。キャッシュ非対応で毎回構築)。
    quant="ck-w4a8": comfy-kitchen W4A8(約 11GiB、キャッシュ可)。
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
                        tab = AdaLNTable(keys.clone(), table, hidden)
                        _replace_adaln(tr, bi, tab.to("cpu") if stream_blocks else tab)
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
                    elif quant == "ck-w4a8":
                        lin = CKW4A8Linear.from_weight(w, device)
                        if stream_blocks and mod_name.startswith("transformer_blocks."):
                            lin = lin.to("cpu")  # ブロックは GPU に溜めない(12GB 級ではビルド中に溢れる)
                        _set_module(tr, mod_name, lin)
                        del w
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
    if stream_blocks:
        paths = [cache_dir / STATE_FILENAME_ST] if file_blocks and (cache_dir / STATE_FILENAME_ST).exists() else None
        if file_blocks and paths is None:
            logger.warning("このキャッシュは safetensors ではないので file streaming は使えません(mmap で続行)")
        install_block_streaming(tr, device, pin=pin_blocks, drop_blocks=drop_blocks, file_paths=paths)
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
    _save_state(sd, tmp)
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
    tmp.rename(cache_dir)
    size = sum(p.stat().st_size for p in cache_dir.rglob("*") if p.is_file()) / 2**30
    logger.info("cache saved: %s (%.1fGiB)", cache_dir, size)


def _save_state(sd: dict, dir_: Path) -> Path:
    """素のテンソルだけなら safetensors(mmap で読める・ファイル位置が分かる=file streaming できる)、
    テンソルのサブクラス(torchao の Int8Tensor)を含むなら torch.save。"""
    if all(type(v) is torch.Tensor for v in sd.values()):
        from safetensors.torch import save_file
        out = dir_ / STATE_FILENAME_ST
        save_file({k: v.contiguous() for k, v in sd.items()}, str(out))
        return out
    out = dir_ / STATE_FILENAME
    torch.save(sd, str(out))
    return out


def _load_state(cache_dir: Path) -> dict:
    """キャッシュの state_dict を mmap で読む。古い .pt(素のテンソルだけ)は safetensors へ変換してから読む。"""
    st = cache_dir / STATE_FILENAME_ST
    if st.exists():
        from safetensors.torch import load_file
        return load_file(str(st), device="cpu")     # safetensors 0.4+ はゼロコピー(mmap)
    pt = cache_dir / STATE_FILENAME
    sd = torch.load(str(pt), map_location="cpu", weights_only=False, mmap=True)
    if all(type(v) is torch.Tensor for v in sd.values()):
        t0 = time.time()
        out = _save_state(sd, cache_dir)
        del sd
        pt.unlink()
        logger.info("cache converted to safetensors: %s (%.0fs, old .pt removed)", out, time.time() - t0)
        from safetensors.torch import load_file
        return load_file(str(out), device="cpu")
    return sd


def cache_exists(cache_dir: Path) -> bool:
    return (cache_dir / STATE_FILENAME).exists() or (cache_dir / STATE_FILENAME_ST).exists()


def load_cache(cache_dir: Path, device: torch.device, stream_blocks: bool = False, pin_blocks: bool = True,
               drop_blocks: str = "auto", file_blocks: bool = False):
    from accelerate import init_empty_weights
    from diffusers import MiniMaxH3ImageTransformer2DModel as Cls

    t0 = time.time()
    config = Cls.load_config(str(cache_dir))
    with init_empty_weights():
        tr = Cls.from_config(config)
    # mmap で読む: 量子化キャッシュ全体(11〜19GiB)をホスト RAM へ展開せず、必要なページだけ読む
    sd = _load_state(cache_dir)
    hidden = config["hidden_size"]
    if any(k.endswith(".qdata") for k in sd):
        _replace_quant_targets_with_ck_shells(tr)
    for i in range(len(tr.transformer_blocks)):
        k = f"transformer_blocks.{i}.adaln_proj."
        _replace_adaln(tr, i, AdaLNTable(sd[k + "keys"], sd[k + "table"], hidden))
    # 量子化済み Linear は meta の nn.Parameter(bf16)に Int8Tensor を assign で差し替える。
    tr.load_state_dict(sd, assign=True)
    if stream_blocks:
        others = []
        for name, child in tr.named_children():
            if name != "transformer_blocks":
                others += [_get(sl) for sl in _slots(child)]   # GPU へ移す前の mmap 上のテンソル
                child.to(device)
        if not pin_blocks:
            pageout_tensors(others)   # GPU に置いた分のページはもう読まない。キャッシュの枠をブロックに譲る
        paths = [cache_dir / STATE_FILENAME_ST] if file_blocks and (cache_dir / STATE_FILENAME_ST).exists() else None
        if file_blocks and paths is None:
            logger.warning("このキャッシュは safetensors ではないので file streaming は使えません(mmap で続行)")
        install_block_streaming(tr, device, pin=pin_blocks, drop_blocks=drop_blocks, file_paths=paths)
    else:
        tr = tr.to(device)
    install_forward_wrapper(tr)
    tr.eval()
    logger.info("cache loaded: %s %.1fs alloc=%.2fGB", cache_dir, time.time() - t0,
                torch.cuda.memory_allocated(device) / 2**30)
    return tr


def load_transformer(repo: str, pipe, device: torch.device, cache_dir: Path, quant: str = "int8wo",
                     lora_path: str | None = None, stream_blocks: bool = False, pin_blocks: bool = True,
                     drop_blocks: str = "auto", file_blocks: bool = False):
    """キャッシュがあれば読む、無ければ構築して保存する(bnb-int8 は毎回構築)。"""
    keys = timestep_keys(pipe)
    logger.info("AdaLN timestep keys: %s", keys.tolist())
    if quant == "bnb-int8":
        return build_fresh(repo, device, keys, verify=True, quant=quant, lora_path=lora_path,
                           stream_blocks=stream_blocks, pin_blocks=pin_blocks, drop_blocks=drop_blocks)
    if cache_exists(cache_dir):
        tr = load_cache(cache_dir, device, stream_blocks=stream_blocks, pin_blocks=pin_blocks, drop_blocks=drop_blocks,
                        file_blocks=file_blocks)
        cached = next(m for m in tr.modules() if isinstance(m, AdaLNTable)).keys.cpu()
        if not torch.equal(cached, keys):
            raise RuntimeError(f"キャッシュの時刻集合 {cached.tolist()} が現在の {keys.tolist()} と異なります。"
                               f"{cache_dir} を削除して作り直してください")
        return tr
    tr = build_fresh(repo, device, keys, verify=True, quant=quant, lora_path=lora_path,
                     stream_blocks=stream_blocks, pin_blocks=pin_blocks, drop_blocks=drop_blocks)
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
