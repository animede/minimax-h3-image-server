"""ファイルから直接 GPU へ流す(ページキャッシュを汚さない)層ストリーミング。Linux / Windows 共通。

mmap 方式(pruned_image.stream_layers の pin=False)は「ページキャッシュに原本を置き、使い終わったら madvise で捨てる」
前提で、Windows には madvise が無い。こちらは原本をページキャッシュに置かず、層ごとに

  ファイル --(バックグラウンドスレッドで read)--> ピン留めした staging バッファ(2 本で交互) --(非同期 H2D)--> GPU

と流す。読み込みは次の層を先に始める(ディスクと GPU 計算を重ねる)。ページキャッシュを汚さないために、
Linux は読んだ範囲を posix_fadvise(DONTNEED) で捨て、Windows は O_SEQUENTIAL(FILE_FLAG_SEQUENTIAL_SCAN)で開く。

テンソルのファイル内オフセットは、safetensors のヘッダから求める。mmap で読んだテンソルは
「data_ptr − マッピングの先頭」がファイル内オフセットなので、名前が分かる 1 本から先頭を逆算して全テンソルに使う
(ロード時の実体は mmap のまま、ストリーミング時はファイルから読む)。
"""

from __future__ import annotations

import json
import logging
import os
import struct
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch

logger = logging.getLogger("h3image.filestream")

_ALIGN = 4096   # staging バッファ内のテンソル境界(dtype の view が取れるように整列)


def safetensors_layout(path: Path) -> tuple[int, dict[str, tuple[int, int]]]:
    """safetensors のヘッダを読み、(データ部の先頭オフセット, {名前: (ファイル内オフセット, バイト数)}) を返す。"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    data0 = 8 + n
    layout = {k: (data0 + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0])
              for k, v in header.items() if k != "__metadata__"}
    return data0, layout


class FileRegion:
    """ファイル内の 1 テンソル分の領域と、その GPU 側の形。"""
    __slots__ = ("path", "offset", "nbytes", "dtype", "shape")

    def __init__(self, path: Path, offset: int, nbytes: int, dtype: torch.dtype, shape: tuple[int, ...]):
        self.path, self.offset, self.nbytes, self.dtype, self.shape = path, offset, nbytes, dtype, shape


_QUANT_STATE_KEYS = {"absmax": ".absmax", "code": ".quant_map"}     # bitsandbytes の QuantState → safetensors の接尾辞


def named_slots(module: torch.nn.Module, prefix: str) -> list:
    """pruned_image._slots() と同じ順序で、各スロットの safetensors 上の名前を返す(無ければ None)。
    bnb の Params4bit は weight 本体に加えて quant_state の absmax(`<name>.absmax`)と code(`<name>.quant_map`)。
    offset / 二重量子化の state2 は名前が無い(小さいので常駐させる)。"""
    names, seen = [], set()
    entries = list(module.named_parameters()) + list(module.named_buffers())
    for name, t in entries:
        if id(t) in seen:
            continue
        seen.add(id(t))
        names.append(prefix + name)
        qs = getattr(t, "quant_state", None)
        if qs is not None:
            _quant_state_names(qs, prefix + name, names)
    return names


def _quant_state_names(qs, base: str, out: list) -> None:
    for attr in ("absmax", "code", "offset"):
        if isinstance(getattr(qs, attr, None), torch.Tensor):
            out.append(base + _QUANT_STATE_KEYS[attr] if attr in _QUANT_STATE_KEYS else None)
    if getattr(qs, "state2", None) is not None:       # 二重量子化の state2: ファイル上の名前が無い → 常駐
        for attr in ("absmax", "code", "offset"):
            if isinstance(getattr(qs.state2, attr, None), torch.Tensor):
                out.append(None)


_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "U8": torch.uint8, "I8": torch.int8,
           "I32": torch.int32, "I64": torch.int64, "F8_E4M3": torch.float8_e4m3fn, "BOOL": torch.bool}


def resolve_regions(paths: list[Path], masters: list[list[tuple]], names: list[list[str | None]]
                    ) -> list[list[FileRegion | None]]:
    """各スロットの safetensors 上の位置を名前で引く。masters[i] = [(slot, 今のテンソル)]、names[i] = 対応する名前。
    名前が無い / ファイルに無い / 大きさが合わないスロットは None(呼び出し側が GPU に常駐させる)。
    mmap のアドレスに頼らないので、safetensors の読み方が違う環境(Windows 等)でも使える。"""
    index: dict[str, tuple[Path, int, int, torch.dtype, tuple]] = {}
    for path in paths:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        data0 = 8 + n
        for k, v in header.items():
            if k != "__metadata__":
                index[k] = (path, data0 + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0],
                            _DTYPES.get(v["dtype"]), tuple(v["shape"]))
    out, missing = [], []
    for lay, lay_names in zip(masters, names):
        regs = []
        for (_, t), name in zip(lay, lay_names):
            ent = index.get(name) if name else None
            nb = t.numel() * t.element_size()
            if ent is None or ent[2] != nb or ent[3] != t.dtype:
                regs.append(None)
                if name:
                    missing.append(name)
            else:
                regs.append(FileRegion(ent[0], ent[1], nb, t.dtype, tuple(t.shape)))
        out.append(regs)
    if missing:
        logger.warning("file streaming: %d tensors not found in the files (kept resident on the GPU), e.g. %s",
                       len(missing), missing[:3])
    return out


_CHUNK = 32 << 20        # 並列読みの単位。1 スレッド 256MB 逐次だと 1.6GB/s、8 スレッド × 32MB で 4GB/s(Gen4 NVMe、実測)
_READ_THREADS = 8


class _Reader:
    """ファイルの指定範囲をピン留めバッファへ読む。32MB ずつ複数スレッドで並列に読み、
    ページキャッシュを汚さない開き方 / 読み方を OS ごとに選ぶ。"""

    def __init__(self):
        self.pool = ThreadPoolExecutor(_READ_THREADS, thread_name_prefix="filestream-read")
        self.local = threading.local()      # Windows(pread 無し)はスレッドごとに fd を持つ(lseek+read が競合しないように)
        self.shared: dict[Path, int] = {}
        self.lock = threading.Lock()
        self.has_pread = hasattr(os, "preadv")

    def _open(self, path: Path) -> int:
        flags = os.O_RDONLY
        if sys.platform == "win32":
            flags |= os.O_BINARY | getattr(os, "O_SEQUENTIAL", 0)
        return os.open(str(path), flags)

    def _fd(self, path: Path) -> int:
        if self.has_pread:
            with self.lock:
                fd = self.shared.get(path)
                if fd is None:
                    fd = self.shared[path] = self._open(path)
            return fd
        fds = getattr(self.local, "fds", None)
        if fds is None:
            fds = self.local.fds = {}
        fd = fds.get(path)
        if fd is None:
            fd = fds[path] = self._open(path)
        return fd

    def _read_chunk(self, path: Path, mv: memoryview, offset: int) -> None:
        fd, got, n = self._fd(path), 0, len(mv)
        while got < n:
            if self.has_pread:
                k = os.preadv(fd, [mv[got:]], offset + got)
            else:
                os.lseek(fd, offset + got, os.SEEK_SET)
                data = os.read(fd, n - got)            # Windows: 1 回余計にコピーする
                mv[got:got + len(data)] = data
                k = len(data)
            if k <= 0:
                raise OSError(f"短い読み込み: {path.name} offset={offset} got={got}/{n}")
            got += k

    def read_layer(self, buf: torch.Tensor, plan: list) -> None:
        """plan = [(slot, FileRegion, staging 内オフセット)] を全部まとめて並列に読む(テンソル単位で待つと、
        小さいテンソルが多い TE の層では並列度が 1〜2 に落ちて 1.6GB/s 止まりになる、実測)。"""
        mv = memoryview(buf.numpy())
        futs = [self.pool.submit(self._read_chunk, r.path, mv[pos + c:pos + min(c + _CHUNK, r.nbytes)], r.offset + c)
                for _, r, pos in plan for c in range(0, r.nbytes, _CHUNK)]
        for f in futs:
            f.result()
        if hasattr(os, "posix_fadvise"):
            # 読んだ範囲をページキャッシュから捨てる(キャッシュに残したい transformer のブロックを押し出さない)
            for _, r, _ in plan:
                os.posix_fadvise(self._fd(r.path), r.offset, r.nbytes, os.POSIX_FADV_DONTNEED)

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        for fd in self.shared.values():
            os.close(fd)
        self.shared.clear()


class FileStreamer:
    """layers[i] のテンソル群を、ファイルからピン留め staging バッファ経由で GPU へ流す。

    - staging は 2 本(ダブルバッファ)。層 i を GPU へコピーしている間に、層 i+1 をもう片方へ読む。
    - pre hook: 層 i の読み込み完了を待ち、copy stream で H2D(record_stream で計算完了まで領域を保護)、
      層 i+1 の読み込みをスレッドへ投げる。
    - post hook: GPU 側テンソルへの参照を外す(原本はファイルなので書き戻しはしない)。
    - staging バッファの再利用は、そのバッファからの H2D 完了イベントを待ってから行う。
    """

    def __init__(self, layer_regions: list[list[tuple]], device: torch.device, label: str):
        # layer_regions[i] = [(slot, FileRegion), ...] → self.layers[i] = [(slot, FileRegion, staging 内オフセット)]
        self.device, self.label = device, label
        self.layers, max_bytes = [], 0
        for lay in layer_regions:
            plan, pos = [], 0
            for slot, r in lay:
                plan.append((slot, r, pos))
                pos += (r.nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
            self.layers.append(plan)
            max_bytes = max(max_bytes, pos)
        self.reader = _Reader()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"filestream-{label}")
        # staging は torch.empty(匿名メモリ)を cudaHostRegister でピン留めする。pin_memory()(cudaHostAlloc)だと
        # cgroup には "file" として計上され、ページキャッシュの予算計算(匿名メモリを差し引く)から漏れて
        # transformer のキャッシュを押し出してしまう(実測)。
        self.bufs = []
        for _ in range(2):
            buf = torch.empty(max(max_bytes, _ALIGN), dtype=torch.uint8)
            err = torch.cuda.cudart().cudaHostRegister(buf.data_ptr(), buf.numel(), 0)
            if int(err) != 0:
                raise RuntimeError(f"cudaHostRegister に失敗しました({err})")
            self.bufs.append(buf)
        self.buf_free: list[torch.cuda.Event | None] = [None, None]   # そのバッファからの H2D 完了イベント
        self.pending: dict[int, tuple[int, Future]] = {}              # 層 index -> (buffer index, read future)
        self.copy_stream = torch.cuda.Stream(device)
        self.compute = torch.cuda.current_stream(device)
        self.n = len(layer_regions)
        self.pinned_gb = 2 * max_bytes / 2**30
        self._wait = self._rt = self._t0 = 0.0
        logger.info("file streaming (%s): %d layers read directly from file, staging 2 x %.2fGB pinned",
                    label, self.n, max_bytes / 2**30)

    # -- 読み込み ------------------------------------------------------------
    def _read_layer(self, i: int, b: int) -> None:
        ev = self.buf_free[b]
        if ev is not None:
            ev.synchronize()           # このバッファからの前回の H2D が終わるまで待つ
        buf = self.bufs[b]
        _t = time.time()
        self.reader.read_layer(buf, self.layers[i])
        self._rt += time.time() - _t

    def prefetch(self, i: int) -> None:
        if 0 <= i < self.n and i not in self.pending:
            b = i % 2
            self.pending[i] = (b, self.pool.submit(self._read_layer, i, b))

    # -- GPU へ -----------------------------------------------------------------
    def load_to_gpu(self, i: int) -> None:
        if i not in self.pending:
            self.prefetch(i)
        b, fut = self.pending.pop(i)
        _t = time.time()
        fut.result()
        self._wait += time.time() - _t
        if i == 0:
            self._t0 = _t
        buf = self.bufs[b]
        with torch.cuda.stream(self.copy_stream):
            for slot, r, pos in self.layers[i]:
                g = buf[pos:pos + r.nbytes].view(r.dtype).view(r.shape).to(self.device, non_blocking=True)
                g.record_stream(self.compute)
                _set(slot, g)
            ev = torch.cuda.Event()
            ev.record(self.copy_stream)
        self.buf_free[b] = ev
        self.compute.wait_event(ev)

    def release(self, i: int, masters: list) -> None:
        for (slot, _, _), (_, master) in zip(self.layers[i], masters):
            _set(slot, master)
        if i == self.n - 1:
            logger.info("file streaming (%s) pass: %.2fs, %.2fs waiting for reads, %.2fs/layer read",
                         self.label, time.time() - self._t0, self._wait, self._rt / max(self.n, 1))
            self._wait = self._rt = 0.0

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        self.reader.close()
        torch.cuda.synchronize(self.device)
        for buf in self.bufs:
            torch.cuda.cudart().cudaHostUnregister(buf.data_ptr())
        self.bufs = []


def _set(slot, value: torch.Tensor) -> None:
    obj, attr = slot
    if attr == "data":
        obj.data = value
    else:
        setattr(obj, attr, value)


def install(layers, masters: list[list[tuple]], paths: list[Path], device: torch.device, label: str,
            prefixes: list[str]) -> FileStreamer:
    """layers(masters / prefixes と同じ長さ)に file streaming の hook を付ける。
    masters[i] = [(slot, 今の原本テンソル)]、prefixes[i] = その層の safetensors 上の名前の接頭辞("...layers.7.")。
    ファイルに無い小さなテンソルは GPU に常駐させる。原本(CPU 側)は 0 要素のプレースホルダに差し替えて手放す
    (RAM に展開されていた重み・mmap の参照を解放する。以後は毎回ファイルから読む)。"""
    names = [named_slots(layer, pfx) for layer, pfx in zip(layers, prefixes)]
    regions = resolve_regions(paths, masters, names)
    layer_regions = []
    for i in range(len(masters)):
        plan = []
        for (slot, t), r in zip(masters[i], regions[i]):
            if r is None:
                _set(slot, t.to(device))          # 常駐
            else:
                plan.append((slot, r))
                _set(slot, torch.empty(0, dtype=t.dtype))
        layer_regions.append(plan)
    masters = [[(slot, torch.empty(0, dtype=r.dtype)) for slot, r in plan] for plan in layer_regions]
    fs = FileStreamer(layer_regions, device, label)
    n = len(layers)

    def make_pre(i: int):
        def pre(mod, args, kwargs):
            fs.load_to_gpu(i)
            if i + 1 < n:
                fs.prefetch(i + 1)
        return pre

    def make_post(i: int):
        def post(mod, args, out):
            fs.release(i, masters[i])
        return post

    for i in range(n):
        layers[i].register_forward_pre_hook(make_pre(i), with_kwargs=True)
        layers[i].register_forward_hook(make_post(i))
    return fs
