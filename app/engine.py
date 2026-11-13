"""MiniMax-H3-Image 生成エンジン。

構成(README「低 VRAM 化・高速化の方法」「調査・実測の記録」参照):
- transformer: AdaLN 事前計算テーブル + torchao int8wo(app/pruned_image.py)。steps 2/3 は pDMD
  LoRA をマージ済みの別キャッシュ。常駐 ~19GB、1024² 1.8s / 2048² 4.1s(3-step)。
- text_encoder: H3 動画用 prequant(nf4・51層、17GB)を流用。GPU が 2 枚あれば cuda:1 へ置き、
  hidden_states[50] だけをホスト経由で戻す(このマシンは GPU 間 P2P コピーが黙って全ゼロを返す)。
- VAE: bf16 2.3GB。
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
import logging
import threading
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from . import pruned_image
from .config import STEP_PRESETS, TEXT_ENCODER_LAYER, Settings

logger = logging.getLogger("h3image.engine")


class Interrupted(RuntimeError):
    pass


def _host_ram_gb() -> float:
    """ホストの搭載メモリ(GB)。Linux は /proc/meminfo、Windows は GlobalMemoryStatusEx。読めなければ大きい値
    (従来どおり pinned を選ぶ)。"""
    return pruned_image.host_memory_gb()[0]


class _CpuEmbedding(torch.nn.Module):
    """CPU 常駐の埋め込み表。参照結果だけを GPU へ送る(TE ストリーミング時の 1.45GB 削減)。"""

    def __init__(self, inner: torch.nn.Embedding, device: torch.device) -> None:
        super().__init__()
        self.inner = inner
        self.device = device
        self.weight = inner.weight  # 互換(dtype 参照など)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.inner(ids.to("cpu")).to(self.device, non_blocking=True)


class _HiddenOnly:
    """text_encoder.model の返り値の代わり(encoder step は .hidden_states[50] しか読まない)。"""
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _UnusedHead(torch.nn.Module):
    """lm_head の代わり。H3-Image は hidden_states[50] しか使わないので、呼ばれたら設計ミスとして落とす。"""

    def forward(self, x):
        raise RuntimeError("text_encoder の lm_head は不要として削除済みです(hidden_states のみ使用)")


@dataclass(frozen=True)
class Profile:
    """VRAM 予算に応じた構成。need_same = TE が transformer と同じ GPU のときに要る量(GB)、
    need_split = TE が別 GPU のとき transformer 側に要る量。いずれも torch の割り当て peak の実測 + 0.3GB。
    予算は CUDA コンテキスト作成後の空きから測るのでコンテキスト分は含めない。余裕は予算側で
    H3IMG_VRAM_RESERVE_GB として別に取る。"""
    name: str
    quant: str
    te_stream: bool
    transformer_stream: bool
    need_same: float
    need_split: float


# 上から順に、予算が足りる最初のものを選ぶ(torch peak の実測: 48gb 38.4 / 32gb 30.2 / 16gb 13.0 / 8gb 2.9)
PROFILES: tuple[Profile, ...] = (
    Profile("48gb", "int8wo", False, False, need_same=38.7, need_split=21.7),
    Profile("32gb", "ck-w4a8", False, False, need_same=30.5, need_split=13.6),
    Profile("16gb", "ck-w4a8", True, False, need_same=13.3, need_split=13.6),
    Profile("8gb", "ck-w4a8", True, True, need_same=3.2, need_split=3.2),
)
VAE_GPU_GB = 1.45               # VAE を GPU に常駐させるのに要る量(重み 1.42GB)
TE_PINNED_MIN_HOST_GB = 48.0    # TE_HOST=auto でピン留め(速い)を選ぶホスト RAM の下限。未満は mmap(必須 RAM が約 13GB 少ない)
EMBED_GPU_GB = 1.5              # 埋め込み表(bf16 1.45GB)を GPU に置くのに要る量


class Engine:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.lock = threading.Lock()         # ロード/解放/生成の直列化
        self.device = torch.device(settings.device)
        if settings.te_device:
            self.te_device = torch.device(settings.te_device)
        else:
            self.te_device = torch.device("cuda:1") if torch.cuda.device_count() > 1 else self.device
        self.pipe = None
        self.te = None
        self.vae = None
        self.transformer = None
        self.loaded_steps: int | None = None
        self._default_sigmas: list[float] | None = None
        self._repo: str | None = None
        self.profile: Profile | None = None       # ロード時に決まる(unload で None に戻る)
        self._profile_floor = 0                   # OOM フォールバックで下げた段(PROFILES の index の下限)

    # -- 状態 ------------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self.transformer is not None

    def status(self) -> dict:
        return {
            "loaded": self.loaded,
            "steps": self.loaded_steps,
            "profile": self.profile.name if self.profile else None,
            "vram_budget_gb": self.s.vram_gb,
            "transformer_quant": self.profile.quant if self.profile else self.s.transformer_quant,
            "device": str(self.device),
            "te_device": str(self.te_device),
            "te_loaded": self.te is not None,
            "te_streamed": self.te_streamed,
            "transformer_streamed": self.transformer_streamed,
            "host_pinned_gb": pruned_image.pinned_gb(),
            "te_host": getattr(self, "te_host", None),
            "transformer_host": getattr(self, "transformer_host", None),
            "vae_on_gpu": getattr(self, "vae_on_gpu", None),
            "embed_on_gpu": getattr(self, "embed_on_gpu", None),
            "vram_alloc_gb": round(torch.cuda.memory_allocated(self.device) / 2**30, 2),
        }

    # -- ロード ----------------------------------------------------------------

    def _resolve_repo(self) -> str:
        if self._repo:
            return self._repo
        if self.s.model_dir:
            self._repo = self.s.model_dir
            return self._repo
        from huggingface_hub.constants import HF_HUB_CACHE
        root = Path(HF_HUB_CACHE) / ("models--" + self.s.model_id.replace("/", "--")) / "snapshots"
        snaps = sorted(root.glob("*"), key=lambda p: p.stat().st_mtime) if root.is_dir() else []
        # 意図的に一部だけ落としたスナップショットを直接参照する(新しい huggingface_hub は
        # local_files_only でも欠けたファイルを検査して例外にするため snapshot_download は使わない)
        for snap in reversed(snaps):
            if (snap / "transformer_diffusers" / "diffusion_pytorch_model.safetensors.index.json").exists() \
                    and (snap / "vae_diffusers" / "config.json").exists():
                self._repo = str(snap)
                return self._repo
        raise RuntimeError(
            f"{self.s.model_id} のスナップショットが見つかりません。scripts/prepare_models.py を実行して "
            "transformer_diffusers / vae_diffusers / scheduler / tokenizer / processor / *.json を取得してください")

    def _build_pipeline(self):
        from diffusers import ModularPipeline

        repo = self._resolve_repo()
        pipe = ModularPipeline.from_pretrained(repo)
        # 実行デバイスを固定する(既定の解決は component 名の並び順で text_encoder(cuda:1)を拾いうる)
        dev = self.device
        type(pipe)._execution_device = property(lambda self: dev)
        pipe.load_components(names=["tokenizer", "processor", "scheduler"])
        self._default_sigmas = list(pipe.config.sample_sigmas)
        return pipe

    def _load_te(self):
        from transformers import AutoModelForImageTextToText

        t0 = time.time()
        if not (self.s.te_dir / "config.json").exists():
            raise RuntimeError(f"text_encoder キャッシュが見つかりません: {self.s.te_dir}(H3IMG_TE_DIR)")
        stream = self.te_device == self.device and self._te_stream_enabled()
        # ストリーミング時は最初から CPU へ読む(GPU へ全部載せてから退避すると、ロードの時点で 16GB を超える)
        te = AutoModelForImageTextToText.from_pretrained(str(self.s.te_dir), dtype=torch.bfloat16,
                                                         device_map="cpu" if stream else str(self.te_device))
        self._keep_only_layer_output(te)
        if self.te_device != self.device:
            self._route_text_encoder(te)
        elif stream:
            self._stream_text_encoder(te)
        logger.info("text_encoder loaded (%d layers) on %s in %.1fs (stream=%s)",
                    te.config.text_config.num_hidden_layers, self.te_device, time.time() - t0, self.te_streamed)
        return te

    te_streamed: bool = False
    transformer_streamed: bool = False

    # -- VRAM 予算 → 構成 ----------------------------------------------------------

    def _vram_budget_gb(self) -> tuple[float, float, float]:
        """(予算, 要求, 実際の空き)。予算 = min(H3IMG_VRAM_GB または総容量, 今の空き - 予備)。
        予備(H3IMG_VRAM_RESERVE_GB、既定 1GB)は画面表示や他のアプリのために必ず空けておく分。"""
        free, total = torch.cuda.mem_get_info(self.device)
        free_gb, total_gb = free / 2**30, total / 2**30
        requested = float(self.s.vram_gb) if self.s.vram_gb != "auto" else total_gb
        return min(requested, free_gb - self.s.vram_reserve_gb), requested, free_gb

    def _resolve_profile(self) -> Profile:
        budget, requested, free_gb = self._vram_budget_gb()
        split = self.te_device != self.device
        chosen = PROFILES[-1]
        for i, p in enumerate(PROFILES):
            if i < self._profile_floor:
                continue
            if budget >= (p.need_split if split else p.need_same):
                chosen = p
                break
        else:
            logger.warning("空き VRAM %.1fGB は最小構成 %s の目安 %.1fGB を下回っています(そのまま試みます)",
                           budget, chosen.name, chosen.need_split if split else chosen.need_same)
        # 明示指定があれば上書き
        quant = self.s.transformer_quant if self.s.transformer_quant != "auto" else chosen.quant
        te_stream = chosen.te_stream if self.s.te_stream == "auto" else self.s.te_stream == "1"
        tr_stream = chosen.transformer_stream if self.s.transformer_stream == "auto" else self.s.transformer_stream == "1"
        if split:
            te_stream = False  # TE が別 GPU ならストリーミングは不要(そちらに常駐)
        prof = Profile(chosen.name, quant, te_stream, tr_stream, chosen.need_same, chosen.need_split)
        self._cap_allocator(budget)
        # 予算に余りがあれば、ストリーミング構成でも VAE と埋め込み表を GPU に置く(往復とホスト RAM が減る)
        headroom = budget - (prof.need_split if split else prof.need_same)
        self.vae_on_gpu = not te_stream or headroom >= VAE_GPU_GB
        if te_stream and self.vae_on_gpu:
            headroom -= VAE_GPU_GB
        self.embed_on_gpu = te_stream and headroom >= EMBED_GPU_GB
        # RAM が少ないときの原本は file(層ごとにファイルから直接ピン留めバッファへ並列読み。OS を問わず動き、
        # mmap + madvise より速い: 12GB 上限で 10.5s vs 12.3s、実測)
        auto_host = "pinned" if _host_ram_gb() >= TE_PINNED_MIN_HOST_GB else "file"
        self.te_host = (self.s.te_host if self.s.te_host != "auto" else auto_host) if te_stream else None
        self.transformer_host = (self.s.transformer_host if self.s.transformer_host != "auto" else auto_host) \
            if tr_stream else None
        logger.info("VRAM profile: %s (budget %.1fGB = min(requested %.1f, free %.1f - reserve %.1f), TE %s) -> quant=%s "
                    "te_stream=%s(%s) transformer_stream=%s(%s) vae_on_gpu=%s embed_on_gpu=%s", prof.name, budget,
                    requested, free_gb, self.s.vram_reserve_gb, "split" if split else "same GPU", prof.quant,
                    prof.te_stream, self.te_host, prof.transformer_stream, self.transformer_host, self.vae_on_gpu,
                    self.embed_on_gpu)
        return prof

    def _cap_allocator(self, budget_gb: float) -> None:
        """PyTorch のキャッシングアロケータが予約できる量を予算に制限する(予算はコンテキスト作成後の空き - 予備)。

        先読み(record_stream)は解放が計算の完了まで遅れるので、上限が無いとアロケータは空いている VRAM を
        全部予約してしまう(実測: 空き 7.3GB の GPU で割り当て 3.1GB に対しプロセスが 7.3GB を占有)。
        上限に達するとアロケータはキャッシュを解放してから再確保するので、同じ GPU の他のプロセスを圧迫しない。
        予算が GPU 全体と同じ(既定の auto で空いている大きい GPU)ならほぼ無制限のまま。
        """
        total_gb = torch.cuda.get_device_properties(self.device).total_memory / 2**30
        frac = min(1.0, max(0.05, budget_gb / total_gb))
        torch.cuda.set_per_process_memory_fraction(frac, self.device)
        logger.info("allocator cap: %.1fGB (budget %.1fGB)", frac * total_gb, budget_gb)

    def _transformer_stream_enabled(self) -> bool:
        return self.profile.transformer_stream

    def _te_stream_enabled(self) -> bool:
        return self.profile.te_stream

    def _stream_text_encoder(self, te) -> None:
        """decoder 51 層をピン留めした原本(cudaHostRegister)に置き、1 層ずつ GPU へ流す。

        bnb nf4 の量子化状態(QuantState の absmax / code)も一緒に流す。GPU からの書き戻しはしない。
        lm_head(1.45GB)は hidden_states[50] しか読まないので捨て、embed_tokens(bf16 1.45GB)は CPU で引く。
        TE は device_map="cpu" で読んでいるので、ここで原本をピン留めバッファへ移すと safetensors の mmap は
        参照を失って解放される(埋め込みも複製して mmap から切り離す)。
        """
        dev = self.device
        lm = te.model.language_model
        t0 = time.time()
        te.model.visual.to(dev)
        lm.norm.to(dev)
        lm.rotary_emb.to(dev)
        # 先読みはしない: TE は 1 回しか通らず、先読みの効果は実測で 0.1s 程度(誤差の範囲)。先読みすると
        # record_stream で解放が遅れ、アロケータの予約が膨らむ(8gb 構成で 4.2GB → 6.6GB)。
        # te_host == "mmap": 原本は safetensors の mmap(ページキャッシュ)のまま。ピン留めしないので必須の RAM が
        # 約 13GB 減る(OS が回収できるメモリになる)。"pinned": cudaHostRegister でピン留めした原本(速いが常に使う)。
        # mmap のときは使い終わった層のページをその場で捨てる(1 回しか読まないので、残すと transformer の
        # ページキャッシュを押し出すだけ。RAM の少ない PC で transformer を読み直さずに済むようにする)
        # "file": ページキャッシュを経由せず、層ごとにファイルから直接ピン留めバッファへ読んで流す(Windows の既定)
        pruned_image.stream_layers(lm.layers, dev, owner="te", prefetch=False, pin=self.te_host == "pinned",
                                   drop_after_use=self.te_host in ("mmap", "file"),
                                   file_paths=sorted(self.s.te_dir.glob("*.safetensors")) if self.te_host == "file" else None,
                                   file_root=te)
        te.lm_head = _UnusedHead()
        emb = lm.embed_tokens
        if self.embed_on_gpu:
            emb.to(dev)
        else:
            if self.te_host == "pinned":   # mmap から切り離す(pinned 方式ではファイルのマップを残さない)
                emb.weight = nn.Parameter(emb.weight.detach().to("cpu").clone(), requires_grad=False)
            lm.embed_tokens = _CpuEmbedding(emb, dev)
        torch.cuda.empty_cache()
        self.te_streamed = True
        logger.info("text_encoder decoder layers streamed from %s (%d layers, %.1fs, resident=%.2fGB, embed_on_gpu=%s, "
                    "pinned=%s)", self.te_host, len(lm.layers), time.time() - t0,
                    torch.cuda.memory_allocated(dev) / 2**30, self.embed_on_gpu, pruned_image.pinned_gb())

    class _Captured(Exception):
        def __init__(self, hidden):
            self.hidden = hidden

    def _keep_only_layer_output(self, te) -> None:
        """te.model を包み、`hidden_states[50]`(= 50 番目の decoder 層の出力)だけを返す。

        diffusers の encoder step は `output_hidden_states=True` で呼ぶので、そのままだと 52 本の hidden state
        (51 層 × トークン数 × 5120 × 2 バイト)が全部 GPU に残る。テキストだけなら数十 MB だが、参照画像は
        1024² で約 1k トークン、2048² で約 4k トークンになり、参照 3 枚の編集で 1.6GB、2048² の参照 2 枚で 4GB を
        超えて 8gb 構成では OOM した(実測)。50 番目の層の出力を hook で取り、そこで forward を打ち切る
        (51 番目の層も読まなくて済む)。返り値は `.hidden_states[50]` だけが入った tuple。
        """
        inner = te.model
        layer = inner.language_model.layers[TEXT_ENCODER_LAYER - 1]
        orig = inner.forward
        captured = self._Captured

        def hook(mod, args, out):
            raise captured(out[0] if isinstance(out, tuple) else out)

        layer.register_forward_hook(hook)
        n = TEXT_ENCODER_LAYER

        def forward(*args, **kwargs):
            kwargs["output_hidden_states"] = False
            try:
                orig(*args, **kwargs)
            except captured as c:
                return _HiddenOnly(tuple([None] * n + [c.hidden]))
            raise RuntimeError(f"text_encoder の {n} 番目の層が呼ばれませんでした(層数の設定を確認)")

        inner.forward = forward

    def _route_text_encoder(self, te) -> None:
        """te.model.forward を包み、入力を TE 側 GPU へ、hidden_states[50] だけをホスト経由で返す。"""
        te_dev = self.te_device
        inner = te.model
        orig = inner.forward

        def via_host(t):
            if isinstance(t, torch.Tensor) and t.device != te_dev:
                return t.to("cpu").to(te_dev) if t.device.type == "cuda" else t.to(te_dev)
            return t

        def forward(*args, **kwargs):
            args = tuple(via_host(a) for a in args)
            kwargs = {k: via_host(v) for k, v in kwargs.items()}
            out = orig(*args, **kwargs)
            hs = out.hidden_states
            out.hidden_states = tuple(h.to("cpu") if h is not None else None for h in hs)
            return out

        inner.forward = forward

    def _load_vae(self):
        from diffusers import AutoencoderKLMiniMaxH3Image

        t0 = time.time()
        vae = AutoencoderKLMiniMaxH3Image.from_pretrained(self._resolve_repo(), subfolder="vae_diffusers",
                                                          torch_dtype=torch.bfloat16)
        swap = not self.vae_on_gpu
        if swap:
            self._swap_vae(vae)
        else:
            vae.to(self.device)
        logger.info("vae loaded in %.1fs (swap=%s)", time.time() - t0, swap)
        return vae

    def _swap_vae(self, vae) -> None:
        """VAE をピン留めした原本に置き、encode / decode の間だけ GPU へ置く(TE ストリーミング時)。

        参照画像付きの編集では TE が画像トークン分の活性を使うため、16GB 級では VAE が常駐していると
        TE のエンコード中に OOM する(実測)。往復は 1.4GB で 0.1〜0.2s。コピーは計算と同じ stream で行うので、
        戻した後の領域の再利用はストリーム順序で守られる。
        """
        dev = self.device
        vae.to("cpu")
        masters = pruned_image.pin_module(vae, owner="vae")

        def wrap(fn):
            def inner(*args, **kwargs):
                pruned_image.to_device(masters, dev)
                try:
                    return fn(*args, **kwargs)
                finally:
                    pruned_image.to_host(masters)
                    torch.cuda.empty_cache()
            return inner

        vae.encode = wrap(vae.encode)
        vae.decode = wrap(vae.decode)

    def _load_transformer(self, steps: int):
        sigmas, tag, need_lora = STEP_PRESETS[steps]
        # sample_sigmas を先に確定させる(時刻集合 = AdaLN テーブルのキー)
        self.pipe.update_components(sample_sigmas=list(sigmas) if sigmas else list(self._default_sigmas))
        lora = None
        if need_lora:
            if not self.s.lora_path.exists():
                raise RuntimeError(f"steps={steps} には pDMD LoRA が必要です: {self.s.lora_path}(H3IMG_LORA_PATH)")
            lora = str(self.s.lora_path)
        t0 = time.time()
        quant = self.profile.quant
        cache_name = {"int8wo": "int8wo_adaln_table", "ck-w4a8": "ckw4a8_adaln_table"}[quant] + tag
        stream = self._transformer_stream_enabled()
        from .sources import ShardSource
        repo = self._resolve_repo()
        # キャッシュが無いときの bf16 シャードは 1 つずつ取得して消す(H3IMG_KEEP_SOURCE=1 なら消さない。
        # 元から置いてあるファイルは消さない)
        shards = ShardSource(self.s.model_id, "transformer_diffusers", Path(repo) / "transformer_diffusers",
                             keep=self.s.keep_source) if not self.s.model_dir else None
        tr = pruned_image.load_transformer(repo, self.pipe, self.device,
                                           self.s.cache_dir / cache_name, quant=quant, lora_path=lora, shards=shards,
                                           stream_blocks=stream, pin_blocks=self.transformer_host == "pinned",
                                           drop_blocks=self.s.transformer_drop_blocks,
                                           file_blocks=self.transformer_host == "file")
        self.transformer_streamed = stream
        self._install_progress_hook(tr, steps)
        total, avail = pruned_image.host_memory_gb()
        logger.info("transformer (steps=%d) ready in %.1fs, alloc=%.2fGB; host RAM: process %.1fGB, available %.1fGB "
                    "of %.1fGB, pinned %s", steps, time.time() - t0, torch.cuda.memory_allocated(self.device) / 2**30,
                    pruned_image._process_anon_gb(), avail, total, pruned_image.pinned_gb())
        return tr

    def _install_progress_hook(self, tr, steps: int) -> None:
        """forward を包んで進捗通知と中断チェックを行う(pruned_image のラッパーの外側)。"""
        orig = tr.forward
        engine = self

        @functools.wraps(orig)
        def forward(*args, **kwargs):
            ctx = engine._ctx
            if ctx is not None:
                if ctx["cancel"].is_set():
                    raise Interrupted("中断要求により生成を停止しました")
                i = kwargs.get("pdd_step_index", 0)
                ctx["progress"](i, steps)
                ctx.setdefault("t_first_step", time.time())   # ここまでが text_encoder + 前処理
            out = orig(*args, **kwargs)
            if ctx is not None:
                torch.cuda.synchronize(engine.device)
                ctx["t_last_step"] = time.time()
            return out

        tr.forward = forward

    def load(self, steps: int | None = None) -> None:
        steps = steps or self.s.default_steps
        with self.lock:
            self._load_locked(steps)

    def _load_locked(self, steps: int) -> None:
        if self.profile is None:
            self.profile = self._resolve_profile()
        if self.pipe is None:
            self.pipe = self._build_pipeline()
        if self.te is None:
            self.te = self._load_te()
        if self.vae is None:
            self.vae = self._load_vae()
        if self.transformer is not None and self.loaded_steps != steps:
            logger.info("steps %s → %s: transformer を差し替えます", self.loaded_steps, steps)
            self._drop_transformer()
        if self.transformer is None:
            self.transformer = self._load_transformer(steps)
            self.loaded_steps = steps
            self.pipe.update_components(transformer=self.transformer, text_encoder=self.te, vae=self.vae)

    def _drop_transformer(self) -> None:
        """旧 transformer への参照を全て切って VRAM を解放する。

        参照は3か所: self.transformer、パイプラインの属性(update_components は setattr で保持)、
        forward ラッパーのクロージャ(bound method `orig` 経由で自分自身を参照する循環)。
        ラッパーをクラスの forward へ戻して循環を断ち、gc してから empty_cache する。
        (2026-11-11 実機: これを怠ると steps 切替で二重常駐 → 39.8GB → OOM)
        """
        import gc

        old = self.transformer
        self.transformer = None
        self.loaded_steps = None
        if old is None:
            return
        if self.pipe is not None and getattr(self.pipe, "transformer", None) is old:
            object.__setattr__(self.pipe, "transformer", None)
        try:
            old.forward = type(old).forward.__get__(old)
        except Exception:  # noqa: BLE001
            pass
        del old
        gc.collect()
        unpinned = pruned_image.release_pinned("transformer")
        pruned_image.release_file_streamers("transformer")
        torch.cuda.empty_cache()
        logger.info("transformer dropped, alloc=%.2fGB, unpinned=%.1fGB",
                    torch.cuda.memory_allocated(self.device) / 2**30, unpinned)

    def unload(self) -> dict:
        with self.lock:
            return self._unload_locked()

    def _unload_locked(self) -> dict:
        freed = [n for n, v in (("transformer", self.transformer), ("text_encoder", self.te),
                                ("vae", self.vae)) if v is not None]
        self._drop_transformer()
        self.te = self.vae = self.pipe = None
        self.te_streamed = self.transformer_streamed = False
        import gc as _gc
        _gc.collect()
        for owner in ("te", "vae", "transformer"):
            pruned_image.release_pinned(owner)
            pruned_image.release_file_streamers(owner)
        self.profile = None
        import gc
        gc.collect()
        for d in {self.device, self.te_device}:
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
        logger.info("unloaded: %s", freed)
        return {"freed": freed}

    def _fallback_after_oom(self, where: str) -> bool:
        """OOM 後に一段下の構成へ下げる。下げられなければ False。"""
        if not self.s.vram_fallback or self.profile is None:
            return False
        idx = next(i for i, p in enumerate(PROFILES) if p.name == self.profile.name)
        if idx + 1 >= len(PROFILES):
            return False
        prev = self.profile.name
        self._unload_locked()
        self._profile_floor = idx + 1
        self.profile = self._resolve_profile()
        logger.warning("CUDA OOM (%s): 構成を %s -> %s に下げてやり直します", where, prev, self.profile.name)
        return True

    # -- 生成 ------------------------------------------------------------------

    _ctx: dict | None = None

    def generate(self, req, images: list, progress, cancel: threading.Event) -> tuple:
        """req: schemas.GenerateRequest、images: PIL 画像(edit の参照)。戻り値 (PIL.Image, stats)。"""
        steps = req.steps or self.s.default_steps
        area = req.generation_area or self.s.default_generation_area
        with self.lock:
            if req.width * req.height > self.s.max_pixels:
                raise ValueError(f"width*height が上限 {self.s.max_pixels} px を超えています")
            for attempt in range(len(PROFILES)):
                err = None
                try:
                    return self._generate_locked(req, images, progress, cancel, steps, area)
                except torch.cuda.OutOfMemoryError as exc:
                    err = exc
                # except ブロックの外で処理する: 例外のトレースバックがパイプライン内部のフレーム(旧 transformer や
                # 途中の latents への参照)を掴んだままだと、unload しても VRAM が解放されない(実測 +13GB 残留)。
                can_fallback = self.s.vram_fallback and self.profile is not None \
                    and self.profile.name != PROFILES[-1].name
                if not can_fallback:
                    raise err
                del err
                torch.cuda.empty_cache()
                self._fallback_after_oom("generate")
            raise RuntimeError("unreachable")

    def _generate_locked(self, req, images, progress, cancel, steps: int, area: str) -> tuple:
        self._load_locked(steps)
        kw = dict(prompt=req.prompt, height=req.height, width=req.width, output="images")
        if area == "requested":
            kw["min_generation_area"] = 1024  # 4MP 最低キャンバスを無効化(要求サイズで生成)
        if images:
            kw["image"] = images if len(images) > 1 else images[0]
        seed = req.seed if req.seed is not None else int(torch.randint(0, 2**31 - 1, (1,)).item())
        kw["generator"] = torch.Generator().manual_seed(seed)
        for d in {self.device, self.te_device}:
            torch.cuda.reset_peak_memory_stats(d)
        self._ctx = {"cancel": cancel, "progress": progress}
        t0 = time.time()
        try:
            if cancel.is_set():
                raise Interrupted("中断要求により生成を停止しました")
            out = self.pipe(**kw)
            torch.cuda.synchronize(self.device)
            ctx = self._ctx
        finally:
            self._ctx = None
        elapsed = time.time() - t0
        if ctx and "t_first_step" in ctx:
            # どこに時間が掛かっているかを見るための内訳(RAM の少ない PC では text_encoder がディスク律速になる)
            logger.info("phases: text_encoder+prep %.1fs, denoise %.1fs (%d steps), vae decode %.1fs, total %.1fs",
                        ctx["t_first_step"] - t0, ctx["t_last_step"] - ctx["t_first_step"], steps,
                        t0 + elapsed - ctx["t_last_step"], elapsed)
        stats = {
            "generation_seconds": round(elapsed, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated(self.device) / 2**30, 2),
            "peak_vram_te_gb": round(torch.cuda.max_memory_allocated(self.te_device) / 2**30, 2),
            "steps_used": steps,
            "seed": seed,
            "generation_area": area,
            "profile": self.profile.name,
        }
        return out[0], stats
