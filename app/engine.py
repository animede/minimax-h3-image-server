"""MiniMax-H3-Image 生成エンジン。

構成(probes/h3-image の実測に基づく、RESULTS.md 参照):
- transformer: AdaLN 事前計算テーブル + torchao int8wo(app/pruned_image.py)。steps 2/3 は pDMD
  LoRA をマージ済みの別キャッシュ。常駐 ~19GB、1024² 1.8s / 2048² 4.1s(3-step)。
- text_encoder: H3 動画用 prequant(nf4・51層、17GB)を流用。GPU が 2 枚あれば cuda:1 へ置き、
  hidden_states[50] だけをホスト経由で戻す(このマシンは GPU 間 P2P コピーが黙って全ゼロを返す)。
- VAE: bf16 2.3GB。
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from pathlib import Path

import torch

from . import pruned_image
from .config import STEP_PRESETS, Settings

logger = logging.getLogger("h3image.engine")


class Interrupted(RuntimeError):
    pass


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

    # -- 状態 ------------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self.transformer is not None

    def status(self) -> dict:
        return {
            "loaded": self.loaded,
            "steps": self.loaded_steps,
            "device": str(self.device),
            "te_device": str(self.te_device),
            "te_loaded": self.te is not None,
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
            f"{self.s.model_id} のスナップショットが見つかりません。probes/h3-image/download.py 相当で "
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
        te = AutoModelForImageTextToText.from_pretrained(str(self.s.te_dir), dtype=torch.bfloat16,
                                                         device_map=str(self.te_device))
        if self.te_device != self.device:
            self._route_text_encoder(te)
        logger.info("text_encoder loaded (%d layers) on %s in %.1fs",
                    te.config.text_config.num_hidden_layers, self.te_device, time.time() - t0)
        return te

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
            out.hidden_states = tuple(h.to("cpu") if i == 50 else None for i, h in enumerate(hs))
            return out

        inner.forward = forward

    def _load_vae(self):
        from diffusers import AutoencoderKLMiniMaxH3Image

        t0 = time.time()
        vae = AutoencoderKLMiniMaxH3Image.from_pretrained(self._resolve_repo(), subfolder="vae_diffusers",
                                                          torch_dtype=torch.bfloat16).to(self.device)
        logger.info("vae loaded in %.1fs", time.time() - t0)
        return vae

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
        tr = pruned_image.load_transformer(self._resolve_repo(), self.pipe, self.device,
                                           self.s.cache_dir / ("int8wo_adaln_table" + tag),
                                           quant="int8wo", lora_path=lora)
        self._install_progress_hook(tr, steps)
        logger.info("transformer (steps=%d) ready in %.1fs, alloc=%.2fGB", steps, time.time() - t0,
                    torch.cuda.memory_allocated(self.device) / 2**30)
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
            return orig(*args, **kwargs)

        tr.forward = forward

    def load(self, steps: int | None = None) -> None:
        steps = steps or self.s.default_steps
        with self.lock:
            self._load_locked(steps)

    def _load_locked(self, steps: int) -> None:
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
        torch.cuda.empty_cache()
        logger.info("transformer dropped, alloc=%.2fGB", torch.cuda.memory_allocated(self.device) / 2**30)

    def unload(self) -> dict:
        with self.lock:
            freed = [n for n, v in (("transformer", self.transformer), ("text_encoder", self.te),
                                    ("vae", self.vae)) if v is not None]
            self._drop_transformer()
            self.te = self.vae = self.pipe = None
            import gc
            gc.collect()
            for d in {self.device, self.te_device}:
                with torch.cuda.device(d):
                    torch.cuda.empty_cache()
            logger.info("unloaded: %s", freed)
            return {"freed": freed}

    # -- 生成 ------------------------------------------------------------------

    _ctx: dict | None = None

    def generate(self, req, images: list, progress, cancel: threading.Event) -> tuple:
        """req: schemas.GenerateRequest、images: PIL 画像(edit の参照)。戻り値 (PIL.Image, stats)。"""
        steps = req.steps or self.s.default_steps
        area = req.generation_area or self.s.default_generation_area
        with self.lock:
            self._load_locked(steps)
            if req.width * req.height > self.s.max_pixels:
                raise ValueError(f"width*height が上限 {self.s.max_pixels} px を超えています")
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
            finally:
                self._ctx = None
            elapsed = time.time() - t0
            stats = {
                "generation_seconds": round(elapsed, 2),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated(self.device) / 2**30, 2),
                "peak_vram_te_gb": round(torch.cuda.max_memory_allocated(self.te_device) / 2**30, 2),
                "steps_used": steps,
                "seed": seed,
                "generation_area": area,
            }
            return out[0], stats
