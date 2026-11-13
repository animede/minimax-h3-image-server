#!/usr/bin/env python
"""モデル類の準備(初回のみ)。既にあるものはスキップする。

  1. 本家 MiniMaxAI/MiniMax-H3-Image の必要部分を HF キャッシュへ(vae_diffusers 2.3GB、
     scheduler / tokenizer / processor / *.json、transformer の config / index)。text_encoder の重みと
     transformer の bf16 シャード(62GB)は落とさない(後者はキャッシュ構築時に 1 つずつ取得して消す)。
  2. pDMD 2-step LoRA(akhaliq/minimax-h3-image-pdmd-2step、Apache-2.0、1.4GB)→ models/lora/
  3. 量子化済み text_encoder(Qwen3-VL-32B、bnb nf4・51 層)→ models/te/
     MiniMaxAI/MiniMax-H3 の text_encoder(66.7GB、H3-Image のものとビット一致)を落とし、
     `num_hidden_layers=51` で層ごとに nf4 量子化して保存する(17GB。GPU は要るが VRAM は 1GB 未満、
     RAM も 1GB 程度。6GB の GPU・16GB の RAM の PC でも作れる)。bf16 シャード(14 個、66.7GB)は
     1 つずつ落として量子化し、消す(ディスクのピークは +5GB)。
     既に他のインストールにある場合は `--link-te <dir>` で symlink にできる。
  4. (任意)`--build-cache` で transformer の量子化キャッシュ(各 ~19GiB)を先に作る。
     省略すると初回生成時に自動構築される(約 40 秒)。

使い方:
  .venv/bin/python scripts/prepare_models.py                     # 1〜3
  .venv/bin/python scripts/prepare_models.py --link-te /path/to/te_bnb-4bit_prune1 \\
      --link-cache /path/to/cache --link-lora /path/to/lora        # 既存を参照(ダウンロードなし)
  .venv/bin/python scripts/prepare_models.py --build-cache 3 6     # キャッシュも事前構築
  .venv/bin/python scripts/prepare_models.py --keep-source         # bf16 ソースを全部落として残す(再構築が速い)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import LORA_FILENAME, TEXT_ENCODER_LAYER, settings  # noqa: E402

IMAGE_REPO = settings.model_id
TE_REPO = "MiniMaxAI/MiniMax-H3"
LORA_REPO = "akhaliq/minimax-h3-image-pdmd-2step"
LORA_SRC_FILE = "lora_model_0.safetensors"


import logging  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def log(msg: str) -> None:
    print(f"[prepare] {msg}", flush=True)


def _link(target: Path, link: Path) -> None:
    if link.is_symlink() or link.exists():
        if link.is_symlink() and link.resolve() == target.resolve():
            log(f"既に参照済み: {link} -> {target}")
            return
        raise SystemExit(f"{link} が既に存在します。手で整理してから再実行してください")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target.resolve())
    log(f"symlink: {link} -> {target}")


def download_image_model(keep_source: bool) -> str:
    """本家リポジトリの必要部分を落とす。transformer の bf16 シャード(62GB)は既定では落とさず、
    量子化キャッシュの構築時に 1 つずつ取得して消す(--keep-source なら従来どおり全部落として残す)。"""
    from huggingface_hub import snapshot_download

    if settings.model_dir:
        log(f"H3IMG_MODEL_DIR 指定のためダウンロードを省略: {settings.model_dir}")
        return settings.model_dir
    t0 = time.time()
    path = snapshot_download(
        IMAGE_REPO,
        allow_patterns=["transformer_diffusers/*", "vae_diffusers/*", "scheduler/*", "tokenizer/*",
                        "processor/*", "model_index.json", "modular_model_index.json", "pdd_config.json",
                        "text_encoder/*.json", "examples/*"],
        ignore_patterns=None if keep_source else ["transformer_diffusers/*.safetensors"],
    )
def download_lora() -> None:
    from huggingface_hub import hf_hub_download

    dst = settings.lora_path
    if dst.exists():
        log(f"LoRA あり: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download(LORA_REPO, LORA_SRC_FILE)
    shutil.copyfile(src, dst)
    log(f"LoRA: {dst}({dst.stat().st_size / 2**30:.2f}GiB)")


def build_text_encoder(keep_source: bool) -> None:
    dst = settings.te_dir
    if (dst / "config.json").exists():
        log(f"text_encoder あり: {dst}")
        return
    import torch
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available():
        raise SystemExit("text_encoder の nf4 量子化には GPU が必要です(--link-te で既存を参照することもできます)")
    # config と index だけ先に落とし、bf16 シャード(14 個、計 66.7GB)は 1 つずつ取得して量子化し、消す
    # (--keep-source なら全部落として残す)。VRAM は 0.3GB、RAM は 1GB、ディスクのピークはシャード 1 つ分(5GB)。
    from app import te_quant
    from app.sources import ShardSource

    src = Path(snapshot_download(TE_REPO, allow_patterns=["text_encoder/*.json"] + (["text_encoder/*"] if keep_source else [])))
    src_te = src / "text_encoder"
    shards = ShardSource(TE_REPO, "text_encoder", src_te, keep=keep_source)
    te_quant.build(src_te, dst, TEXT_ENCODER_LAYER + 1, torch.device("cuda"), log=log, shards=shards)
    size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / 2**30
    log(f"text_encoder 保存: {dst}({size:.1f}GiB)")


def download_pe_prompts() -> None:
    """プロンプト拡張用の公式ルール文書(PE リポジトリの prompt/*、小さなテキスト)を models/prompts/ へ。"""
    from huggingface_hub import hf_hub_download
    from app.enhance import PE_FILES, PE_REPO

    dst_root = settings.models_dir / "prompts"
    for rel in PE_FILES.values():
        dst = dst_root / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(hf_hub_download(PE_REPO, rel), dst)
        log(f"PE prompt: {rel}")


def build_cache(steps_list: list[int]) -> None:
    import torch
    from app.engine import Engine
    from app.config import STEP_PRESETS

    engine = Engine(settings)
    for steps in steps_list:
        if steps not in STEP_PRESETS:
            raise SystemExit(f"steps は {sorted(STEP_PRESETS)} のいずれか: {steps}")
        t0 = time.time()
        engine.load(steps)
        log(f"キャッシュ構築 steps={steps}: {time.time() - t0:.0f}s")
    engine.unload()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--link-te", type=Path, help="既存の量子化済み text_encoder ディレクトリを symlink で参照")
    ap.add_argument("--link-cache", type=Path, help="既存の transformer キャッシュディレクトリを symlink で参照")
    ap.add_argument("--link-lora", type=Path, help="既存の LoRA ディレクトリを symlink で参照")
    ap.add_argument("--build-cache", type=int, nargs="*", metavar="STEPS", help="transformer キャッシュを事前構築(例: 3 6)")
    ap.add_argument("--keep-source", action="store_true",
                    help="bf16 ソース(text_encoder 66.7GB・transformer 62GB)を全部落として残す(既定は 1 シャードずつ取得して消す)")
    ap.add_argument("--delete-te-source", action="store_true", help=argparse.SUPPRESS)   # 旧オプション(今は既定の挙動)
    ap.add_argument("--skip-download", action="store_true", help="画像モデルのダウンロードを省略")
    a = ap.parse_args()

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    if a.link_te:
        _link(a.link_te, settings.te_dir)
    if a.link_cache:
        _link(a.link_cache, settings.cache_dir)
    if a.link_lora:
        _link(a.link_lora, settings.lora_path.parent)

    if not a.skip_download:
        download_image_model(a.keep_source)
    download_lora()
    download_pe_prompts()
    build_text_encoder(a.keep_source)
    if a.build_cache:
        build_cache(a.build_cache)
    log("完了")


if __name__ == "__main__":
    main()
