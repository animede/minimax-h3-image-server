#!/usr/bin/env python
"""モデル類の準備(初回のみ)。既にあるものはスキップする。

  1. 本家 MiniMaxAI/MiniMax-H3-Image の必要部分を HF キャッシュへ(transformer_diffusers 64GB、
     vae_diffusers 2.3GB、scheduler / tokenizer / processor / *.json)。text_encoder の重みは落とさない。
  2. pDMD 2-step LoRA(akhaliq/minimax-h3-image-pdmd-2step、Apache-2.0、1.4GB)→ models/lora/
  3. 量子化済み text_encoder(Qwen3-VL-32B、bnb nf4・51 層)→ models/te/
     MiniMaxAI/MiniMax-H3 の text_encoder(66.7GB、H3-Image のものとビット一致)を落とし、
     `num_hidden_layers=51` の config で nf4 ロードして保存する(17GB、GPU が必要)。
     既に他のインストールにある場合は `--link-te <dir>` で symlink にできる。
  4. (任意)`--build-cache` で transformer の量子化キャッシュ(各 ~19GiB)を先に作る。
     省略すると初回生成時に自動構築される(約 40 秒)。

使い方:
  .venv/bin/python scripts/prepare_models.py                     # 1〜3
  .venv/bin/python scripts/prepare_models.py --link-te /path/to/te_bnb-4bit_prune1 \\
      --link-cache /path/to/cache --link-lora /path/to/lora        # 既存を参照(ダウンロードなし)
  .venv/bin/python scripts/prepare_models.py --build-cache 3 6     # キャッシュも事前構築
  .venv/bin/python scripts/prepare_models.py --delete-te-source    # TE 量子化後に bf16 66.7GB を削除
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import LORA_FILENAME, settings  # noqa: E402

IMAGE_REPO = settings.model_id
TE_REPO = "MiniMaxAI/MiniMax-H3"
LORA_REPO = "akhaliq/minimax-h3-image-pdmd-2step"
LORA_SRC_FILE = "lora_model_0.safetensors"
TEXT_ENCODER_LAYER = 50  # hidden_states[50] を読む → 51 層あればよい


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


def download_image_model() -> str:
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
    )
    log(f"画像モデル: {path}({time.time() - t0:.0f}s)")
    return path


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


def build_text_encoder(delete_source: bool) -> None:
    dst = settings.te_dir
    if (dst / "config.json").exists():
        log(f"text_encoder あり: {dst}")
        return
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForImageTextToText, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise SystemExit("text_encoder の nf4 量子化には GPU が必要です(--link-te で既存を参照することもできます)")
    t0 = time.time()
    src = snapshot_download(TE_REPO, allow_patterns=["text_encoder/*"])
    log(f"TE ソース(bf16 66.7GB): {src}({time.time() - t0:.0f}s)")
    cfg = AutoConfig.from_pretrained(src, subfolder="text_encoder")
    # MiniMax-H3 は hidden_states[50] しか読まない。51 層で止める(50 層にすると最終 norm 後の
    # 値になってしまい条件付けが変わる。層 51〜63 と lm_head は読み込まれず捨てられる)。
    cfg.text_config.num_hidden_layers = TEXT_ENCODER_LAYER + 1
    t0 = time.time()
    te = AutoModelForImageTextToText.from_pretrained(
        src, subfolder="text_encoder", config=cfg, dtype=torch.bfloat16, device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16),
    )
    log(f"nf4 ロード: {time.time() - t0:.0f}s(GPU {torch.cuda.memory_allocated() / 2**30:.1f}GB)")
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    te.save_pretrained(str(tmp))
    tmp.rename(dst)
    size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / 2**30
    log(f"text_encoder 保存: {dst}({size:.1f}GiB)")
    if delete_source:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        for repo in info.repos:
            if repo.repo_id == TE_REPO:
                shutil.rmtree(repo.repo_path)
                log(f"TE ソースを削除: {repo.repo_path}")


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
    ap.add_argument("--delete-te-source", action="store_true", help="TE 量子化後に bf16 ソース(66.7GB)を HF キャッシュから削除")
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
        download_image_model()
    download_lora()
    download_pe_prompts()
    build_text_encoder(a.delete_te_source)
    if a.build_cache:
        build_cache(a.build_cache)
    log("完了")


if __name__ == "__main__":
    main()
