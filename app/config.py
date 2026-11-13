"""MiniMax-H3-Image サーバの設定(環境変数 H3IMG_*)。

モデル類の置き場はリポジトリ直下の `models/`(`H3IMG_MODELS_DIR` で変更可):
  models/te/     量子化済み text_encoder(Qwen3-VL-32B、bnb nf4・51 層)。`scripts/prepare_models.py` が作る
  models/lora/   pDMD 2-step LoRA(steps 2 / 3 で使う)
  models/cache/  量子化済み transformer キャッシュ(各 ~19GiB、初回生成時に自動構築)
いずれも symlink でよい(他のインストールと共有する場合)。本家モデルの bf16 シャードは
HF キャッシュ(`~/.cache/huggingface/hub`)に置く(`H3IMG_MODEL_DIR` で直接指定も可)。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

STEP_PRESETS = {
    # steps → (sample_sigmas の上書き(None=チェックポイント既定の 6 step)、キャッシュ名接尾辞、LoRA 要否)
    6: (None, "", False),
    # pDMD 2-step LoRA(akhaliq/minimax-h3-image-pdmd-2step)。shift 12、σ = 12b/(1+11b)。
    3: ([1.0, 0.96, 0.8571428571428571, 0.0], "_pdmd3", True),
    2: ([1.0, 0.9230769230769231, 0.0], "_pdmd2", True),
}

LORA_FILENAME = "minimax_h3_image_pdmd_2step_lora.safetensors"


def _load_dotenv(path: Path) -> None:
    """`.env`(KEY=VALUE、# コメント)を環境変数へ(既存の環境変数が優先)。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(BASE_DIR / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


@dataclass
class Settings:
    port: int = int(_env("H3IMG_PORT", "8633"))
    model_id: str = _env("H3IMG_MODEL_ID", "MiniMaxAI/MiniMax-H3-Image")
    # ローカルのスナップショットを直接指す場合(未指定なら HF キャッシュの最新スナップショット)
    model_dir: str = _env("H3IMG_MODEL_DIR", "")
    models_dir: Path = Path(_env("H3IMG_MODELS_DIR", str(BASE_DIR / "models")))
    # 使う GPU: H3IMG_GPU=1 のように番号で指定(H3IMG_DEVICE=cuda:1 と同じ)。TE は H3IMG_TE_GPU / H3IMG_TE_DEVICE。
    # 既定: transformer は cuda:0、TE は GPU が 2 枚以上あれば cuda:1(engine 側で解決)
    device: str = (f"cuda:{_env('H3IMG_GPU', '')}" if _env("H3IMG_GPU", "") else _env("H3IMG_DEVICE", "cuda:0"))
    te_device: str = (f"cuda:{_env('H3IMG_TE_GPU', '')}" if _env("H3IMG_TE_GPU", "") else _env("H3IMG_TE_DEVICE", ""))
    # VRAM 予算(GB、transformer 側 GPU)。auto(既定)= その GPU の総容量。起動時/ロード時に実際の空きと比べ
    # 小さい方を予算にして、48gb / 32gb / 16gb / 8gb の構成(engine.PROFILES)を自動選択する。
    # 生成中に CUDA OOM が起きたら一段下の構成へ自動で下げて 1 回だけやり直す(H3IMG_VRAM_FALLBACK=0 で無効)。
    vram_gb: str = _env("H3IMG_VRAM_GB", "auto")
    vram_fallback: bool = _env("H3IMG_VRAM_FALLBACK", "1") == "1"
    # 予算から差し引いて必ず空けておく VRAM(GB)。画面表示や他のアプリの分。予算 = min(要求, 空き - これ)
    vram_reserve_gb: float = float(_env("H3IMG_VRAM_RESERVE_GB", "1.0"))
    # text_encoder をストリーミングするときの原本の置き場所: pinned(ピン留めしたホスト RAM、約 13GB を常に使う)/
    # mmap(safetensors ファイルのページキャッシュから直接流す。OS が回収できるメモリなので RAM の少ない PC 向け。
    # ピン留めしない分、生成が 0.7〜0.9 秒遅い)/ auto(既定。ホスト RAM 48GB 以上なら pinned、未満なら mmap)
    te_host: str = _env("H3IMG_TE_HOST", "auto")
    # transformer ブロックをストリーミングするときの原本の置き場所: pinned / mmap(量子化キャッシュのページキャッシュから
    # 直接流す。必須 RAM が約 10GB 減るが、毎生成でディスクから読み直す)/ auto(既定 = te_host と同じ規則)
    transformer_host: str = _env("H3IMG_TRANSFORMER_HOST", "auto")
    # transformer_host=mmap のとき、使い終わったらページを捨てる末尾ブロック数。auto = ページキャッシュに使える RAM から
    # はみ出す分だけ(全部が押し出されて毎ステップ読み直すのを避ける)。0 で捨てない
    transformer_drop_blocks: str = _env("H3IMG_TRANSFORMER_DROP_BLOCKS", "auto")
    # 以下 3 つは auto(既定)なら VRAM 予算から決まる。明示すると予算より優先する。
    # transformer の量子化: int8wo(torchao、19GiB)/ ck-w4a8(comfy-kitchen W4A8、約 11GiB)
    transformer_quant: str = _env("H3IMG_TRANSFORMER_QUANT", "auto")
    # text_encoder の層ストリーミング(1 / 0): 51 層の decoder を pinned ホストから 1 層ずつ流す(常駐 ~0.3GB、
    # エンコード +1.1s、出力は bit 一致)。lm_head 削除・埋め込み CPU 化・VAE スワップも同時に有効になる。
    te_stream: str = _env("H3IMG_TE_STREAM", "auto")
    # transformer ブロックのストリーミング(1 / 0): 50 ブロックを pinned ホストから先読みしながら流す(8GB 級)。
    transformer_stream: str = _env("H3IMG_TRANSFORMER_STREAM", "auto")
    # 既定のステップ数(2 / 3 / 6)。リクエストごとに上書き可(transformer の差し替えが走る)
    default_steps: int = int(_env("H3IMG_STEPS", "3"))
    # 起動時にロードする(1)か、初回リクエストまで遅延する(0、既定)
    preload: bool = _env("H3IMG_PRELOAD", "0") == "1"
    output_dir: Path = Path(_env("H3IMG_OUTPUT_DIR", str(BASE_DIR / "outputs")))
    input_dir: Path = Path(_env("H3IMG_INPUT_DIR", str(BASE_DIR / "inputs")))
    max_upload_mb: int = int(_env("H3IMG_MAX_UPLOAD_MB", "200"))
    max_queue: int = int(_env("H3IMG_MAX_QUEUE", "4"))
    # 既定の生成キャンバス: "native" = 最低 4MP で生成して縮小(モデル推奨、1024² でも 2048² 相当の時間)、
    # "requested" = 要求サイズで生成(1024² が 2〜3 倍速い。品質は良好)
    default_generation_area: str = _env("H3IMG_GENERATION_AREA", "requested")
    max_pixels: int = int(_env("H3IMG_MAX_PIXELS", str(4096 * 4096)))
    # プロンプト拡張に使う OpenAI 互換 LLM(未設定なら拡張は 501)
    llm_base_url: str = _env("H3IMG_LLM_BASE_URL", "")
    llm_model: str = _env("H3IMG_LLM_MODEL", "")
    llm_api_key: str = _env("H3IMG_LLM_API_KEY", "")
    llm_timeout_s: float = float(_env("H3IMG_LLM_TIMEOUT_S", "180"))
    # プロンプト拡張で本家 GitHub の prompt-writing skill のガイド(references/t2i-en.txt / edit-en.txt)も system に足す
    enhance_skill_guide: bool = _env("H3IMG_ENHANCE_SKILL_GUIDE", "1") == "1"
    cache_tags: dict = field(default_factory=lambda: {k: v[1] for k, v in STEP_PRESETS.items()})
    # 個別上書き(未指定なら models_dir 配下。__post_init__ で解決)
    te_dir: Path = field(default_factory=lambda: Path())
    lora_path: Path = field(default_factory=lambda: Path())
    cache_dir: Path = field(default_factory=lambda: Path())

    def __post_init__(self) -> None:
        self.te_dir = Path(_env("H3IMG_TE_DIR", str(self.models_dir / "te")))
        self.lora_path = Path(_env("H3IMG_LORA_PATH", str(self.models_dir / "lora" / LORA_FILENAME)))
        self.cache_dir = Path(_env("H3IMG_CACHE_DIR", str(self.models_dir / "cache")))
        if self.default_steps not in STEP_PRESETS:
            raise ValueError(f"H3IMG_STEPS は {sorted(STEP_PRESETS)} のいずれかです: {self.default_steps}")
        if self.vram_gb != "auto":
            try:
                float(self.vram_gb)
            except ValueError:
                raise ValueError("H3IMG_VRAM_GB は auto か数値(GB)です") from None
        if self.te_host not in ("auto", "pinned", "mmap"):
            raise ValueError("H3IMG_TE_HOST は auto / pinned / mmap のいずれかです")
        if self.transformer_host not in ("auto", "pinned", "mmap"):
            raise ValueError("H3IMG_TRANSFORMER_HOST は auto / pinned / mmap のいずれかです")
        if self.te_stream not in ("auto", "0", "1"):
            raise ValueError("H3IMG_TE_STREAM は auto / 0 / 1 のいずれかです")
        if self.transformer_stream not in ("auto", "0", "1"):
            raise ValueError("H3IMG_TRANSFORMER_STREAM は auto / 0 / 1 のいずれかです")
        if self.transformer_quant not in ("auto", "int8wo", "ck-w4a8"):
            raise ValueError("H3IMG_TRANSFORMER_QUANT は auto / int8wo / ck-w4a8 のいずれかです")
        if self.default_generation_area not in ("native", "requested"):
            raise ValueError("H3IMG_GENERATION_AREA は native / requested のいずれかです")
        for d in (self.cache_dir, self.output_dir, self.input_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
