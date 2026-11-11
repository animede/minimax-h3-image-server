# minimax-h3-image-server

[MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)(33B、静止画生成 +
最大 16 枚参照の編集)を **約 21GB の VRAM で、1024² を 1.8 秒・2048² を 4.1 秒**で動かす
単独の画像サーバ + ブラウザ UI です。diffusers(main)ベース、コードは MIT ライセンス。

- 本家の bf16 は transformer 66GB + text_encoder 66.7GB で 80GB GPU にも同時に載りませんが、
  本サーバは **AdaLN 射影の事前計算テーブル(bit 一致・近似なし)+ int8 重み量子化 + nf4 text_encoder**で
  transformer 常駐 ~19GB、text_encoder ~17GB に収めています。
- **pDMD 2-step LoRA** をマージした 3 step を既定にし、6 step とほぼ同品質で 1.7 倍速。
- 非同期ジョブ API と、生成・参照編集・履歴・ダウンロード・削除ができる UI。

## 目次

1. [要求 VRAM](#要求-vram)
2. [測定した生成速度](#測定した生成速度)
3. [導入](#導入)
4. [起動と使い方](#起動と使い方)
5. [環境変数](#環境変数)
6. [軽量化の中身](#軽量化の中身)
7. [ライセンスの注意](#ライセンスの注意)

## 要求 VRAM

| 部品 | 常駐 | 備考 |
|---|---|---|
| transformer(int8wo + AdaLN テーブル)+ VAE | **~19GB** | 生成時 peak 20.8GB(1024²)〜21.4GB(2048² 編集)。実測は 4096² 未確認 |
| text_encoder(Qwen3-VL-32B、nf4・51 層) | **~17GB** | 2 枚目の GPU があれば自動でそちらへ(`H3IMG_TE_DEVICE`) |

| 構成 | 必要な GPU |
|---|---|
| 1 枚で動かす | **48GB 級**(transformer 21GB + TE 17GB ≈ 38GB + 余裕) |
| 2 枚で動かす(推奨) | GPU0 **24GB 級**(transformer)+ GPU1 **20GB 以上**(TE) |

ホスト RAM: 量子化キャッシュがある状態では数 GB。初回のキャッシュ構築も bf16 を 1 テンソルずつ扱うため
大きな RAM は要りません(実測ピーク +数 GB)。ディスクは [導入](#導入) 参照。

## 測定した生成速度

RTX PRO 6000 Blackwell(96GB、transformer)+ RTX PRO 5000 Blackwell(48GB、text_encoder)、seed 固定、
いずれも生成時間のみ(モデルロード済み)。

| ケース | **3 step(既定)** | 6 step | 2 step |
|---|---|---|---|
| T2I 1024²(要求サイズで生成) | **1.5〜1.8s** | 2.4s | 0.9〜1.5s |
| T2I 2048² | **4.1s** | 7.0s | 3.0s |
| 参照編集 1024²(参照 1 枚) | **4.5s** | 7.4s | 3.4s |
| 参照編集 2048²(参照 1 枚) | **6.7s** | 11.3s | 5.0s |
| peak VRAM(GPU0) | 20.8〜21.4GB | 同 | 同 |

- ロード(初回生成時): text_encoder 3〜8s + transformer キャッシュ 12s + VAE 1s ≈ 20 秒。
  キャッシュ未作成の初回のみ +約 40 秒(bf16 シャードからの量子化)。
- 品質: 3 step は 6 step と等倍目視でほぼ同等。**2 step は織り目状の微細パターンが出る**ためプレビュー向け。
- `generation_area: "native"`(本家推奨の最低 4MP キャンバスで生成して縮小)にすると 1024² でも 2048² と同じ時間。
- 比較: 同じ重みを bnb int8(`from_pretrained` の `load_in_8bit`)で素直に載せた場合は 1024² 3.5s / 2048² 9.7s、
  peak 33〜34GB。

## 導入

### 動作環境

- NVIDIA GPU(上記)。sm_120(Blackwell)で検証。torch 2.11.0+cu130 のため CUDA 13 系ドライバが必要
- Python 3.12、[uv](https://github.com/astral-sh/uv)(pip でも可)
- ディスク: 本家モデル 64GB(HF キャッシュ)+ LoRA 1.4GB + 量子化済み text_encoder 17GB +
  transformer キャッシュ 19GiB × 使うステップ種(3 step だけなら 19GiB)。
  **text_encoder の作成時のみ一時的に +66.7GB**(bf16 ソース。`--delete-te-source` で作成後に削除可)

### 手順

```bash
git clone <this repo> minimax-h3-image-server && cd minimax-h3-image-server
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# モデル類の準備(初回のみ。本家モデル 64GB + LoRA のダウンロードと text_encoder の量子化。GPU が必要)
.venv/bin/python scripts/prepare_models.py --delete-te-source

# 任意: transformer の量子化キャッシュを先に作っておく(省略すると初回生成時に自動構築)
.venv/bin/python scripts/prepare_models.py --skip-download --build-cache 3
```

既に別のインストール(diffusers-movie-server 等)にモデル類がある場合は、ダウンロードせず symlink で参照できます:

```bash
.venv/bin/python scripts/prepare_models.py --skip-download \
  --link-te /path/to/te_bnb-4bit_prune1 --link-cache /path/to/cache --link-lora /path/to/lora
```

`models/` の中身(`te/` `lora/` `cache/`)は symlink でも実体でも構いません。

### Hugging Face へのアクセス

本家リポジトリは公開ですが、環境によっては `HF_TOKEN` が必要です。`scripts/prepare_models.py` は
`huggingface_hub` の通常のキャッシュ(`~/.cache/huggingface/hub`)を使います。

## 起動と使い方

```bash
./run.sh                   # http://localhost:8633/
H3IMG_PORT=8634 ./run.sh   # ポート変更
H3IMG_STEPS=6 ./run.sh     # 既定ステップを 6 に
```

ブラウザ UI: モード(t2i / 参照編集)、プロンプト、サイズ、steps、キャンバス、seed、参照画像(最大 16 枚)、
履歴、ダウンロード、削除。

API(非同期ジョブ方式):

| メソッド | パス | 内容 |
|---|---|---|
| POST | `/api/assets` | 参照画像(PNG/JPG/WebP/BMP、multipart `file`)→ `{id, kind, filename, size}` |
| POST | `/api/jobs` | 生成ジョブ(202)。body は下記 |
| GET | `/api/jobs?limit=N` / `/api/jobs/{id}` | 履歴 / 状態(`queued/running/completed/failed/interrupted`、`image_url`) |
| DELETE | `/api/jobs/{id}` / `/api/outputs/{file}` | 履歴削除(実行中なら中断)/ 生成物削除 |
| POST | `/api/interrupt` | `{job_id?}` 中断 |
| POST | `/api/admin/unload` / `/api/admin/load` | VRAM 解放(プロセス温存)/ 明示ロード(`{steps}`) |
| GET | `/api/health` / `/api/status` | 状態 |

```json
{"mode": "t2i" | "edit", "prompt": "...", "width": 2048, "height": 2048, "seed": 42,
 "steps": 2 | 3 | 6, "generation_area": "native" | "requested",
 "conditions": [{"asset_id": "..."}]}
```

- `edit` はプロンプト中で `<Picture 1>`, `<Picture 2>` … と参照します(本家の流儀。長く具体的な指示が効きます)。
- `width` / `height` は 32 の倍数に丸められます。上限は `H3IMG_MAX_PIXELS`(既定 4096²)。
- `steps` をリクエストで変えると transformer を差し替えます(キャッシュ読込 ~12s)。

## 環境変数

| 変数 | 既定 | 内容 |
|---|---|---|
| `H3IMG_PORT` | 8633 | |
| `H3IMG_STEPS` | 3 | 既定ステップ(2 / 3 / 6) |
| `H3IMG_DEVICE` / `H3IMG_TE_DEVICE` | cuda:0 /(2 枚あれば cuda:1) | transformer / text_encoder の配置先 |
| `H3IMG_GENERATION_AREA` | requested | 既定キャンバス(`native` = 最低 4MP) |
| `H3IMG_PRELOAD` | 0 | 1 で起動時にロード |
| `H3IMG_MODELS_DIR` | `./models` | te / lora / cache の置き場 |
| `H3IMG_TE_DIR` / `H3IMG_LORA_PATH` / `H3IMG_CACHE_DIR` / `H3IMG_MODEL_DIR` | 上記配下 / HF キャッシュ | 個別の上書き |
| `H3IMG_MAX_PIXELS` | 4096² | 1 リクエストの上限画素 |
| `H3IMG_MAX_QUEUE` / `H3IMG_MAX_UPLOAD_MB` | 4 / 200 | キュー長 / アップロード上限 |

## 軽量化の中身

- **AdaLN 射影の事前計算テーブル(厳密)**: 各ブロックの `Linear(2688→96768)`(計 26GB、全体の 39%)は
  時刻埋め込みだけの関数で、このモデルはサンプリング時刻が固定(6 値 + 参照行の 1 値)。出力を先に
  計算して `[7, 96768]` のテーブル(計 68MB)に置き換えます。元の演算と同じ device/dtype で計算するため
  **bit 一致**(ブロック 0 で検証)。未知の時刻が来たら黙って誤らず例外にします。
- **残り 312 層の Linear を torchao `Int8WeightOnly`(PerRow)**。bf16 シャードから 1 テンソルずつ GPU で
  量子化し、量子化済みを `models/cache/` に保存(各 ~19GiB)。
- **text_encoder は bnb nf4 + 51 層に刈り込み**(モデルは `hidden_states[50]` しか読まない。
  50 層にすると最終 norm 後の値になって条件付けが変わるため 51 層)。
- **pDMD 2-step LoRA**(rank 128)は量子化前の bf16 重みに fp32 でマージ。steps 2 / 3 は別キャッシュ。
- 2 GPU 構成では text_encoder の出力(`hidden_states[50]`)だけをホスト経由で transformer 側へ渡します
  (GPU 間 P2P コピーが黙って全ゼロを返す環境があったため)。

## ライセンスの注意

| 対象 | ライセンス |
|---|---|
| **本リポジトリのコード** | **MIT**([LICENSE](LICENSE)) |
| モデル本体 MiniMaxAI/MiniMax-H3-Image(transformer / VAE) | **MiniMax-H3-Image Community License**(本家の [LICENSE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE)) |
| text_encoder(MiniMaxAI/MiniMax-H3 の Qwen3-VL-32B) | MiniMax-H3 のライセンス |
| pDMD 2-step LoRA(akhaliq/minimax-h3-image-pdmd-2step) | Apache-2.0(ただし出力はモデルのライセンスに従う) |

- **コードが MIT でも、モデルの重みと生成物には MiniMax のライセンスが適用されます。** 本家ライセンスには
  商用利用の条件(別途の書面許諾)や地域制限、表示義務などがあります。利用前に必ず本家ページの原文を確認してください。
- 本リポジトリは**モデルの重みを含みません**。量子化済み text_encoder や transformer キャッシュも各自の環境で
  生成する前提です(派生物の再配布可否は本家ライセンスの条件に従います)。
- 生成した画像の扱い(商用利用・公開)もモデルのライセンスの範囲内で行ってください。

## 経緯

diffusers-movie-server(MiniMax-H3 動画 + LTX-2.5)の第 3 バックエンドとして実装したものを、
単独で動く画像サーバとして切り出したものです。

---

### English summary

Standalone server + browser UI for MiniMax-H3-Image (33B text-to-image / reference editing) running in
~21GB of VRAM for the transformer plus ~17GB for the nf4 text encoder (ideally on a second GPU): the AdaLN
projections (39% of the weights) are replaced by an exact precomputed table (the sampling timesteps are
fixed), the remaining linears are torchao int8 weight-only, and the pDMD 2-step LoRA is merged for a 3-step
default. Measured on an RTX PRO 6000: 1.8s @1024², 4.1s @2048², 6.7s for a 2048² reference edit.
`scripts/prepare_models.py` downloads the model and the LoRA and builds the nf4 text encoder.
**Code: MIT. Model weights and outputs: MiniMax-H3-Image Community License (see the model card).**
