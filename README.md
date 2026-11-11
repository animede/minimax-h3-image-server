# minimax-h3-image-server

[MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)(33B、静止画生成 +
最大 16 枚参照の編集)を **1 枚の GPU(約 21GB)で、1024² を 1.8 秒・2048² を 4.1 秒**で動かす
単独の画像サーバ + ブラウザ UI です。diffusers(main)ベース。

| | 本家(bf16) | **本サーバ(3 step 既定)** |
|---|---|---|
| transformer | 66GB、80GB GPU でも TE と同時に載らない | **~19GB 常駐 / peak ~21GB** |
| text_encoder(Qwen3-VL-32B) | 66.7GB | **~17GB**(nf4・51 層。2 枚目の GPU があればそちらへ) |
| T2I 1024² / 2048² | — | **1.8s / 4.1s**(6 step 2.4s / 7.0s) |
| 参照編集 2048² | — | **6.7s** |

## 軽量化の中身

- **AdaLN 射影の事前計算テーブル(厳密)**: 各ブロックの `Linear(2688→96768)`(計 26GB、全体の 39%)は
  時刻だけの関数で、このモデルはサンプリング時刻が固定(6 値 + 参照行)。出力を先に計算して
  テーブル(68MB)に置き換える。元の演算と **bit 一致**(近似なし)。
- **残り 312 層を torchao int8wo**(bf16 シャードから 1 テンソルずつ量子化、CPU/GPU に全展開しない)。
  量子化済みは `models/cache/` に保存(各 ~19GiB)。
- **pDMD 2-step LoRA**(akhaliq、Apache-2.0)をマージした 3 step を既定に。6 step とほぼ同品質で 1.7 倍速。
  2 step は織り目状の微細パターンが出るためプレビュー向け。
- text_encoder は MiniMax-H3(動画)と同一の重みなので、動画サーバと共用できる。

## 動作環境

- NVIDIA GPU(推奨 24GB 以上 ×1。TE を同じ GPU に置くと合計 ~40GB、2 枚目があれば TE をそちらへ)
- Python 3.12、CUDA 13 系ドライバ(torch 2.11.0+cu130)
- ディスク: HF キャッシュに本家モデル 64GB + 量子化済み TE 17GB + LoRA 1.4GB + キャッシュ 19GiB/steps 種
  (TE の作成時のみ一時的に bf16 ソース 66.7GB が要る。`--delete-te-source` で作成後に削除可)

## セットアップ

```bash
git clone <this repo> minimax-h3-image-server && cd minimax-h3-image-server
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# モデル類(初回のみ。画像モデル 64GB + LoRA + TE の量子化。GPU が必要)
.venv/bin/python scripts/prepare_models.py
# 既に別インストール(diffusers-movie-server 等)にある場合はダウンロードせず参照する
.venv/bin/python scripts/prepare_models.py --skip-download \
  --link-te /path/to/te_bnb-4bit_prune1 --link-cache /path/to/cache --link-lora /path/to/lora
```

Hugging Face の本家リポジトリは **MiniMax-H3-Image Community License**(商用は別途許諾、地域制限あり)です。
利用条件は本家ページで確認してください。

## 起動

```bash
./run.sh                       # http://localhost:8633/
H3IMG_PORT=8634 ./run.sh       # ポート変更
H3IMG_STEPS=6 ./run.sh         # 既定ステップを 6 に
```

初回生成時にロード(TE 3〜8s + transformer 12s + VAE 1s、キャッシュ未作成なら +40s)。

## 使い方

ブラウザ UI(生成・参照編集・履歴・ダウンロード・削除)のほか、非同期ジョブ API:

| メソッド | パス | 内容 |
|---|---|---|
| POST | `/api/assets` | 参照画像(PNG/JPG/WebP/BMP、multipart `file`)→ `{id, kind, filename, size}` |
| POST | `/api/jobs` | 生成ジョブ(202)。body は下記 |
| GET | `/api/jobs?limit=N` / `/api/jobs/{id}` | 履歴 / 状態(`queued/running/completed/failed/interrupted`、`image_url`) |
| DELETE | `/api/jobs/{id}` / `/api/outputs/{file}` | 履歴削除(実行中なら中断)/ 生成物削除 |
| POST | `/api/interrupt` | `{job_id?}` 中断 |
| POST | `/api/admin/unload` / `/api/admin/load` | VRAM 解放 / 明示ロード(`{steps}`) |
| GET | `/api/health` / `/api/status` | 状態 |

```json
{"mode": "t2i" | "edit", "prompt": "...", "width": 2048, "height": 2048, "seed": 42,
 "steps": 2 | 3 | 6, "generation_area": "native" | "requested",
 "conditions": [{"asset_id": "..."}]}
```
- `edit` はプロンプト中で `<Picture 1>`, `<Picture 2>` … と参照する(本家の流儀。長く具体的な指示が効く)。
- `generation_area`: `native` = 本家推奨の最低 4MP キャンバスで生成して縮小(1024² でも 2048² と同じ時間)、
  `requested` = 要求サイズで生成(既定、1024² が 2〜3 倍速い)。
- `steps` をリクエストで変えると transformer を差し替える(キャッシュ読込 ~12s)。

## 環境変数(H3IMG_*)

| 変数 | 既定 | 内容 |
|---|---|---|
| `H3IMG_PORT` | 8633 | |
| `H3IMG_STEPS` | 3 | 既定ステップ(2 / 3 / 6) |
| `H3IMG_DEVICE` / `H3IMG_TE_DEVICE` | cuda:0 /(2 枚あれば cuda:1) | TE の配置先 |
| `H3IMG_GENERATION_AREA` | requested | 既定キャンバス |
| `H3IMG_PRELOAD` | 0 | 1 で起動時ロード |
| `H3IMG_MODELS_DIR` | `./models` | te / lora / cache の置き場(symlink 可) |
| `H3IMG_TE_DIR` / `H3IMG_LORA_PATH` / `H3IMG_CACHE_DIR` / `H3IMG_MODEL_DIR` | 上記配下 | 個別の上書き |
| `H3IMG_MAX_PIXELS` | 4096² | 1 リクエストの上限画素 |

## 既知の注意

- GPU 間の受け渡しはホスト経由にしている(P2P コピーが黙って全ゼロを返す環境があったため)。
- 時刻集合(= AdaLN テーブルのキー)はキャッシュごとに固定。スケジュールを変える場合はキャッシュを作り直す。
- 2 step は等倍で織り目状の微細パターンが出る(3 step で解消)。
- 小さな文字は編集で崩れることがある(量子化起因か本家の限界かは未切り分け)。

## 経緯

diffusers-movie-server(MiniMax-H3 動画 + LTX-2.5)の第 3 バックエンドとして実装したものを、
単独で動く画像サーバとして切り出したもの。調査・実測の記録は元リポジトリの `probes/h3-image/RESULTS.md`。

---

### English summary

Standalone server + browser UI for MiniMax-H3-Image (33B text-to-image / reference editing) running
in ~21GB of VRAM: the AdaLN projections (39% of the weights) are replaced by an exact precomputed
table (the sampling timesteps are fixed), the remaining linears are torchao int8 weight-only, and the
pDMD 2-step LoRA is merged for a 3-step default (1.8s @1024², 4.1s @2048² on an RTX PRO 6000).
`scripts/prepare_models.py` downloads the model, the LoRA, and builds the nf4 text encoder.
Model license: MiniMax-H3-Image Community License (see the model card).
