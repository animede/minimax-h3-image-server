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

## 低 VRAM 化・高速化の方法

本家は transformer 66GB(bf16)+ text_encoder 66.7GB(bf16)で、80GB GPU でも両方は載りません。
本サーバは次の 5 つを組み合わせて、transformer 側 ~19GB・text_encoder ~17GB に収め、6 step で 1.5 倍、
3 step で 2 倍以上速くしています。各項目の数字は実測(RTX PRO 6000、1024² / 2048²)。

### 1. AdaLN 射影を「事前計算テーブル」に置き換える(VRAM −26GB、誤差ゼロ)

transformer の各ブロックには、時刻埋め込み(2688 次元)から 6 種 × 3 モダリティ分の変調パラメータ
(96768 次元)を作る線形層 `adaln_proj` があります。1 層 520MB × 50 層 = **26GB で、重み全体の 39%** を
占めますが、入力は **時刻だけ**の関数で、トークンには依存しません。

MiniMax-H3-Image はサンプリング時刻が固定です(6 step の蒸留モデルで、σ 列はチェックポイントに
埋め込まれ、ステップごとに専用の出力ヘッドを持つ)。参照画像の行に使う時刻も 1 値です。
つまり各ブロックの `adaln_proj` が受け取る入力は **7 通りしかない**ので、
その 7 通りの出力を先に計算してテーブル `[7, 96768]`(計 68MB)に置き換え、生成時は行を引くだけにします。

- 元の forward と同じ演算(`F.linear(silu(temb).to(bf16), W, b)`)を同じ device / dtype で実行するため
  **出力は bit 一致**(近似ではない)。ブロック 0 で `torch.equal` により検証しています。
- 計算量はほぼ変わらない(元々 7 行 × 50 層の小さな行列積)ので、**効果は VRAM とロード時間**。
- 時刻集合はキャッシュごとに固定。steps 2 / 3(別の σ 列)は別テーブル・別キャッシュで、未知の時刻が
  来たら黙って誤らず例外にします。
- 同じ発想のコミュニティ版(rank-8 の低ランク近似、"AdaLN pruned")もありますが、時刻が固定なら
  近似する必要はなく、厳密なテーブルで済みます。動画版 H3 で同じ構造を扱った経験から着想しました。

### 2. 残りの Linear を int8 重み量子化(VRAM 40GB → 19GB、速度 1.5 倍)

attention と FF の Linear 312 層(50 ブロック × 6 + token refiner 2 × 6)を torchao の
`Int8WeightOnlyConfig(PerRow)` で量子化します。fp32 で持つべき層(入力射影・時刻埋め込み・出力ヘッド・RoPE)
はモデル側の指定に従って量子化しません。

- bf16 シャード(66GB)を **1 テンソルずつ** GPU に上げて量子化するため、bf16 を CPU/GPU に全展開しません
  (ロード時のピークは常駐 +1.5GB、ホスト RAM は数 GB)。
- 量子化済みを `models/cache/` に保存し、2 回目以降は 12 秒で読み込みます。
- 単層ベンチでは bnb の LLM.int8 と同等でしたが、モデル全体では **int8wo が 1.6 倍速**
  (1024² で bnb 3.5s → 2.4s)。bnb は外れ値分解のオーバーヘッドが積み重なります。
- 罠: `accelerate.set_module_tensor_to_device` は dtype を渡さないと meta モデルの既定 dtype(fp32)に
  キャストします。これで norm 等が fp32 になり活性全体が fp32 で流れ、最初は 2 倍遅くなっていました。

### 3. text_encoder を nf4 + 51 層に刈り込む(66.7GB → 17GB)

条件付けに使う Qwen3-VL-32B は MiniMax-H3(動画)と同一の重みで、モデルは `hidden_states[50]`
(51 層目の出力)しか読みません。そこで config の `num_hidden_layers` を 51 にしてロードし
(層 51〜63 と lm_head は捨てる)、bitsandbytes nf4 で量子化して保存します。

- 50 層にすると最終 norm 後の値になって条件付けが変わるため、**51 層が正解**です。
- エンコードは 1 回 0.1〜0.5 秒しか掛からないので、速度への影響はほぼありません。
- 2 枚目の GPU があれば text_encoder だけそちらに置き、`hidden_states[50]` だけをホスト経由で
  transformer 側へ渡します(GPU 間 P2P コピーが黙って全ゼロを返す環境があったため、直接コピーは使いません)。

### 4. pDMD 2-step LoRA で 6 step → 3 step(速度 1.7 倍、品質ほぼ同等)

[akhaliq/minimax-h3-image-pdmd-2step](https://huggingface.co/akhaliq/minimax-h3-image-pdmd-2step)
(rank 128、Apache-2.0)を量子化前の bf16 重みに fp32 でマージし、shift 12 の σ 列
(3 step: `[1.0, 0.96, 0.857, 0.0]`)で動かします。

- 時間の 7〜9 割が denoise なので、ステップ数の削減が最も効きます(2048² 7.0s → 4.1s)。
- **2 step は等倍で織り目状の微細パターン**が出て肌が平坦になるため、プレビュー用。**3 step は格子が消え、
  髪の毛筋・ニットの編み目まで出て 6 step とほぼ同等**なのでこちらを既定にしました。
- LoRA は AdaLN を含まないので 1. のテーブルとは独立ですが、σ 列が変わるのでテーブル(キャッシュ)は別です。

### 5. 1024² は要求サイズで生成する(2〜3 倍速)

本家の実装は最低 4MP(2048² 相当)のキャンバスで生成して縮小するため、1024² を頼んでも 2048² と同じ
時間が掛かります。`generation_area: "requested"`(既定)で要求サイズのまま生成すると 1024² が 3.5s → 1.5s に
なります。品質は良好ですが、本家推奨の画質が欲しい場合は `"native"` を指定してください。

### 試して採らなかったこと

| 案 | 結果 |
|---|---|
| sage attention | トークン数は VAE 16× + 2×2 パッチ化で 32× 縮小され、1024² で約 1.1k、2048² で約 4.2k。attention は 2048² でも全体の約 6% で効果が小さい。1024² は GPU ではなく CPU 側のカーネル発行が律速(CUDA Graph 化が次の候補) |
| torchao int8wo + `torch.compile(dynamic=True)` | 毎回再コンパイルして使い物にならず(単層で 200〜3400ms)。静的形状での設計が必要 |
| コミュニティの GGUF / nvfp4 / W4A8 | ComfyUI 専用形式か専用カーネルが要るため、diffusers 経路では未採用。nvfp4 は Blackwell で int8 比 +31% の報告あり(今後の候補) |

### VRAM の内訳(3 step、2048² 生成時)

| | GB |
|---|---|
| transformer int8wo(312 層) | ~17.5 |
| AdaLN テーブル・fp32 層・norm | ~0.5 |
| VAE(bf16) | 2.3 |
| 生成時の活性・デコード | +1〜2 |
| **GPU0 合計(peak)** | **~21** |
| text_encoder nf4(別 GPU 可) | ~17 |

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
How: (1) the AdaLN projections only depend on the (fixed) sampling timesteps, so their outputs are
precomputed into a 68MB table with bit-exact results (−26GB); (2) the remaining 312 linears are int8
weight-only, quantized shard-by-shard without materializing bf16 (−21GB, 1.6× faster than bnb LLM.int8);
(3) the Qwen3-VL-32B text encoder is nf4 and truncated to 51 layers (the model reads hidden_states[50]);
(4) the pDMD LoRA makes 3 steps look like 6 (1.7×); (5) generating at the requested size instead of the
4MP minimum canvas makes 1024² 2–3× faster. `scripts/prepare_models.py` downloads the model and the LoRA
and builds the nf4 text encoder.
**Code: MIT. Model weights and outputs: MiniMax-H3-Image Community License (see the model card).**
