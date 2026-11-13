# minimax-h3-image-server

日本語 | [English](README.en.md)

[MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)(33B、静止画生成 +
最大 16 枚参照の編集)を **約 21GB の VRAM で、1024² を 1.8 秒・2048² を 4.1 秒**で動かす
単独の画像サーバ + ブラウザ UI です。**W4A8 量子化 + text_encoder の層ストリーミングで 16GB 1 枚でも動き、さらに transformer の
ブロックストリーミングで 8GB 級でも動きます**(VRAM を 1GB 空けたまま、1024² 1.7〜2.6 秒・2048² 3.4〜4.2 秒・
2048² 級の編集 6.2〜7.0 秒。ホスト RAM 32GB の PC でも動く設定あり。出力は常駐時とピクセル一致)。diffusers(main)ベース、コードは MIT ライセンス。

- 本家の bf16 は transformer 66GB + text_encoder 66.7GB で 80GB GPU にも同時に載りませんが、
  本サーバは **AdaLN 射影の事前計算テーブル(bit 一致・近似なし)+ int8 重み量子化 + nf4 text_encoder**で
  transformer 常駐 ~19GB、text_encoder ~17GB に収めています。
- **pDMD 2-step LoRA** をマージした 3 step を既定にし、6 step とほぼ同品質で 1.7 倍速。
- **構成は VRAM 予算から自動選択**(`H3IMG_VRAM_GB`、既定は GPU の総容量と実際の空きの小さい方。OOM 時は一段下へ自動で下げる)。
- 低 VRAM 構成(`32gb` 以下のプロファイル): transformer を comfy-kitchen の W4A8 で ~11GB にし
  (GEMM は int8 より速い)、text_encoder の 51 層を pinned ホスト RAM から 1 層ずつ流し、未使用の lm_head を
  捨て、埋め込み表を CPU に置いて常駐 ~0.3GB にする(出力は全 GPU 常駐時と bit 一致)。VAE も encode / decode
  の間だけ GPU に置く。**32GB 1 枚で peak ~29GB、16GB 1 枚で peak ~13GB**。
- さらに `H3IMG_TRANSFORMER_STREAM=1`(`8gb` プロファイルで自動)で transformer の 50 ブロックも pinned ホスト RAM から
  先読みしながら流し、**8GB 級(実使用 4.2GB)**まで下がる。先読みが効くので速度は 16GB 構成と同じ。
- 非同期ジョブ API と、生成・参照編集・履歴・ダウンロード・削除ができる UI。

## 目次

1. [要求 VRAM](#要求-vram)
2. [測定した生成速度](#測定した生成速度)
3. [導入](#導入)
4. [起動と使い方](#起動と使い方)(GPU・VRAM 予算・量子化の[起動例](#起動例gpuvram-予算量子化の指定))
5. [環境変数](#環境変数)
6. [低 VRAM 化・高速化の方法](#低-vram-化高速化の方法)
7. [プロンプトの書き方(本家の方針)](#プロンプトの書き方本家の方針)
8. [調査・実測の記録(要点)](#調査実測の記録要点)
9. [ライセンスの注意](#ライセンスの注意)

## 要求 VRAM

| 部品 | 常駐 | 備考 |
|---|---|---|
| transformer(int8wo + AdaLN テーブル)+ VAE | **~19GB** | 生成時 peak 20.8GB(1024²)〜21.4GB(2048² 編集)。実測は 4096² 未確認 |
| text_encoder(Qwen3-VL-32B、nf4・51 層) | **~17GB** | 2 枚目の GPU があれば自動でそちらへ(`H3IMG_TE_DEVICE`) |

構成は **VRAM 予算から自動で選びます**。予算は「`H3IMG_VRAM_GB`(既定 `auto` = GPU の総容量)」と
「ロード時の実際の空き − 予備 1GB(`H3IMG_VRAM_RESERVE_GB`、画面表示や他のアプリの分)」の小さい方です。
`H3IMG_VRAM_GB=16` のように起動時に指定すれば、大きい GPU でも 16GB 級の構成で動かせます。

| プロファイル | 構成 | 選ばれる条件(予算) | 実測 peak(torch) | 必須のホスト RAM |
|---|---|---|---|---|
| `48gb` | int8wo、TE 常駐(最高速) | ≥ 38.7GB(TE が別 GPU なら ≥ 21.7GB) | 37〜38.4GB | 2GB |
| `32gb` | W4A8、TE 常駐 | ≥ 30.5GB(別 GPU なら ≥ 13.6GB) | 28.7〜30.2GB | 2.5GB |
| `16gb` | W4A8 + TE 層ストリーミング + VAE スワップ | ≥ 13.3GB | 12.5〜13.0GB | 18.3GB(`mmap` なら 3.5GB) |
| `8gb` | 上記 + transformer ブロックストリーミング | それ未満 | 2.4〜2.9GB(VAE を GPU に置くと +1.4GB) | 27.2GB(`mmap` なら 12.4GB) |

ホスト RAM の `mmap` は text_encoder の原本をページキャッシュに置く設定で、搭載 RAM 48GB 未満の PC では既定で
選ばれます([下記](#vram-の予備とram-の少ない-pc-向けの設定))。

- GPU が 2 枚あれば TE は既定で 2 枚目に置きます(`H3IMG_TE_GPU`)。使う GPU は `H3IMG_GPU=1` のように番号で指定。
- 生成中に CUDA OOM が起きたら(ロード後に他のプロセスが VRAM を取った等)、**一段下のプロファイルへ自動で下げて
  やり直します**(`H3IMG_VRAM_FALLBACK=0` で無効)。
- `H3IMG_TRANSFORMER_QUANT` / `H3IMG_TE_STREAM` / `H3IMG_TRANSFORMER_STREAM` を明示すると予算より優先します。
- 選ばれたプロファイルは `/api/status` の `engine.profile` とジョブ結果の `profile` で確認できます。

W4A8 の 3 構成は GPU の空きを 23.5GB / 15.5GB / 7.3GB に制限した RTX PRO 6000(sm_120)で検証した値です
(本物の 8〜32GB カードでは未検証。カーネルは Triton/CUDA 実装なので他世代でも動く見込みですが確認していません)。

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

W4A8 構成(`ck-w4a8`、3 step、TE も同じ GPU):

| ケース | 32GB 構成(TE 常駐) | 16GB 構成(TE ストリーミング + VAE スワップ) | 8GB 構成(+ ブロックストリーミング) |
|---|---|---|---|
| T2I 1024² | 0.7s | 1.2s | 1.8s |
| T2I 2048² | 2.8s | 3.4s | 3.5s |
| 参照編集 1024×1536(参照 1 枚) | — | 5.0s | — |
| 参照編集 1664×2496(参照 1 枚) | 5.6s | 6.0〜6.3s | 6.1〜6.5s |
| T2I 2048² 6 step | — | 6.1s(24GB 相当で測定) | — |
| peak VRAM(torch) | 28.7〜30.2GB | 12.5〜13.1GB | 2.4〜2.9GB(プロセス実使用 max 4.2GB) |

W4A8 は int8wo より **GEMM が速い**(2048² で 4.1s → 2.8s)ので、TE ストリーミングの +1.1s を足しても
int8wo と同程度です。品質は同一 seed で int8wo と並べて目視した範囲では同等でした(細部は seed 相当に
変わります。下記[低 VRAM 化](#低-vram-化高速化の方法) 6 参照)。

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
  transformer キャッシュ 19GiB(int8wo)または 10.7GiB(ck-w4a8)× 使うステップ種。
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
./run.sh                        # http://127.0.0.1:8633/(既定はローカルのみ待ち受け)
H3IMG_HOST=0.0.0.0 ./run.sh     # LAN に公開する場合(認証は無いので信頼できるネットワークで)
H3IMG_PORT=8634 ./run.sh        # ポート変更
H3IMG_STEPS=6 ./run.sh          # 既定ステップを 6 に
```

### 起動例(GPU・VRAM 予算・量子化の指定)

何も指定しなければ、使う GPU の大きさと実際の空きから構成(プロファイル)を自動で選びます。
ロードした後に他のプロセスが VRAM を取って生成中に OOM になった場合も、一段下の構成に自動で下げてやり直します。

```bash
# 自動(既定): GPU0 の総容量と実際の空きの小さい方を予算にして 48gb / 32gb / 16gb / 8gb を選ぶ
./run.sh

# 使う GPU を番号で指定(transformer を GPU1 に、TE も同じ GPU に)
H3IMG_GPU=1 H3IMG_TE_GPU=1 ./run.sh

# GPU が 2 枚: transformer は GPU0、TE は GPU1(2 枚あればこれが既定)
H3IMG_GPU=0 H3IMG_TE_GPU=1 ./run.sh

# 予算を指定: 96GB の GPU でも 16GB 級の構成(W4A8 + TE ストリーミング)で動かす
H3IMG_VRAM_GB=16 ./run.sh

# 8GB 級の構成を強制(必須のホスト RAM は約 27GB、RAM 48GB 未満の PC では自動で mmap になり約 12GB)
H3IMG_VRAM_GB=8 ./run.sh

# 量子化だけ明示(予算より優先): W4A8 で TE は常駐(32GB 級の構成)
H3IMG_TRANSFORMER_QUANT=ck-w4a8 H3IMG_TE_STREAM=0 ./run.sh

# OOM 時の自動ダウングレードを止める(OOM をそのままエラーで返す)
H3IMG_VRAM_FALLBACK=0 ./run.sh
```

| 指定 | 選択肢 | 意味 |
|---|---|---|
| `H3IMG_GPU` / `H3IMG_TE_GPU` | GPU 番号 | transformer / text_encoder を置く GPU。TE 未指定なら 2 枚目があればそちら |
| `H3IMG_VRAM_GB` | `auto` / 数値 | 予算。`auto` = GPU 総容量。実際の空きと小さい方で下表のプロファイルを選ぶ |
| `H3IMG_TRANSFORMER_QUANT` | `auto` / `int8wo` / `ck-w4a8` | transformer の量子化(19GB / 11GB) |
| `H3IMG_TE_STREAM` | `auto` / `0` / `1` | text_encoder の層ストリーミング(常駐 17GB → 0.3GB) |
| `H3IMG_TRANSFORMER_STREAM` | `auto` / `0` / `1` | transformer のブロックストリーミング(常駐 11GB → 1GB) |
| `H3IMG_VRAM_FALLBACK` | `1` / `0` | 生成中の OOM で一段下のプロファイルへ自動で下げる |
| `H3IMG_VRAM_RESERVE_GB` | 数値(既定 1.0) | 必ず空けておく VRAM(画面表示などの分) |
| `H3IMG_TE_HOST` | `auto` / `pinned` / `mmap` | text_encoder の原本の置き場所(mmap は RAM 32GB の PC 向け) |
| `H3IMG_TRANSFORMER_HOST` | `auto` / `pinned` / `mmap` | transformer ブロックの原本の置き場所(mmap は RAM 16GB の PC 向け、ディスク速度に依存) |

| プロファイル | 量子化 | TE | transformer | 要る予算 | 速度(1024² / 2048²、TE は pinned) |
|---|---|---|---|---|---|
| `48gb` | int8wo | 常駐 | 常駐 | 38.7GB(TE が別 GPU なら 21.7GB) | 1.8s / 4.1s |
| `32gb` | W4A8 | 常駐 | 常駐 | 30.5GB(別 GPU なら 13.6GB) | 0.7s / 2.8s |
| `16gb` | W4A8 | ストリーミング | 常駐 | 13.3GB | 1.2s / 3.4s |
| `8gb` | W4A8 | ストリーミング | ストリーミング | それ未満 | 1.8s / 3.5s |

選ばれたプロファイルは起動ログ(`VRAM profile: ...`)、`/api/status` の `engine.profile`、ジョブ結果の `profile` で
確認できます。`auto` の各項目は予算から決まり、明示した項目はそれより優先されます。

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

## プロンプトの書き方(本家の方針)

本家は「**画像全体の詳細な英語の記述**」を期待します。2026-11-11 に本家 GitHub へ
[prompt-writing skill](https://github.com/MiniMax-AI/MiniMax-H3-Image/tree/main/skills/h3-image-prompt-writing)
(`SKILL.md` + `references/t2i-en.txt` / `edit-en.txt`、Claude Code 等のエージェント向け)が公開され、書き方の
ルールと完成例がまとまっています。以下はその要点(本サーバのような自前デプロイは API と違って拡張を挟まないので、
このルールで書いたプロンプトがそのまま入力になります)。

**共通のルール**
- **用途が先、スタイルは後**: 何の画像で何のためか → 主役とその状態・動作 → 最後にスタイル。
- **配置が先、細部は後**: 主役の数・位置・視点・余白を先に。密な内容は「上に見出し、左に写真、右に 3 段落」のように
  領域を名付けてから中身を埋める。
- **スタイルは見える細部で**: 「高級感」「cinematic」ではなく、色・素材・光・質感(「柔らかいサイド光、ヘアライン仕上げの
  金属の反射、たっぷりの余白」)。
- **画像に出る文字は全文を引用符で**: 言語・空白はそのまま、位置・大小の階層・改行も書く。コピーを作らせたい時は
  実際の文言を書く。
- **文字を出さないなら明示**: t2i では "The image contains no text, letters or numbers." を足す(勝手なタイトルや
  飾り文字を防ぐ)。編集では「他の文字は追加しない」と書き、元画像の文字を守る。
- **データは正確に**: 図表は値・単位・関係・期間を書く。例示データは例示と明記し、架空の数字を実データとして出さない。
- **格子は行×列で**: "2 rows and 3 columns" / "3×3 (rows × columns)"。不規則な配置はそのまま書く("3 panels on top
  and 2 below")。1 マスは "row 2, column 1"。

**参照画像で編集(edit)**
- 参照はアップロード順に `<Picture 1>`, `<Picture 2>` …。
- 1 段落で **定義 → 完成形と各参照の役割 → 変わる部分の詳細** の順。命令ではなく完成した状態を語る
  ("The former location is filled by continuous tabletop")。
- 定義: `The red cup on the left in <Picture 1> is <Subject 1>.` のように、独立した扱いが要る範囲ごとに `<Subject N>`。
  色だけ借りる参照は「色という性質」として定義し、形や素材は借りないと書く。
- 役割: 完成形を `The target image …` で始め、各 Subject に 1 つずつ関係を書く。
  完全保持は定型文 `<Subject N> is fully preserved.`、部分保持は「何が残り何が変わるか」、
  性質の転写は `<Subject 2> supplies the exact body color transferred to <Subject 1>.`、
  弱い参照は「借りる性質と変えてよい範囲」。移動・回転・複製・照明変更も「変更」に数える。
- 変更の詳細: 消すなら跡地に何があるか。移動と複製は区別する。文字を変えるなら行全体の完成した文言を書く。
  保持したものの見た目を書き直さない。
- 複数回の編集は 1 回 1 目標。「さっきのやつ」ではなく、今回使う画像・変えるもの・残すものを毎回書く。
- 本家の例(局所的な色変更):
  > The red cup on the left in <Picture 1> is <Subject 1>. The lemon on the right in <Picture 1> is <Subject 2>.
  > The target image shows <Subject 1> with a blue finish. <Subject 1> is partially preserved; its shape, material,
  > position, and scale remain while its surface color changes. <Subject 2> is fully preserved. …

**最終チェック(本家 skill の Final check)**
文字は全部引用して綴りどおりか / 数・位置・左右・前後・接触が要求どおりか / 値・単位・ラベルが揃っているか /
各参照に役割があり保持するものを名指ししたか / アスペクト比が構図に合うか / 「見える細部」で言い換えられる曖昧な
品質語が残っていないか。

アスペクト比は 1:3〜3:1。本家 API の推奨サイズ(約 16MP)に対し本サーバは 4MP(2048² 相当)を既定にしています。

**プロンプト拡張(本サーバの機能)**
UI の「拡張」ボタン / `POST /api/prompts/enhance` は、プロンプト拡張モデル
[MiniMaxAI/MiniMax-H3-Image-PE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE) 同梱の公式ルール
(`prompt/t2i/task_prompt.txt` / `prompt/edit/task_prompt.txt`、公式 API の拡張ステップで使われているもの)を
system prompt とし、さらに上記 skill のガイド本文(`references/t2i-en.txt` / `edit-en.txt`、
`H3IMG_ENHANCE_SKILL_GUIDE=0` で外せる)を補足として足して、OpenAI 互換の LLM(例: llama.cpp server の gemma 系)に
送ります。短い日本語の指示が公式ルール準拠の英語プロンプトと推奨アスペクト比になります。PE モデル本体
(FP4 約 19GB)を載せずに済む代わりに、精度は本家 PE に劣る可能性があります。編集モードでは参照画像も LLM に渡す
(マルチモーダル対応の LLM が必要)。設定は `H3IMG_LLM_BASE_URL` / `H3IMG_LLM_MODEL` / `H3IMG_LLM_API_KEY`
(環境変数または `.env`)。ルール文書は初回に PE リポジトリと GitHub から `models/prompts/` へ取得します。

## 環境変数

| 変数 | 既定 | 内容 |
|---|---|---|
| `H3IMG_HOST` / `H3IMG_PORT` | 127.0.0.1 / 8633 | 待ち受けアドレスとポート |
| `H3IMG_STEPS` | 3 | 既定ステップ(2 / 3 / 6) |
| `H3IMG_TRANSFORMER_QUANT` | auto | transformer の量子化(`int8wo` 19GiB / `ck-w4a8` ~11GB)。auto = プロファイルに従う |
| `H3IMG_TE_STREAM` | auto | text_encoder の層ストリーミング(+ lm_head 削除・埋め込み CPU 化・VAE スワップ)。0 / 1 で明示 |
| `H3IMG_TRANSFORMER_STREAM` | auto | transformer 50 ブロックのストリーミング(8GB 級)。0 / 1 で明示 |
| `H3IMG_GPU` / `H3IMG_TE_GPU` | 0 /(2 枚あれば 1) | transformer / text_encoder を置く GPU 番号(`H3IMG_DEVICE=cuda:N` / `H3IMG_TE_DEVICE` でも可) |
| `H3IMG_VRAM_GB` | auto | VRAM 予算(GB)。auto = GPU の総容量。実際の空きと小さい方で 48gb / 32gb / 16gb / 8gb プロファイルを選ぶ |
| `H3IMG_VRAM_FALLBACK` | 1 | 生成中の CUDA OOM で一段下のプロファイルへ自動で下げてやり直す |
| `H3IMG_VRAM_RESERVE_GB` | 1.0 | 予算から差し引いて必ず空けておく VRAM(画面表示・他のアプリ用) |
| `H3IMG_TE_HOST` | auto | text_encoder の原本: `pinned`(速い、RAM 約 13GB)/ `mmap`(ページキャッシュ、RAM の少ない PC 向け)。auto は搭載 48GB 以上で pinned |
| `H3IMG_TRANSFORMER_HOST` | auto | transformer ブロックの原本(8gb 構成): `pinned`(RAM 約 10GB)/ `mmap`(量子化キャッシュのページキャッシュから流す。RAM 16GB の PC 向け、毎生成でディスクから読み直す)。auto は TE と同じ規則 |
| `H3IMG_TRANSFORMER_DROP_BLOCKS` | auto | `mmap` のとき使い終わったらページを捨てる末尾ブロック数。auto = キャッシュに使える RAM からはみ出す分だけ。0 で捨てない |
| `H3IMG_GENERATION_AREA` | requested | 既定キャンバス(`native` = 最低 4MP) |
| `H3IMG_PRELOAD` | 0 | 1 で起動時にロード |
| `H3IMG_MODELS_DIR` | `./models` | te / lora / cache の置き場 |
| `H3IMG_TE_DIR` / `H3IMG_LORA_PATH` / `H3IMG_CACHE_DIR` / `H3IMG_MODEL_DIR` | 上記配下 / HF キャッシュ | 個別の上書き |
| `H3IMG_MAX_PIXELS` | 4096² | 1 リクエストの上限画素 |
| `H3IMG_MAX_QUEUE` / `H3IMG_MAX_UPLOAD_MB` | 4 / 200 | キュー長 / アップロード上限 |
| `H3IMG_LLM_BASE_URL` / `H3IMG_LLM_MODEL` / `H3IMG_LLM_API_KEY` | (未設定=拡張無効) | プロンプト拡張に使う OpenAI 互換 LLM(`.env` にも書ける) |
| `H3IMG_LLM_TIMEOUT_S` | 180 | LLM のタイムアウト |
| `H3IMG_ENHANCE_SKILL_GUIDE` | 1 | 拡張時に本家 prompt-writing skill のガイド本文も system に足す |

## 低 VRAM 化・高速化の方法

本家は transformer 66GB(bf16)+ text_encoder 66.7GB(bf16)で、80GB GPU でも両方は載りません。
本サーバは次の 5 つを組み合わせて、transformer 側 ~19GB・text_encoder ~17GB に収め、6 step で 1.5 倍、
3 step で 2 倍以上速くしています。さらに 6・7・8 で 32GB / 16GB / 8GB の 1 枚構成に対応します。
各項目の数字は実測(RTX PRO 6000、1024² / 2048²)。

### 全体の考え方(133GB → 4GB まで、何をどこに置くか)

VRAM を減らす手段は「小さくする」「GPU に置かない」「使う瞬間だけ置く」の 3 種類で、本サーバは部品ごとに
使い分けています。鍵は **GPU は計算の場所であって保管場所ではない**、という見方です。

| 部品 | 本家 | 手段 | 結果 |
|---|---|---|---|
| AdaLN 射影(transformer の 39%) | 26GB | 入力が固定(サンプリング時刻)なので**出力を事前計算**して重みを捨てる(1) | 68MB、誤差ゼロ |
| 残りの Linear 312 層 | 40GB | **小さくする**: int8(2)→ W4A8(6) | 19GB → 11GB |
| transformer ブロック | 11GB | **使う瞬間だけ置く**: 50 ブロックを pinned ホスト RAM から先読み(8) | GPU 常駐 1GB |
| text_encoder(Qwen3-VL-32B) | 66.7GB | **小さくする**: nf4 + 使わない 13 層を切る(3)→ **置かない**: lm_head は未使用なので捨てる、埋め込み表は CPU で引く(7) | 17GB → 0.3GB |
| text_encoder の 51 層 | 11.6GB | **使う瞬間だけ置く**: 1 層ずつ流す(7) | GPU 常駐 0 |
| VAE | 1.4GB | **使う瞬間だけ置く**: encode / decode の間だけ GPU(7) | 常駐 0 |

「使う瞬間だけ置く」が速度を落とさない理由は 2 つあります。text_encoder はプロンプト 1 本につき 1 回しか通らない
ので、層を流すコストは生成全体で +1 秒程度です。transformer は毎ステップ 50 ブロック全部を通りますが、
ブロック i の計算中に i+1 を別の CUDA stream でコピーしておけば(先読み)、PCIe の転送は計算の裏に隠れます。
その結果 8GB 構成でも 16GB 構成と同じ速度で、出力はピクセル単位で一致します。

どの手段をどこまで使うかは VRAM 予算([起動例](#起動例gpuvram-予算量子化の指定))から自動で決めます。
「置かない / 使う瞬間だけ置く」ものはホスト RAM に残るので、8GB 構成ではプロセスのホスト RAM 使用量が約 29GB になります
(text_encoder をページキャッシュに置く `mmap` 設定なら必須分は約 12GB)。

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

### 6. W4A8(4bit 重み・int8 活性)で transformer を ~11GB にする(32GB 級向け、GEMM 1.5 倍速)

`H3IMG_TRANSFORMER_QUANT=ck-w4a8`(`32gb` 以下のプロファイルで自動)。ComfyUI が実装した W4A8 の実体である
[comfy-kitchen](https://pypi.org/project/comfy-kitchen/)(Apache-2.0、Triton/CUDA カーネル)を使い、
2 の 312 層を「重み 4bit(group16 のスケール + 16 値 codebook)・活性 int8(回転付き)」にします。
int8wo の 19GiB が 10.7GiB になり、GEMM 単体は bf16 の 1.5 倍、int8wo より速い(2048² 4.1s → 2.8s)。
pDMD LoRA は fp32 でマージしてから量子化します(2 と同じ)。キャッシュは `models/cache/ckw4a8_adaln_table*`。

- 品質: 同一 seed で int8wo と並べて目視した範囲では同等(肌・髪・看板の文字とも破綻なし)。
  ただし蒸留 3 step は量子化誤差で構図が seed 相当に動くので、ピクセル一致は期待できません。
- 検証は sm_120(Blackwell)のみ。カーネル自体は Ampere 以降を想定した実装ですが、他世代は未確認。

### 7. text_encoder の 51 層を 1 層ずつ流す(常駐 17GB → 0.3GB、16GB 級向け)

`H3IMG_TE_STREAM=1`(`16gb` 以下のプロファイルで自動)。nf4 の decoder 51 層を最初から CPU に読み(GPU に載せて
から退避すると、ロードの時点で 16GB を超える)、pinned ホスト RAM から forward hook で 1 層(~0.3GB)ずつ
GPU へ流します。bitsandbytes の nf4 は量子化状態(absmax / code)も一緒に原本から流すので、
**hidden_states[50] は全 GPU 常駐時と bit 一致**(実測、最大誤差 0.0)。あわせて:

- **lm_head(1.45GB)を捨てる**: H3-Image は hidden_states[50] しか読まないので使われていません。
- **埋め込み表(bf16 1.45GB)を CPU に置く**: 参照結果だけを GPU へ送ります(数 ms)。
- GPU に残るのは vision tower・norm・RoPE の ~0.3GB。
- **VAE(2.3GB)は encode / decode の間だけ GPU に置く**: 参照画像付きの編集では TE が画像トークン分の活性を
  使うため、VAE が常駐していると 16GB 級では TE のエンコード中に OOM しました(実測)。往復は 0.1〜0.2s。

プロンプトは 1 回しか通らないので、層を流すコストは生成全体で 1 秒未満です。16GB 構成の 2048² は 3.4 秒で、
int8wo(48GB 構成)の 4.1 秒より速くなります(W4A8 の GEMM が速いため)。プロセスのホスト RAM 使用量は約 19GB。

### 8. transformer の 50 ブロックを先読みしながら流す(常駐 11GB → 1GB、8GB 級向け)

`H3IMG_TRANSFORMER_STREAM=1`(`8gb` プロファイルで自動)。W4A8 化した transformer_blocks(各 ~0.2GB)を
pinned ホスト RAM に置き、forward hook でブロック i の計算中に i+1 を別の CUDA stream でコピーします。
GPU に残るのは token refiner・入出力射影などの ~1GB。コピーが計算に隠れるので、**速度は 16GB 構成とほぼ同じ**
(2048² 3.5s)、出力は常駐時と**ピクセル一致**です。プロセスのホスト RAM 使用量は約 29GB です(下記)。

- 重みは書き換わらないので GPU → CPU の書き戻しはしません。pre hook で GPU へコピーした tensor を
  `param.data` に差し、post hook で pinned のマスターへ戻すだけです。
- 罠: 先読みは別 stream で行うため、GPU 側の tensor を解放するときに `record_stream(計算 stream)` を付けないと、
  計算が終わる前にアロケータがその領域を次の先読みに使い回し、**出力が真っ黒になります**(実測)。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(run.sh で既定)で reserved の膨張を抑えます
  (プロセス実使用 6.8GB → 4.2GB)。

### ホスト RAM: 切り上げないピン留めと、書き戻さない流し方

GPU から外したものはホスト RAM に置きます。最初の実装は 8gb 構成で **51GB** も使っていました。原因は 3 つで、
いずれも直してあります(`app/pruned_image.py` の `pin_module` / `stream_layers`)。

1. **`Tensor.pin_memory()` の切り上げ**: PyTorch のピン留めアロケータは確保サイズを切り上げます(実体 24.6GB に
   対して 33.1GB)。層・ブロックごとに 1 本の匿名メモリを確保し、`cudaHostRegister` でその場でピン留めして、
   各テンソルをそのビューに差し替えるようにしました。切り上げは起きません。登録は unload の前に解除します。
2. **text_encoder を `.to()` で往復させていた**: `mod.to("cpu")` で戻すと毎回新しい(ピン留めされていない)CPU
   テンソルができ、1 回目以降は転送も遅くなっていました。transformer と同じく「ピン留めした原本を保持し、GPU 側は
   捨てるだけ」の方式にしました。bitsandbytes の量子化状態(absmax / code)も一緒に流します。
3. **safetensors の mmap が残っていた**: text_encoder の重みを原本へ移すと mmap は参照を失って解放されます
   (埋め込み表も複製して切り離す)。量子化キャッシュも `torch.load(mmap=True)` で読み、全体をホストへ展開しません。

2048² を生成した後のプロセスのメモリ(RSS):

| プロファイル | 改善前 | 改善後 |
|---|---|---|
| `48gb` | 2.1GB | 2.2GB |
| `16gb` | 32〜35GB | **18.8GB** |
| `8gb` | 50〜51GB | **29.2GB** |

ロード中に増えるのは mmap のページキャッシュ(OS が回収できる)だけで、匿名メモリの最大は生成後と同じです。
**8gb 構成はホスト RAM 32GB でぎりぎり、48GB 以上なら余裕があります。16gb 構成は 32GB で動きます。**
副次効果として、text_encoder の転送がピン留めのまま行われるようになり、16gb / 8gb 構成が速くなりました
(1024² 2.6s → 1.2s / 1.8s、2048² 4.1〜4.2s → 3.4〜3.5s)。

また、ロード時に PyTorch のアロケータが予約できる量を VRAM 予算までに制限しています
(`torch.cuda.set_per_process_memory_fraction`)。ストリーミングで解放が遅れても、予算を超えて VRAM を抱え込みません。

### VRAM の予備と、RAM の少ない PC 向けの設定

**VRAM の予備(`H3IMG_VRAM_RESERVE_GB`、既定 1GB)**: 予算を「空き − 予備」で計算し、画面表示や他のアプリのために
必ず 1GB を空けておきます。予算に余りがあれば、ストリーミング構成でも VAE(1.4GB)と text_encoder の埋め込み表
(1.45GB)を GPU に置きます(往復とホスト RAM が減る)。PyTorch のアロケータが予約できる量も予算までに制限します。

**text_encoder の原本の置き場所(`H3IMG_TE_HOST`)**:

- `pinned`: ピン留めしたホスト RAM(約 13GB を常に使う)。速い。
- `mmap`: safetensors ファイルのページキャッシュから直接流す。ページキャッシュは OS がいつでも回収できる
  メモリなので、**必須のホスト RAM が約 13GB 減ります**。代わりにピン留めしない分、生成が 0.7〜0.9 秒遅くなります。
- `auto`(既定): 搭載 RAM が 48GB 以上なら `pinned`、未満なら `mmap`。

8GB / 16GB のカードを模擬した実測(GPU の空きを 7.3GB / 15.5GB に制限、予備 1GB。出力はすべて同一):

| カード | TE 原本 | プロセスの VRAM | 実行中の GPU の空き(最小) | 必須のホスト RAM | 1024² | 2048² | 編集 1664×2496 |
|---|---|---|---|---|---|---|---|
| 8GB | mmap | 5.9GB(VAE も GPU) | 1.5GB | **12.4GB**(+ ページキャッシュ 14GB) | 2.6s | 4.2s | 7.0s |
| 8GB | pinned | 5.9GB | 1.5GB | 27.2GB | 1.7s | 3.4s | 6.2s |
| 16GB | mmap | 14.1GB | 1.7GB | **3.5GB**(+ ページキャッシュ 14GB) | 2.2s | 4.4s | 7.2s |
| 16GB | pinned | 14.1GB | 1.7GB | 18.3GB | 1.3s | 3.5s | 6.2s |

**RAM 32GB の PC では `mmap`(既定で自動選択)**、48GB 以上なら `pinned` で速くなります。

### RAM 16GB・GPU 6GB の PC(transformer もページキャッシュから流す)

`H3IMG_TRANSFORMER_HOST=mmap` にすると、transformer の 50 ブロックも量子化キャッシュのページキャッシュから直接流し、
必須のホスト RAM は **約 1.8GB**(プロセスの匿名メモリ)まで下がります。代わりに、RAM に収まらない分は
**生成のたびにディスクから読み直す**ので、速度はディスクの連続読み出し速度で決まります。

実測(RTX 4000 SFF Ada 20GB の空きを 6GB に制限、プロセスのメモリ上限 12GB = RAM 16GB の PC 相当、
NVMe は Gen4 で連続読み 2.3GB/s、予備 1GB、出力は pinned 構成とピクセル一致):

| 構成 | 1024² | 2048² | 編集 1664×2496 | 1 枚あたりのディスク読み込み | 必須 RAM |
|---|---|---|---|---|---|
| pinned(RAM 制限なし、同じ Ada カード) | 8.2s | 12.7s | 24.3s | 0 | 27GB |
| mmap × mmap、RAM 12GB 上限(現行) | **12.5〜12.8s** | **16.6s** | **28.9s** | 13〜14GB | 1.8GB |
| 同上、RAM 11GB 上限(末尾 6 ブロックを捨てる) | 13.5s | 17.3s | — | 16.7GB | 1.8GB |
| 同上、RAM 10GB 上限(末尾 11 ブロックを捨てる) | 13.9s | 17.6s | — | 19.7GB | 1.8GB |
| 参考: 使い終わった層を捨てない場合 | 17.8〜21.3s | 20.5〜23.5s | 31.2s | 24〜27GB | 1.8GB |

- 遅くなる分はディスクの速度そのものです(読み込み量 ÷ 2.3GB/s)。`madvise(MADV_WILLNEED)` による先読みは、
  ディスクが既に限界なので効きませんでした。連続読み 7GB/s 級の NVMe なら差は 2 秒程度に縮む計算です。SATA SSD では
  +25 秒前後になります。
- **text_encoder の層は 1 回しか読まない**ので、使い終わった層のページをその場でページキャッシュから捨てます
  (`madvise(MADV_PAGEOUT)`)。これを入れないと、13GB の TE が通過するたびに transformer 10GB のページが押し出され、
  毎回 24GB を読み直していました。捨てるようにすると transformer はキャッシュに残り、読み直しは TE の 13GB だけになります
  (RAM 12GB 上限でも成立。出力は不変)。
- **transformer がキャッシュに収まりきらないときは、はみ出す分の末尾ブロックだけを使い終わったら捨てます**
  (`H3IMG_TRANSFORMER_DROP_BLOCKS`、既定 `auto`)。全部が LRU で押し出されて毎ステップ読み直す(RAM 11GB 上限で 24.6GB)
  より、決まった K ブロックを毎ステップ読み直す(0.2GB × K × 3 step)方がずっと安く、上限が 1GB 下がるごとに
  +0.4〜0.5 秒ずつ緩やかに遅くなります。K はページキャッシュに使える RAM(cgroup の上限も見る)から自動で決めます。
  GPU に置いた部品(token refiner など)の mmap ページも、ロード後に捨ててキャッシュの枠を空けます。
- 上の Ada の数値は、このプロジェクトで初めての Blackwell 以外・70W の小型カードでの実測です(x4 接続)。W4A8 カーネル
  (comfy-kitchen)は sm_89 でも動き、単層で bf16 の 1.9 倍速でした。Blackwell とは数値が異なるので出力はピクセル一致
  しません(平均差 7/255 程度、構図は同じ)。

### 試して採らなかったこと

| 案 | 結果 |
|---|---|
| sage attention | トークン数は VAE 16× + 2×2 パッチ化で 32× 縮小され、1024² で約 1.1k、2048² で約 4.2k。attention は 2048² でも全体の約 6% で効果が小さい。1024² は GPU ではなく CPU 側のカーネル発行が律速(CUDA Graph 化が次の候補) |
| torchao int8wo + `torch.compile(dynamic=True)` | 毎回再コンパイルして使い物にならず(単層で 200〜3400ms)。静的形状での設計が必要 |
| コミュニティの GGUF / nvfp4 | ComfyUI 専用形式か専用カーネルが要るため、diffusers 経路では未採用。nvfp4 は Blackwell で int8 比 +31% の報告あり(今後の候補)。W4A8 は comfy-kitchen を直接呼ぶ形で採用した(6) |

### VRAM の内訳(3 step、2048² 生成時)

| | GB |
|---|---|
| transformer int8wo(312 層) | ~17.5 |
| AdaLN テーブル・fp32 層・norm | ~0.5 |
| VAE(bf16) | 2.3 |
| 生成時の活性・デコード | +1〜2 |
| **GPU0 合計(peak)** | **~21** |
| text_encoder nf4(別 GPU 可) | ~17 |

W4A8 構成: transformer ck-w4a8 ~10.7 + テーブル等 0.5 + VAE 2.3 + 活性 → **~13〜14GB**。
TE を同じ GPU に常駐させると +16.3GB(peak 29.1GB、32GB 級)。層ストリーミング + lm_head 削除 +
埋め込み CPU 化 + VAE スワップなら TE 分は +0.3〜0.9GB で **peak 12.5〜13.1GB(16GB 級)**。
transformer のブロックストリーミングを足すと transformer 分が ~1GB + 先読み 2 ブロックになり **torch peak 2.4〜2.9GB、
プロセス実使用 4.2GB(8GB 級)**。

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

## 調査・実測の記録(要点)

実装前に本家モデルを調査し、量子化方式の比較を実機で行いました。要点のみ記します。

### モデルについて分かったこと

- text_encoder(Qwen3-VL-32B)は MiniMax-H3(動画)と **14 シャード全てハッシュ一致**。どちらも
  `hidden_states[50]` を条件付けに使う。動画用に量子化した text_encoder がそのまま使える。
- diffusers では Modular Pipeline としてのみ提供(`MiniMaxH3ImageModularPipeline`)。本書執筆時点で
  main ブランチのみ(0.42.1 には未収録)。
- サンプリングは 6 step 固定(σ 列はチェックポイント内、ステップごとに専用の出力ヘッド)。CFG なし。
- 既定では最低 4MP のキャンバスで生成して縮小するため、`min_generation_area` を指定しないと
  1024² の要求も 2048² と同じ時間が掛かる。
- トークン数は VAE 16× と 2×2 パッチ化で 32× 縮小: 1024² ≈ 1.1k、2048² ≈ 4.2k、4096² ≈ 16.5k。

### 時間の内訳(3 step・int8wo、ロード済み)

| ケース | 合計 | text_encoder | transformer | VAE decode |
|---|---|---|---|---|
| T2I 1024² | 2.05s(6 step) | 0.11 | 1.74 | 0.17 |
| T2I 2048² | 7.04s(6 step) | 0.12 | 6.02 | 0.83 |
| 編集 2048² | 11.4s(6 step) | 0.45 | 9.76 | 0.82(+encode 0.25) |

時間の 7〜9 割が transformer(denoise)。text_encoder と VAE は小さく、text_encoder の軽量化は
VRAM 目的のみ。1024² では GPU より CPU 側のカーネル発行が律速になる(CUDA Graph 化が今後の候補)。

### 品質の所見(等倍目視)

- テキスト生成は写真・アニメ調とも良好。参照編集は人物・キャラクターの同一性と衣装・小物をよく保つ。
- 編集で**背景の小さな文字が崩れる**ことがある(大きな文字は保たれる)。量子化起因か本家の限界かは未切り分け。
- int8wo の出力は bnb int8 と同一シードで構図・細部とも同等(差は量子化方式由来の微差のみ)。
- 2 step は織り目状の微細パターンが出る。3 step で解消し 6 step に近い。

### 実装上の罠(実機で踏んだもの)

- `ModularPipeline._execution_device` は components の先頭の nn.Module のデバイスを返すため、
  text_encoder を別 GPU に置くと実行デバイスがそちらになる → プロパティを上書きして固定。
- GPU 間の受け渡しは必ずホスト経由(P2P コピーが黙って全ゼロを返す環境があった)。
- 新しい huggingface_hub は `snapshot_download(local_files_only=True)` でも欠けたファイルを検査して
  例外にする。一部だけ落としたスナップショットは `snapshots/<commit>` を直接参照する。
- `accelerate.set_module_tensor_to_device` は dtype 未指定だと fp32 にキャストする(活性が fp32 化し 2 倍遅い)。
- `init_empty_weights` はバッファ(`rope.inv_freq`)を CPU に残す → 明示的に移す。
- torchao int8wo の素の `torch.compile(dynamic=True)` は毎回再コンパイルする。
- steps 切替で旧 transformer をパイプライン属性と forward ラッパーのクロージャが掴んだままになり
  二重常駐 → 参照を切ってから gc する。

### 未検証

- bf16 本体との品質比較(文字崩れの切り分け)、4096² 以上の VRAM・時間、複数参照(2 枚以上)の編集、
  nvfp4、CUDA Graph 化、実物の 8〜32GB カードでの計測。

## 経緯

diffusers-movie-server(MiniMax-H3 動画 + LTX-2.5 の統合サーバ)の第 3 バックエンドとして実装したものを、
単独で動く画像サーバとして切り出したものです。

---

English: see [README.en.md](README.en.md).
