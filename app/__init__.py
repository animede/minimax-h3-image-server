"""パッケージ初期化: huggingface_hub を import する前に Windows 向けの環境変数を整える。

Windows では開発者モードが OFF(または非管理者)だと symlink を作れない。huggingface_hub はそれを検出して
コピーに切り替えるが、検出が「先に True を書いてから試す」順序なので、並列ダウンロード中に他スレッドが
symlink を試みて `OSError: [WinError 1314]` で落ちる(Diffusers_minimax-h3 の Windows 10 機で実測)。
Windows では symlink を使わない設定にして回避する(blob を snapshot 側へ移動するだけなのでディスクは増えない)。
明示的に `HF_HUB_DISABLE_SYMLINKS=0` が置かれていればそれを尊重する。
app.main / app.config / scripts/prepare_models.py のいずれも `app` パッケージを先に import するので、ここで足りる。
"""
import os as _os
import sys as _sys

if _sys.platform == "win32":
    _os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    _os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
