"""HF Hub のシャードを 1 つずつ取得し、使い終わったら消す(ディスクのピークを「常駐分 + シャード 1 つ」に抑える)。

text_encoder の量子化(bf16 66.7GB)と transformer のキャッシュ構築(bf16 62GB)は、どちらも bf16 シャードを
順に読んで量子化するだけなので、全シャードを先に揃えておく必要がない。ここでは
  fetch(shard)   : 無ければダウンロードして、そのパスを返す
  release(shard) : この ShardSource が自分でダウンロードしたものだけ消す(元からあったファイルは消さない)
の 2 つを提供し、ビルド側は「fetch → 読む → release」の順に呼ぶ。`keep=True` なら消さない。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("h3image.sources")


class ShardSource:
    def __init__(self, repo_id: str, subfolder: str, local_dir: Path | None, keep: bool = False):
        self.repo_id, self.subfolder, self.local_dir, self.keep = repo_id, subfolder, local_dir, keep
        self.downloaded: dict[str, Path] = {}

    def fetch(self, shard: str) -> Path:
        if self.local_dir is not None and (self.local_dir / shard).exists():
            return self.local_dir / shard
        from huggingface_hub import hf_hub_download

        logger.info("downloading %s/%s/%s", self.repo_id, self.subfolder, shard)
        p = Path(hf_hub_download(self.repo_id, f"{self.subfolder}/{shard}"))
        self.downloaded[shard] = p
        return p

    def release(self, shard: str) -> None:
        p = self.downloaded.pop(shard, None)
        if p is None or self.keep:
            return
        real = Path(os.path.realpath(p))
        size = real.stat().st_size if real.exists() else 0
        # Windows は mmap がまだ生きている(読んだテンソルの参照が GC されていない)ファイルを消せず
        # PermissionError(WinError 5)になる。GC して少し待って数回やり直し、それでも消せなければ残して続ける
        # (ディスクがシャード 1 つ分増えるだけで、構築は完走する)。
        import gc
        import time

        for q in (p, real):
            for attempt in range(6):
                try:
                    q.unlink()
                    break
                except FileNotFoundError:
                    break
                except PermissionError:
                    if attempt == 5:
                        logger.warning("could not remove %s (still mapped or open elsewhere); leaving it in place", q)
                        return
                    gc.collect()
                    time.sleep(1.0 + attempt)
        logger.info("removed %s (%.1fGB freed)", shard, size / 2**30)
