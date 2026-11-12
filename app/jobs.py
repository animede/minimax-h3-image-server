"""非同期ジョブ(ltx25 と同じ API 形: queued → running → completed / failed / interrupted)。

同時実行 1(ワーカースレッド 1 本)。履歴はメモリ内(gateway 側が SQLite で永続化する)。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from PIL import Image

from .config import Settings
from .engine import Engine, Interrupted
from .schemas import GenerateRequest

logger = logging.getLogger("h3image.jobs")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    request: GenerateRequest
    status: str = "queued"
    progress: float = 0.0
    error: str | None = None
    image_url: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    generation_seconds: float | None = None
    peak_vram_gb: float | None = None
    steps_used: int | None = None
    profile: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id, "status": self.status, "progress": self.progress, "error": self.error,
            "image_url": self.image_url, "video_url": None,
            "request": self.request.model_dump(), "created_at": self.created_at,
            "updated_at": self.updated_at, "generation_seconds": self.generation_seconds,
            "peak_vram_gb": self.peak_vram_gb, "steps_used": self.steps_used, "profile": self.profile,
        }


class JobManager:
    def __init__(self, settings: Settings, engine: Engine) -> None:
        self.s = settings
        self.engine = engine
        self.jobs: dict[str, Job] = {}
        self._queue: queue.Queue[Job] = queue.Queue(maxsize=settings.max_queue)
        self._lock = threading.Lock()
        self.current: Job | None = None
        self._worker = threading.Thread(target=self._run, name="h3image-worker", daemon=True)
        self._worker.start()

    # -- API 側 ---------------------------------------------------------------

    def submit(self, request: GenerateRequest) -> Job:
        for c in request.conditions:
            if not self._asset_path(c.asset_id):
                raise ValueError(f"アセットが見つかりません: {c.asset_id}")
        if request.mode == "edit" and not request.conditions:
            raise ValueError("edit には参照画像(conditions)が 1 件以上必要です")
        if request.mode == "t2i" and request.conditions:
            raise ValueError("t2i は参照画像を受け付けません(編集は mode=edit)")
        job = Job(id=uuid.uuid4().hex, request=request)
        self._queue.put_nowait(job)  # 満杯なら queue.Full → main.py が 429 に変換
        with self._lock:
            self.jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def history(self, limit: int = 50) -> list[Job]:
        jobs = sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:max(1, min(limit, 500))]

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                return False
            if job.status in ("queued", "running"):
                job.cancel.set()
            del self.jobs[job_id]
        return True

    def interrupt(self, job_id: str | None = None) -> list[str]:
        """running(と queued)のジョブに中断要求を出す。job_id 省略時は全て。"""
        hit = []
        for job in list(self.jobs.values()):
            if job_id and job.id != job_id:
                continue
            if job.status in ("queued", "running"):
                job.cancel.set()
                hit.append(job.id)
        return hit

    def active_ids(self) -> list[str]:
        return [j.id for j in self.jobs.values() if j.status in ("queued", "running")]

    def _asset_path(self, asset_id: str):
        for p in self.s.input_dir.glob(asset_id + ".*"):
            return p
        return None

    # -- ワーカー --------------------------------------------------------------

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job.cancel.is_set():
                job.status, job.error, job.updated_at = "interrupted", "開始前に中断されました", utc_now()
                continue
            self.current = job
            job.status, job.updated_at = "running", utc_now()
            try:
                images = []
                for c in job.request.conditions:
                    p = self._asset_path(c.asset_id)
                    if p is None:
                        raise ValueError(f"アセットが見つかりません: {c.asset_id}")
                    images.append(Image.open(p).convert("RGB"))

                def progress(step: int, total: int, job=job):
                    # ステップ i の開始時に呼ばれる。デコードを残すため上限 0.9
                    job.progress = round(min(0.9, 0.05 + 0.85 * step / max(1, total)), 3)
                    job.updated_at = utc_now()

                image, stats = self.engine.generate(job.request, images, progress, job.cancel)
                name = f"{job.request.mode}_{job.id}.png"
                image.save(self.s.output_dir / name)
                job.image_url = f"/outputs/{name}"
                job.generation_seconds = stats["generation_seconds"]
                job.peak_vram_gb = stats["peak_vram_gb"]
                job.steps_used = stats["steps_used"]
                job.profile = stats.get("profile")
                job.progress, job.status = 1.0, "completed"
                logger.info("job %s completed: %s %dx%d steps=%s %.1fs peak=%.1fGB seed=%s",
                            job.id, job.request.mode, job.request.width, job.request.height,
                            stats["steps_used"], stats["generation_seconds"], stats["peak_vram_gb"], stats["seed"])
            except Interrupted as exc:
                job.status, job.error = "interrupted", str(exc)
                logger.info("job %s interrupted", job.id)
            except Exception as exc:  # noqa: BLE001
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
                logger.exception("job %s failed", job.id)
            finally:
                job.updated_at = utc_now()
                self.current = None
                self._queue.task_done()
