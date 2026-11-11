"""MiniMax-H3-Image サーバ(単独、既定 port 8633)。

API(非同期ジョブ方式。diffusers-movie-server のバックエンド版と同一):
  GET  /api/health           {status, model, loaded, ...}
  GET  /api/status           health + engine 状態 + busy
  POST /api/assets           参照画像アップロード → {id, kind, filename, size}
  POST /api/jobs             生成ジョブ投入(202)→ JobResponse
  GET  /api/jobs?limit=N     履歴(新しい順)
  GET  /api/jobs/{id}        ジョブ状態
  DELETE /api/jobs/{id}
  DELETE /api/outputs/{filename}  生成物の削除(履歴のジョブは image_url が外れる)
  POST /api/prompts/enhance  {prompt, mode, conditions?} 公式 PE ルールでのプロンプト拡張(LLM 必要)
  POST /api/interrupt        {job_id?} 実行中ジョブの中断
  POST /api/admin/unload     VRAM 解放(プロセス温存、resident 切替用)。実行中は 409
  POST /api/admin/load       {steps?} 明示ロード
  GET  /outputs/...          生成物
"""

from __future__ import annotations

import logging
import queue
import uuid
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .config import STEP_PRESETS, settings
from .engine import Engine
from .enhance import EnhanceConnectionError, EnhanceError, EnhanceUnavailable, PromptEnhancer
from .jobs import JobManager
from pydantic import BaseModel

from .schemas import AssetResponse, ConditionInput, GenerateRequest, JobResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("h3image")

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

app = FastAPI(title="MiniMax-H3-Image server", version="0.1.0")
engine = Engine(settings)
manager = JobManager(settings, engine)
enhancer = PromptEnhancer(settings)

app.mount("/outputs", StaticFiles(directory=str(settings.output_dir)), name="outputs")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.on_event("startup")
def _startup():
    if settings.preload:
        engine.load()


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model": settings.model_id, "loaded": engine.loaded,
            "steps": engine.loaded_steps, "prompt_enhancer": enhancer.available}


@app.get("/api/status")
def status():
    return {**health(), "engine": engine.status(), "busy": bool(manager.active_ids()),
            "default_steps": settings.default_steps, "steps_available": sorted(STEP_PRESETS),
            "default_generation_area": settings.default_generation_area}


@app.post("/api/assets", response_model=AssetResponse, status_code=201)
async def upload_asset(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise HTTPException(status_code=415, detail="Supported inputs: PNG, JPG, WebP, BMP")
    asset_id = uuid.uuid4().hex
    tmp = settings.input_dir / f".{asset_id}{suffix}.part"
    dst = settings.input_dir / f"{asset_id}{suffix}"
    limit = settings.max_upload_mb * 1024 * 1024
    size = 0
    try:
        with tmp.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail=f"Upload exceeds {settings.max_upload_mb} MB")
                out.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        with Image.open(tmp) as im:
            im.verify()
        tmp.replace(dst)
    except HTTPException:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
    finally:
        await file.close()
    return AssetResponse(id=asset_id, kind="image", filename=file.filename or dst.name, size=size)


@app.post("/api/jobs", response_model=JobResponse, status_code=202)
def create_job(request: GenerateRequest):
    try:
        return manager.submit(request).public()
    except queue.Full:
        raise HTTPException(status_code=429, detail="Generation queue is full")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/jobs", response_model=list[JobResponse])
def list_jobs(limit: int = 50):
    return [j.public() for j in manager.history(limit)]


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.public()


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    if not manager.delete(job_id):
        raise HTTPException(status_code=404, detail="Job not found")


@app.delete("/api/outputs/{filename}")
def delete_output(filename: str):
    """生成物を削除する(履歴のジョブは image_url を外して残す)。"""
    if "/" in filename or "\\" in filename or ".." in filename or not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    target = (settings.output_dir / filename).resolve()
    if target.parent != settings.output_dir.resolve():
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Output not found")
    target.unlink()
    url = f"/outputs/{filename}"
    cleared = [j.id for j in manager.jobs.values() if j.image_url == url]
    for jid in cleared:
        manager.jobs[jid].image_url = None
    return {"result": "deleted", "filename": filename, "jobs_cleared": cleared}


class EnhanceRequest(BaseModel):
    prompt: str
    mode: str = "t2i"                       # t2i | edit(それ以外は t2i 扱い)
    conditions: list[ConditionInput] = []   # edit: 参照画像(アップロード順 = <Picture N>)
    aspect_ratio: Optional[str] = None      # edit: "W:H" を指定すると固定、省略で adaptive


@app.post("/api/prompts/enhance")
def enhance_prompt(req: EnhanceRequest):
    """公式 PE ルールで短い指示を英語の詳細プロンプト + 推奨比率へ(OpenAI 互換 LLM を使用)。"""
    mode = "edit" if req.mode == "edit" else "t2i"
    paths = []
    for c in req.conditions:
        p = manager._asset_path(c.asset_id)
        if p is None:
            raise HTTPException(status_code=400, detail=f"アセットが見つかりません: {c.asset_id}")
        paths.append(p)
    try:
        return enhancer.enhance(req.prompt, mode, paths, req.aspect_ratio)
    except EnhanceUnavailable as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except EnhanceConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except EnhanceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/interrupt")
def api_interrupt(job_id: Optional[str] = Body(None, embed=True)):
    return {"interrupted": manager.interrupt(job_id)}


@app.post("/api/admin/unload")
def admin_unload():
    active = manager.active_ids()
    if active:
        raise HTTPException(status_code=409, detail=f"Generation in progress ({len(active)} job(s)); cannot unload")
    return {"result": "unloaded", **engine.unload()}


@app.post("/api/admin/load")
def admin_load(steps: Optional[int] = Body(None, embed=True)):
    if steps is not None and steps not in STEP_PRESETS:
        raise HTTPException(status_code=400, detail=f"steps は {sorted(STEP_PRESETS)} のいずれか")
    if manager.active_ids():
        raise HTTPException(status_code=409, detail="Generation in progress; cannot reload")
    try:
        engine.load(steps)
    except Exception as exc:  # noqa: BLE001
        logger.exception("load failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    return {"result": "loaded", **engine.status()}
