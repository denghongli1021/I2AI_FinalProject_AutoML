"""
AutoML Backend API — Routes only
================================
業務邏輯都在 api/preprocess, api/train, api/visualize 三個資料夾裡。
這裡只負責:
  - FastAPI app 設定 + CORS
  - HTTP layer (請求驗證、ID 指派、in-memory store)
  - 把參數轉交給對應模組的 run()

啟動 (在專案根目錄):
  python -m uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import platform
import queue
import sys
import threading
import time
import uuid
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api import preprocess, train, visualize
from api.store import DATASETS, MODELS

app = FastAPI(title="AutoML API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "datasets": len(DATASETS),
        "models": len(MODELS),
        "engine": "python-sklearn",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


# ============================================================
# 1. PREPROCESS
# ============================================================
@app.post("/api/preprocess")
async def preprocess_endpoint(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    try:
        df, response = preprocess.run(raw, file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"preprocess 失敗: {e}")

    dataset_id = str(uuid.uuid4())
    DATASETS[dataset_id] = {
        "df": df,
        "fileName": file.filename,
        "loadedAt": time.time(),
    }
    return {"id": dataset_id, **response}


# ============================================================
# 2. TRAIN — 同步版本 (一次回傳所有結果)
# ============================================================
class TrainRequest(BaseModel):
    datasetId: str
    target: str
    features: list[str] | None = None
    algorithms: list[str]
    options: dict[str, Any] = {}


def _resolve_dataset(req: TrainRequest):
    bundle = DATASETS.get(req.datasetId)
    if bundle is None:
        raise HTTPException(status_code=404, detail="datasetId 不存在,請先呼叫 /api/preprocess")
    df = bundle["df"]
    if req.target not in df.columns:
        raise HTTPException(status_code=400, detail=f"target '{req.target}' 不在欄位中")
    return df


def _store_models(results) -> list[dict[str, Any]]:
    """把 train.run() 的 (bundle, estimator, scaler) 結果存進 MODELS,並回傳 bundle 列表。"""
    bundles = []
    for bundle, estimator, scaler in results:
        model_id = f"model_{uuid.uuid4().hex[:8]}"
        bundle["id"] = model_id
        MODELS[model_id] = {
            "bundle": bundle,
            "estimator": estimator,
            "scaler": scaler,
            "featureNames": bundle["featureNames"],
        }
        bundles.append(bundle)
    return bundles


@app.post("/api/train")
def train_endpoint(req: TrainRequest) -> dict[str, Any]:
    df = _resolve_dataset(req)
    try:
        results = train.run(df, req.target, req.features, req.algorithms, req.options)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"train 失敗: {e}")
    return {"models": _store_models(results)}


# ============================================================
# 2b. TRAIN (SSE) — 即時推送進度與 log
# ============================================================
@app.post("/api/train/stream")
def train_stream_endpoint(req: TrainRequest):
    df = _resolve_dataset(req)

    def event_stream():
        q: queue.Queue = queue.Queue()
        result_box: dict[str, Any] = {"results": None, "error": None}

        def worker():
            try:
                result_box["results"] = train.run(
                    df, req.target, req.features, req.algorithms, req.options,
                    on_progress=lambda ev: q.put(ev),
                )
            except Exception as e:
                result_box["error"] = str(e)
            finally:
                q.put(None)  # sentinel

        threading.Thread(target=worker, daemon=True).start()

        # 串流階段:把 worker thread 發的事件即時轉成 SSE
        while True:
            ev = q.get()
            if ev is None:
                break
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

        # 訓練完成 — 存 estimator,把 model bundles 一次推回
        if result_box["error"]:
            yield f"data: {json.dumps({'type': 'error', 'message': result_box['error']}, ensure_ascii=False)}\n\n"
        else:
            bundles = _store_models(result_box["results"])
            yield f"data: {json.dumps({'type': 'done', 'models': bundles}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================================
# 3. VISUALIZE
# ============================================================
class VisualizeRequest(BaseModel):
    modelId: str
    chartType: str
    options: dict[str, Any] = {}


@app.post("/api/visualize")
def visualize_endpoint(req: VisualizeRequest) -> dict[str, Any]:
    entry = MODELS.get(req.modelId)
    if entry is None:
        raise HTTPException(status_code=404, detail="modelId 不存在")
    return visualize.run(entry["bundle"], req.chartType, req.options)


# ============================================================
# 4. PREDICT — What-If Simulator 用
# ============================================================
class PredictRequest(BaseModel):
    modelId: str
    features: list[float]  # 原始尺度 (未標準化),順序需對應 featureNames


@app.post("/api/predict")
def predict_endpoint(req: PredictRequest) -> dict[str, Any]:
    entry = MODELS.get(req.modelId)
    if entry is None:
        raise HTTPException(status_code=404, detail="modelId 不存在")

    estimator = entry["estimator"]
    scaler = entry["scaler"]
    feature_names = entry["featureNames"]

    if estimator is None:
        raise HTTPException(status_code=400, detail="此模型訓練失敗,無法預測")
    if len(req.features) != len(feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"features 長度 {len(req.features)} 不符 (應為 {len(feature_names)})",
        )

    x = np.asarray([req.features], dtype=float)
    x_norm = scaler.transform(x)
    pred = estimator.predict(x_norm)[0]
    # numpy types 不是 JSON serializable
    if hasattr(pred, "item"):
        pred = pred.item()
    return {"prediction": pred}
