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

# Windows console defaults to cp950 — 強制 utf-8 才不會被隊友 module 裡的 emoji 噴掉
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import io

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from api import preprocess, train, visualize
from api.preprocess import (
    run_data_audit,
    preprocess_for_training,
    preprocess_for_inference,
)
from api.preprocess.core import AutoRouter
from api.store import DATASETS, MODELS, PREPROCESSORS
from api.visualize import AutoMLVisualizer

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

    # 隊友 preprocessing 模組的健康診斷 (target 之後從 /api/train 才會傳)
    try:
        response["auditReport"] = run_data_audit(df, None)
    except Exception as e:
        response["auditReport"] = {"error": str(e)}

    dataset_id = str(uuid.uuid4())
    DATASETS[dataset_id] = {
        "df": df,
        "fileName": file.filename,
        "loadedAt": time.time(),
    }
    return {"id": dataset_id, **response}


# ============================================================
# 1b. PREPROCESS — 隊友模組: audit / transform / inference
# ============================================================
class AuditRequest(BaseModel):
    datasetId: str
    target: str | None = None


@app.post("/api/preprocess/audit")
def preprocess_audit_endpoint(req: AuditRequest) -> dict[str, Any]:
    """快速健檢:不真的跑 pipeline,只回傳 audit + 欄位分類預覽。"""
    bundle = DATASETS.get(req.datasetId)
    if bundle is None:
        raise HTTPException(status_code=404, detail="datasetId 不存在")
    df = bundle["df"]
    if req.target and req.target not in df.columns:
        raise HTTPException(status_code=400, detail=f"target '{req.target}' 不在欄位中")

    try:
        audit = run_data_audit(df, req.target)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"audit 失敗: {e}")

    # 連帶跑 Router 預覽欄位分類 (排除 target)
    scan_df = df.drop(columns=[req.target]) if req.target else df
    try:
        router = AutoRouter(categorical_threshold=50, text_length_threshold=20)
        feature_groups = router.fit_predict(scan_df)
    except Exception as e:
        feature_groups = {"error": str(e)}

    return {"auditReport": audit, "featureGroups": feature_groups}


class TransformRequest(BaseModel):
    datasetId: str
    target: str
    testSize: float = 0.2


@app.post("/api/preprocess/transform")
def preprocess_transform_endpoint(req: TransformRequest) -> dict[str, Any]:
    """跑完整的 preprocess_for_training,把 fitted preprocessor 存起來。"""
    bundle = DATASETS.get(req.datasetId)
    if bundle is None:
        raise HTTPException(status_code=404, detail="datasetId 不存在")
    df = bundle["df"]
    if req.target not in df.columns:
        raise HTTPException(status_code=400, detail=f"target '{req.target}' 不在欄位中")

    try:
        X_train, X_test, y_train, y_test, fitted = preprocess_for_training(
            df, req.target, test_size=req.testSize,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"transform 失敗: {e}")

    feature_names = list(X_train.columns)
    preprocessor_id = f"pp_{uuid.uuid4().hex[:8]}"
    PREPROCESSORS[preprocessor_id] = {
        "preprocessor": fitted,
        "target": req.target,
        "datasetId": req.datasetId,
        "featureNames": feature_names,
        # 存切分結果給後續下載
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    # Router 分類預覽 (用於前端顯示哪些欄位被分到哪一桶)
    scan_df = df.drop(columns=[req.target])
    router = AutoRouter(categorical_threshold=50, text_length_threshold=20)
    feature_groups = router.fit_predict(scan_df)

    audit = run_data_audit(df, req.target)

    # 預覽前 10 列,float 轉乾淨 JSON
    preview_rows = X_train.head(10).fillna(0).values.tolist()
    preview_rows = [[float(v) if isinstance(v, (int, float, np.floating)) else v for v in row]
                    for row in preview_rows]

    return {
        "preprocessorId": preprocessor_id,
        "auditReport": audit,
        "featureGroups": feature_groups,
        "transformedFeatureNames": feature_names,
        "preview": {
            "columns": feature_names,
            "rows": preview_rows,
        },
        "trainSize": int(len(X_train)),
        "testSize": int(len(X_test)),
        "originalFeatureCount": int(df.shape[1] - 1),
        "transformedFeatureCount": int(X_train.shape[1]),
    }


@app.get("/api/preprocess/list")
def preprocess_list_endpoint() -> dict[str, Any]:
    """列出目前後端記憶體裡所有可用的 preprocessor (給實驗室「資料來源」下拉用)。"""
    items = []
    for pid, entry in PREPROCESSORS.items():
        ds_id = entry.get("datasetId")
        ds_bundle = DATASETS.get(ds_id) or {}
        items.append({
            "id": pid,
            "datasetId": ds_id,
            "fileName": ds_bundle.get("fileName"),  # 比 UUID 穩定的對應依據
            "target": entry.get("target"),
            "featureCount": len(entry.get("featureNames", [])),
            "trainSize": int(len(entry["X_train"])) if entry.get("X_train") is not None else 0,
            "testSize": int(len(entry["X_test"])) if entry.get("X_test") is not None else 0,
        })
    return {"preprocessors": items}


@app.get("/api/preprocess/download/{preprocessor_id}/{split}")
def preprocess_download_endpoint(preprocessor_id: str, split: str):
    """下載 train.csv / test.csv (處理後特徵 + 目標)。"""
    import pandas as pd
    entry = PREPROCESSORS.get(preprocessor_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="preprocessorId 不存在")
    if split not in ("train", "test"):
        raise HTTPException(status_code=400, detail="split 必須是 'train' 或 'test'")

    X = entry["X_train"] if split == "train" else entry["X_test"]
    y = entry["y_train"] if split == "train" else entry["y_test"]
    target = entry["target"]

    # 把 y 加在最右邊一欄,index reset 才不會欄位錯位
    df = X.reset_index(drop=True).copy()
    df[target] = pd.Series(list(y)).reset_index(drop=True)

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode("utf-8-sig")  # BOM 讓 Excel 開中文不亂碼

    filename = f"{split}_{preprocessor_id}.csv"
    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class InferenceRequest(BaseModel):
    preprocessorId: str
    rows: list[dict[str, Any]]  # 每筆 {column_name: value}


@app.post("/api/preprocess/inference")
def preprocess_inference_endpoint(req: InferenceRequest) -> dict[str, Any]:
    """用已 fit 好的 preprocessor 套用到新資料。"""
    import pandas as pd
    entry = PREPROCESSORS.get(req.preprocessorId)
    if entry is None:
        raise HTTPException(status_code=404, detail="preprocessorId 不存在")

    try:
        new_df = pd.DataFrame(req.rows)
        transformed = entry["preprocessor"].transform(new_df)
        # ColumnTransformer 可能回 sparse matrix
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return {
            "columns": entry["featureNames"],
            "rows": np.asarray(transformed).tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference 失敗: {e}")


# ============================================================
# 2. TRAIN — 同步版本 (一次回傳所有結果)
# ============================================================
class TrainRequest(BaseModel):
    datasetId: str
    target: str
    features: list[str] | None = None
    algorithms: list[str]
    options: dict[str, Any] = {}
    # 資料來源 — 可多選: "raw" (原始資料集) / "preprocessed" (已預處理資料)
    sources: list[str] = ["raw"]
    preprocessorId: str | None = None  # sources 含 "preprocessed" 時必填


def _resolve_dataset(req: TrainRequest):
    bundle = DATASETS.get(req.datasetId)
    if bundle is None:
        raise HTTPException(status_code=404, detail="datasetId 不存在,請先呼叫 /api/preprocess")
    df = bundle["df"]
    if req.target not in df.columns:
        raise HTTPException(status_code=400, detail=f"target '{req.target}' 不在欄位中")
    return df


def _prefix_event(ev: dict, label: str) -> dict:
    """在 log / progress 事件前面加上資料來源標籤。"""
    if ev.get("type") == "log":
        return {**ev, "msg": f"[{label}] {ev.get('msg', '')}"}
    if ev.get("type") == "progress":
        return {**ev, "step": f"[{label}] {ev.get('step', '')}"}
    return ev


def _run_sources(req: TrainRequest, on_progress=None):
    """依 req.sources 跑指定的資料來源,回傳合併後的
    [(bundle, estimator, scaler, X_test_df), ...]。bundle 已標上 dataSource。"""
    sources = req.sources or ["raw"]
    multi = len([s for s in sources if s in ("raw", "preprocessed")]) > 1
    combined: list = []

    # --- 來源 1: 原始資料集 ---
    if "raw" in sources:
        df = _resolve_dataset(req)
        emit = (lambda ev: on_progress(_prefix_event(ev, "原始"))) if on_progress else None
        raw_results = train.run(df, req.target, req.features, req.algorithms, req.options, on_progress=emit)
        for bundle, est, scaler, xtdf in raw_results:
            bundle["dataSource"] = "raw"
            bundle["dataSourceLabel"] = "原始資料"
            if multi:
                bundle["name"] = f"[原始] {bundle['name']}"
            combined.append((bundle, est, scaler, xtdf))

    # --- 來源 2: 已預處理資料 ---
    if "preprocessed" in sources:
        if not req.preprocessorId:
            raise HTTPException(status_code=400, detail="選了「已預處理資料」但未提供 preprocessorId")
        entry = PREPROCESSORS.get(req.preprocessorId)
        if entry is None:
            raise HTTPException(status_code=404, detail="preprocessorId 不存在,請重新執行預處理")
        emit = (lambda ev: on_progress(_prefix_event(ev, "預處理"))) if on_progress else None
        pp_results = train.run_prepared(
            entry["X_train"], entry["X_test"], entry["y_train"], entry["y_test"],
            entry["target"], req.algorithms, req.options, on_progress=emit,
        )
        for bundle, est, scaler, xtdf in pp_results:
            bundle["dataSource"] = "preprocessed"
            bundle["dataSourceLabel"] = f"預處理 ({req.preprocessorId})"
            bundle["preprocessorId"] = req.preprocessorId  # 批次預測時要用它把原始 CSV 轉換
            if multi:
                bundle["name"] = f"[預處理] {bundle['name']}"
            combined.append((bundle, est, scaler, xtdf))

    if not combined:
        raise HTTPException(status_code=400, detail="沒有指定有效的資料來源 (raw / preprocessed)")

    # 跨來源一起重新排序,方便直接比較
    combined.sort(key=lambda r: r[0]["metrics"].get("testScore", 0.0), reverse=True)
    return combined


def _store_models(results) -> list[dict[str, Any]]:
    """把 (bundle, estimator, scaler, X_test_df) 結果存進 MODELS,並回傳 bundle 列表。"""
    bundles = []
    for bundle, estimator, scaler, X_test_df in results:
        model_id = f"model_{uuid.uuid4().hex[:8]}"
        bundle["id"] = model_id
        MODELS[model_id] = {
            "bundle": bundle,
            "estimator": estimator,
            "scaler": scaler,
            "featureNames": bundle["featureNames"],
            "X_test_df": X_test_df,  # for SHAP visualizer
            "preprocessorId": bundle.get("preprocessorId"),  # 預處理來源模型才有
        }
        bundles.append(bundle)
    return bundles


@app.post("/api/train")
def train_endpoint(req: TrainRequest) -> dict[str, Any]:
    try:
        results = _run_sources(req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"train 失敗: {e}")
    return {"models": _store_models(results)}


# ============================================================
# 2b. TRAIN (SSE) — 即時推送進度與 log
# ============================================================
@app.post("/api/train/stream")
def train_stream_endpoint(req: TrainRequest):
    def event_stream():
        q: queue.Queue = queue.Queue()
        result_box: dict[str, Any] = {"results": None, "error": None}

        def worker():
            try:
                result_box["results"] = _run_sources(
                    req, on_progress=lambda ev: q.put(ev),
                )
            except HTTPException as he:
                result_box["error"] = str(he.detail)
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
# 3b. VISUALIZE (SHAP) — 隊友 AutoMLVisualizer,回傳 Plotly figure JSON
# ============================================================
class ShapRequest(BaseModel):
    modelId: str
    sampleIndex: int = 0
    targetFeature: str | None = None
    maxSamples: int = 200  # SHAP 算太多會很慢,做個上限


@app.post("/api/visualize/shap")
def visualize_shap_endpoint(req: ShapRequest) -> dict[str, Any]:
    if AutoMLVisualizer is None:
        raise HTTPException(
            status_code=500,
            detail="AutoMLVisualizer 未啟用 — 請安裝 shap + plotly: pip install shap plotly",
        )
    entry = MODELS.get(req.modelId)
    if entry is None:
        raise HTTPException(status_code=404, detail="modelId 不存在")

    estimator = entry.get("estimator")
    X_test_df = entry.get("X_test_df")
    feature_names = entry.get("featureNames", [])
    if estimator is None:
        raise HTTPException(status_code=400, detail="該模型訓練失敗,無法產生 SHAP")
    if X_test_df is None or len(X_test_df) == 0:
        raise HTTPException(status_code=400, detail="測試集不存在,可能是舊版訓練的模型,請重新訓練")

    # 樣本太多會卡 (SHAP 對所有樣本算 explanation),做個上限
    X_sub = X_test_df.head(req.maxSamples)

    sample_idx = max(0, min(req.sampleIndex, len(X_sub) - 1))
    target_feat = req.targetFeature if (req.targetFeature in feature_names) else feature_names[0]

    try:
        viz = AutoMLVisualizer(estimator, X_sub, output_dir=None)
        fig_global = viz.generate_beeswarm_plot(return_fig=True)
        fig_waterfall = viz.generate_waterfall_plot(sample_index=sample_idx, return_fig=True)
        fig_dependence = viz.generate_dependence_plot(target_feature=target_feat, return_fig=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP 計算失敗: {e}")

    # plotly fig.to_json() 是字串;JSON 化讓 FastAPI 自己 serialize 一次
    import plotly.io as pio
    def _f(fig):
        return json.loads(pio.to_json(fig))

    return {
        "modelId": req.modelId,
        "sampleIndex": sample_idx,
        "targetFeature": target_feat,
        "sampleCount": int(len(X_sub)),
        "featureNames": feature_names,
        "global": _f(fig_global),
        "waterfall": _f(fig_waterfall),
        "dependence": _f(fig_dependence),
    }


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


# ============================================================
# 4b. PREDICT (BATCH) — 上傳一份 CSV,用訓練好的模型整批預測,回傳含預測欄的 CSV
# ============================================================
@app.post("/api/predict/batch")
async def predict_batch_endpoint(
    modelId: str = Form(...),
    file: UploadFile = File(...),
    sampleFile: UploadFile | None = File(None),
) -> Response:
    import pandas as pd

    entry = MODELS.get(modelId)
    if entry is None:
        raise HTTPException(status_code=404, detail="modelId 不存在")
    estimator = entry["estimator"]
    scaler = entry["scaler"]
    feature_names = entry["featureNames"]
    preprocessor_id = entry.get("preprocessorId")
    if estimator is None:
        raise HTTPException(status_code=400, detail="此模型訓練失敗,無法預測")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV 解析失敗: {e}")

    if preprocessor_id:
        # ── 預處理來源的模型 ──
        # 上傳的 CSV 是「原始格式」(跟訓練資料同欄位),要先過同一個 preprocessor 轉換
        pp_entry = PREPROCESSORS.get(preprocessor_id)
        if pp_entry is None:
            raise HTTPException(
                status_code=400,
                detail="此模型對應的 preprocessor 已不存在 (後端可能重啟過),請重新執行預處理再訓練",
            )
        preprocessor = pp_entry["preprocessor"]
        pp_target = pp_entry["target"]
        # 丟掉目標欄 (CSV 若有帶),其餘原始欄位交給 preprocessor
        feat_df = df.drop(columns=[pp_target], errors="ignore")
        try:
            X_t = preprocessor.transform(feat_df)
            if hasattr(X_t, "toarray"):
                X_t = X_t.toarray()
            X = np.asarray(X_t, dtype=float)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"套用預處理失敗 — test.csv 的欄位需與訓練資料的原始欄位一致: {e}",
            )
    else:
        # ── 原始來源的模型 ── CSV 需直接含模型的特徵欄位
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            preview = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
            raise HTTPException(
                status_code=400,
                detail=f"CSV 缺少 {len(missing)} 個模型需要的特徵欄位: {preview}",
            )
        X = df[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    try:
        X_norm = scaler.transform(X)
        preds = estimator.predict(X_norm)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批次預測失敗: {e}")

    pred_list = [p.item() if hasattr(p, "item") else p for p in preds]

    if sampleFile is not None:
        # ── 有給範本 submission ── 輸出比照範本格式 (第一欄=ID,第二欄=預測欄)
        sample_raw = await sampleFile.read()
        try:
            sample_df = pd.read_csv(io.BytesIO(sample_raw))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"範本 submission 解析失敗: {e}")
        sample_cols = list(sample_df.columns)
        if len(sample_cols) < 2:
            raise HTTPException(status_code=400, detail="範本 submission 至少需要 2 欄 (ID 欄 + 預測欄)")
        id_col, pred_col = sample_cols[0], sample_cols[1]
        if id_col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"test.csv 缺少範本要求的 ID 欄位「{id_col}」",
            )
        out_df = pd.DataFrame()
        out_df[id_col] = df[id_col].values
        out_df[pred_col] = pred_list
        filename = "submission.csv"
    else:
        # ── 沒給範本 ── 預測結果接回原始 CSV 最右邊
        out_df = df.copy()
        out_df["prediction"] = pred_list
        base = (file.filename or "test.csv").rsplit(".", 1)[0]
        filename = f"{base}_predicted.csv"

    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode("utf-8-sig")  # BOM 讓 Excel 開中文不亂碼

    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
