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

# 必須最先 import — 處理 Windows UTF-8 終端 + 載入 .env,後面 db.py 才讀得到 DATABASE_URL
from api import bootstrap  # noqa: F401

import asyncio
import io
import json
import platform
import queue
import sys
import threading
import time
import uuid
from typing import Any, Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from api import preprocess, train, visualize
from api import storage
from api.auth import init_db, router as auth_router, get_current_user
from api.auth.db import get_db
from api.preprocess import (
    run_data_audit,
    preprocess_for_training,
    preprocess_for_inference,
)
from api.preprocess.core import AutoRouter
# 注意:DATASETS/MODELS/PREPROCESSORS in-memory dicts 還是被 storage 層的 guest 路徑用,
# 不過 main.py 自己已經完全靠 storage helpers,不再直接 import 那些 dict。
from api.visualize import AutoMLVisualizer
from sqlalchemy.orm import Session as DbSession

app = FastAPI(title="AutoML API", version="0.1.0")

# CORS — 允許 credentials 走 (給 Authorization header 用);origins 用 regex 含 localhost / GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 啟動時建表 (User table)
init_db()

# 掛 auth router (/api/auth/*)
app.include_router(auth_router, prefix="/api")


@app.get("/api/health")
def health() -> dict[str, Any]:
    # 不再回 DATASETS/MODELS 計數 — 那是 guest 路徑用的 in-memory dict,跟登入使用者
    # 看到的內容無關。要看真的用量需要 user_id,health 是 public endpoint 拿不到。
    from api.store import DATASETS as _DATASETS, MODELS as _MODELS
    return {
        "ok": True,
        "guestDatasets": len(_DATASETS),
        "guestModels": len(_MODELS),
        "engine": "python-sklearn",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


# ============================================================
# 1. PREPROCESS — CSV 上傳 → DataFrame + analysis,寫入 storage
# ============================================================
@app.post("/api/preprocess")
async def preprocess_endpoint(
    file: UploadFile = File(...),
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
) -> dict[str, Any]:
    raw = await file.read()
    try:
        df, response = preprocess.run(raw, file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"preprocess 失敗: {e}")

    try:
        response["auditReport"] = run_data_audit(df, None)
    except Exception as e:
        response["auditReport"] = {"error": str(e)}

    dataset_id = storage.save_dataset(
        df=df, file_name=file.filename, csv_bytes=raw,
        response=response, user=user, db=db,
    )
    return {"id": dataset_id, **response}


# ============================================================
# 1c. DATASET listing/fetching — 給前端登入時把使用者已上傳的 CSV 還原回來
# ============================================================
@app.get("/api/dataset/list")
def dataset_list_endpoint(
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
) -> dict[str, Any]:
    """列出當前 user 的所有 datasets,只回 metadata。"""
    return {"datasets": storage.list_datasets(user, db)}


@app.get("/api/dataset/{dataset_id}")
def dataset_get_endpoint(
    dataset_id: str,
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
) -> dict[str, Any]:
    """取單一 dataset 的完整資料 (含 rows / analysis / correlation 等)。"""
    bundle = storage.get_dataset(dataset_id, user, db, include_df=False)
    resp = bundle.get("response")
    if not resp:
        raise HTTPException(status_code=500, detail="dataset response 快取遺失,請重新上傳")
    return {"id": dataset_id, **resp}


@app.delete("/api/dataset/{dataset_id}")
def dataset_delete_endpoint(
    dataset_id: str,
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
) -> dict[str, Any]:
    """刪除 dataset + 級聯刪掉它的 preprocessors + models。"""
    return storage.delete_dataset(dataset_id, user, db)


# ============================================================
# 1b. PREPROCESS — 隊友模組: audit / transform / inference
# ============================================================
class AuditRequest(BaseModel):
    datasetId: str
    target: str | None = None


@app.post("/api/preprocess/audit")
def preprocess_audit_endpoint(
    req: AuditRequest,
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
) -> dict[str, Any]:
    """快速健檢:不真的跑 pipeline,只回傳 audit + 欄位分類預覽。"""
    bundle = storage.get_dataset(req.datasetId, user, db, include_df=True)
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
    useMice: bool = False           # MICE (IterativeImputer) 補值開關
    useMiSelection: bool = False    # MI 互資訊特徵選擇開關
    miThreshold: float = 0.01       # MI 篩選門檻


@app.post("/api/preprocess/transform")
def preprocess_transform_endpoint(
    req: TransformRequest,
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
) -> dict[str, Any]:
    """跑完整的 preprocess_for_training,把 fitted preprocessor 存起來。"""
    bundle = storage.get_dataset(req.datasetId, user, db, include_df=True)
    df = bundle["df"]
    if req.target not in df.columns:
        raise HTTPException(status_code=400, detail=f"target '{req.target}' 不在欄位中")

    try:
        X_train, X_test, y_train, y_test, fitted = preprocess_for_training(
            df, req.target, test_size=req.testSize,
            use_mice=req.useMice,
            use_mi_selection=req.useMiSelection,
            mi_threshold=req.miThreshold,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"transform 失敗: {e}")

    feature_names = list(X_train.columns)
    preprocessor_id = storage.save_preprocessor(
        preprocessor=fitted, target=req.target, dataset_id=req.datasetId,
        feature_names=feature_names,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        user=user, db=db, test_size=req.testSize,
        use_mice=req.useMice,
        use_mi_selection=req.useMiSelection,
        mi_threshold=req.miThreshold,
    )

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
        # v3 透明度:實際自動觸發的進階特徵工程 + MI 篩選結果
        "appliedFeatureSteps": getattr(fitted, "applied_feature_steps_", []),
        "miSelection": getattr(fitted, "mi_selection_", None),
    }


@app.get("/api/preprocess/list")
def preprocess_list_endpoint(
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
) -> dict[str, Any]:
    """列出當前 user 的所有 preprocessor (給實驗室「資料來源」下拉用)。"""
    return {"preprocessors": storage.list_preprocessors(user, db)}


@app.get("/api/preprocess/download/{preprocessor_id}/{split}")
def preprocess_download_endpoint(
    preprocessor_id: str,
    split: str,
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
):
    """下載 train.csv / test.csv (處理後特徵 + 目標)。"""
    import pandas as pd
    entry = storage.get_preprocessor(preprocessor_id, user, db)
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
def preprocess_inference_endpoint(
    req: InferenceRequest,
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
) -> dict[str, Any]:
    """用已 fit 好的 preprocessor 套用到新資料。"""
    import pandas as pd
    entry = storage.get_preprocessor(req.preprocessorId, user, db)

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


def _resolve_dataset(req: TrainRequest, user, db: DbSession):
    bundle = storage.get_dataset(req.datasetId, user, db, include_df=True)
    df = bundle["df"]
    if req.target not in df.columns:
        raise HTTPException(status_code=400, detail=f"target '{req.target}' 不在欄位中")
    return df, bundle.get("fileName") or "dataset"


def _prefix_event(ev: dict, label: str) -> dict:
    """在 log / progress 事件前面加上資料來源標籤。"""
    if ev.get("type") == "log":
        return {**ev, "msg": f"[{label}] {ev.get('msg', '')}"}
    if ev.get("type") == "progress":
        return {**ev, "step": f"[{label}] {ev.get('step', '')}"}
    return ev


def _run_sources(req: TrainRequest, user, db: DbSession, on_progress=None, cancel_token=None):
    """依 req.sources 跑指定的資料來源,回傳 (combined results, dataset_name)。
    cancel_token (threading.Event):每個演算法之間檢查,set 後 break — 已訓練的模型保留。"""
    sources = req.sources or ["raw"]
    multi = len([s for s in sources if s in ("raw", "preprocessed")]) > 1
    combined: list = []
    dataset_name = None

    # --- 來源 1: 原始資料集 ---
    if "raw" in sources:
        if cancel_token is not None and cancel_token.is_set():
            return combined, dataset_name or "dataset"
        df, dataset_name = _resolve_dataset(req, user, db)
        emit = (lambda ev: on_progress(_prefix_event(ev, "原始"))) if on_progress else None
        raw_results = train.run(df, req.target, req.features, req.algorithms, req.options,
                                on_progress=emit, cancel_token=cancel_token)
        for bundle, est, scaler, xtdf in raw_results:
            bundle["dataSource"] = "raw"
            bundle["dataSourceLabel"] = "原始資料"
            if multi:
                bundle["name"] = f"[原始] {bundle['name']}"
            combined.append((bundle, est, scaler, xtdf))

    # --- 來源 2: 已預處理資料 ---
    if "preprocessed" in sources:
        if cancel_token is not None and cancel_token.is_set():
            if not combined:
                return [], dataset_name or "dataset"
            # 已有 raw 的結果,直接帶走
            combined.sort(key=lambda r: r[0]["metrics"].get("testScore", 0.0), reverse=True)
            return combined, dataset_name or "dataset"
        if not req.preprocessorId:
            raise HTTPException(status_code=400, detail="選了「已預處理資料」但未提供 preprocessorId")
        entry = storage.get_preprocessor(req.preprocessorId, user, db)
        if dataset_name is None and entry.get("datasetId") == req.datasetId:
            # raw 沒勾,從 dataset bundle 拿名字
            try:
                ds_bundle = storage.get_dataset(req.datasetId, user, db, include_df=False)
                dataset_name = ds_bundle.get("fileName")
            except HTTPException:
                pass
        emit = (lambda ev: on_progress(_prefix_event(ev, "預處理"))) if on_progress else None
        pp_results = train.run_prepared(
            entry["X_train"], entry["X_test"], entry["y_train"], entry["y_test"],
            entry["target"], req.algorithms, req.options,
            on_progress=emit, cancel_token=cancel_token,
        )
        for bundle, est, scaler, xtdf in pp_results:
            bundle["dataSource"] = "preprocessed"
            bundle["dataSourceLabel"] = f"預處理 ({req.preprocessorId})"
            bundle["preprocessorId"] = req.preprocessorId
            if multi:
                bundle["name"] = f"[預處理] {bundle['name']}"
            combined.append((bundle, est, scaler, xtdf))

    if not combined:
        raise HTTPException(status_code=400, detail="沒有指定有效的資料來源 (raw / preprocessed)")

    combined.sort(key=lambda r: r[0]["metrics"].get("testScore", 0.0), reverse=True)
    return combined, dataset_name or "dataset"


def _store_models(results, req: TrainRequest, user, db: DbSession, training_run_id: str | None = None) -> list[dict[str, Any]]:
    """把 (bundle, estimator, scaler, X_test_df) 結果存進 storage,並回傳 bundle 列表。
    每個 model 失敗都記錄並繼續,避免單一 model 壞掉整批回不到前端。"""
    if not results:
        raise HTTPException(status_code=500, detail="_store_models 收到空結果 (results 為 None 或空 list)")
    bundles = []
    fail_count = 0
    for i, (bundle, estimator, scaler, X_test_df) in enumerate(results):
        try:
            model_id = storage.save_model(
                bundle=bundle, estimator=estimator, scaler=scaler, X_test_df=X_test_df,
                user=user, db=db,
                preprocessor_id=bundle.get("preprocessorId"),
                dataset_id=req.datasetId,
                training_run_id=training_run_id,
                hyperparameters=bundle.get("hyperparameters", {}),
            )
            bundle["id"] = model_id
            bundles.append(bundle)
        except Exception as e:
            # 印詳細錯誤到 uvicorn log,但不讓整個 batch 死掉
            import traceback as _tb
            print(f"[_store_models] save_model 第 {i+1}/{len(results)} 筆失敗 "
                  f"(algo={bundle.get('type') or bundle.get('name')}): {type(e).__name__}: {e}",
                  flush=True)
            _tb.print_exc()
            # DB session 可能進入髒狀態,rollback 才能繼續用
            try: db.rollback()
            except Exception: pass
            # 沒 id 還是讓 bundle 回到前端 (至少看得到 metrics)
            bundle["id"] = None
            bundle["_storeError"] = f"{type(e).__name__}: {e}"
            bundles.append(bundle)
            fail_count += 1
    if fail_count and fail_count == len(results):
        # 全失敗 → 整批爛了,讓上層當錯誤處理
        raise HTTPException(status_code=500, detail=f"所有模型存檔失敗 (最後一筆: {bundles[-1].get('_storeError')})")
    return bundles


@app.post("/api/train")
def train_endpoint(
    req: TrainRequest,
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
) -> dict[str, Any]:
    t_start = time.time()
    run_id = storage.create_training_run(
        dataset_id=req.datasetId, dataset_name="(loading)",
        engine="sklearn", target=req.target,
        task_type=req.options.get("taskType", "auto") if req.options else "auto",
        sources=req.sources, options=req.options or {},
        user=user, db=db,
    )
    try:
        results, dataset_name = _run_sources(req, user, db)
    except HTTPException as he:
        storage.finish_training_run(run_id, user, db, status="failed", error_msg=str(he.detail))
        raise
    except Exception as e:
        storage.finish_training_run(run_id, user, db, status="failed", error_msg=str(e))
        raise HTTPException(status_code=500, detail=f"train 失敗: {e}")

    bundles = _store_models(results, req, user, db, training_run_id=run_id)
    # 更新 training run 為 completed + 補 dataset_name + top models
    storage.finish_training_run(
        run_id, user, db, status="completed",
        results_summary={"topModels": [{"id": b.get("id"), "name": b.get("name"),
                                         "score": b.get("metrics", {}).get("testScore", 0.0)}
                                        for b in bundles[:5]],
                         "datasetName": dataset_name},
        model_ids=[b.get("id") for b in bundles if b.get("id")],
        elapsed_sec=round(time.time() - t_start, 2),
    )
    return {"models": bundles}


# ============================================================
# 2c. TRAIN (Daniel Pipeline) — 測試模式專用,跑 Daniel 的完整 AutoML pipeline
#     HPO → NAS → 5-Fold CV → Nelder-Mead Blend + Meta-Learner Stack
# ============================================================
@app.get("/api/train/pipeline/benchmark")
def pipeline_benchmark_endpoint() -> dict[str, Any]:
    """讀取 Daniel 的批次評估結果 CSV,給前端做 pipeline vs baseline 對照表。"""
    import os
    import pandas as pd
    csv_path = os.path.join(
        os.path.dirname(__file__), "train", "pipeline", "pipeline_batch_results.csv",
    )
    if not os.path.isfile(csv_path):
        raise HTTPException(status_code=404, detail="benchmark CSV 不存在")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"讀取 CSV 失敗: {e}")
    # nan 在 JSON 不合法,先轉成 None
    df = df.where(pd.notna(df), None)
    rows = df.to_dict(orient="records")
    return {"rows": rows, "count": len(rows)}


@app.post("/api/train/pipeline/stream")
async def train_pipeline_stream_endpoint(
    # 模式 A:測試模式直接上傳 CSV
    file: UploadFile | None = File(None),
    # 模式 B:實驗室從現有 dataset / preprocessor 跑
    datasetId: str | None = Form(None),
    sources: str | None = Form(None),
    preprocessorId: str | None = Form(None),
    # Option B:訓練時上傳「想預測的 test.csv」(無 label) — 訓練完直接回預測結果
    predictFile: UploadFile | None = File(None),
    # 通用 pipeline options
    target: str | None = Form(None),
    timeSeries: bool = Form(False),
    metric: str = Form("f1"),
    fast: bool = Form(True),
    timeLimit: float = Form(0),
    skipTabular: bool = Form(False),
    skipDl: bool = Form(False),
    noNas: bool = Form(False),
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
):
    """Pipeline 訓練 endpoint。詳見 docstring。"""
    from api.train.daniel_runner import run_pipeline as run_daniel_pipeline
    import io as _io

    # 解析 sources
    src_list: list[str] = []
    if sources:
        try:
            src_list = json.loads(sources)
        except Exception:
            raise HTTPException(status_code=400, detail="sources 必須是 JSON list")
    if not isinstance(src_list, list):
        raise HTTPException(status_code=400, detail="sources 必須是 list")

    direct_upload = file is not None
    from_store = datasetId is not None
    if not direct_upload and not from_store:
        raise HTTPException(status_code=400, detail="需提供 file 或 datasetId")
    if direct_upload and from_store:
        raise HTTPException(status_code=400, detail="file 與 datasetId 擇一即可")

    # 預先讀 predictFile (Option B) — 訓練完拿這份 CSV 對每個 result 做預測
    predict_csv_bytes: bytes | None = None
    predict_csv_name: str | None = None
    if predictFile is not None:
        predict_csv_bytes = await predictFile.read()
        predict_csv_name = predictFile.filename or "predict_input.csv"

    jobs: list[dict[str, Any]] = []
    base_options = {
        "target": target, "timeSeries": timeSeries, "metric": metric, "fast": fast,
        "timeLimit": timeLimit, "skipTabular": skipTabular, "skipDl": skipDl, "noNas": noNas,
    }
    dataset_name_for_run = "(uploaded)"

    if direct_upload:
        raw = await file.read()
        dataset_name_for_run = file.filename or "uploaded.csv"
        # 即使是 direct upload 也支援 predictFile (override 內部 split)
        if predict_csv_bytes:
            jobs.append({
                "label": "上傳", "source": "upload",
                "train_csv_bytes": raw,
                "test_csv_bytes": predict_csv_bytes,
                "train_file_name": dataset_name_for_run,
                "options": base_options,
                "predict_input_bytes": predict_csv_bytes,
            })
        else:
            jobs.append({
                "label": "上傳", "source": "upload",
                "csv_bytes": raw, "file_name": dataset_name_for_run,
                "options": base_options,
            })
    else:
        if not src_list:
            src_list = ["raw"]

        if "raw" in src_list:
            bundle = storage.get_dataset(datasetId, user, db, include_df=True)
            df = bundle["df"]
            dataset_name_for_run = bundle.get("fileName") or "dataset.csv"
            if target and target not in df.columns:
                raise HTTPException(status_code=400, detail=f"target '{target}' 不在 dataset 欄位中")
            buf = _io.StringIO()
            df.to_csv(buf, index=False)
            train_csv = buf.getvalue().encode("utf-8")

            if predict_csv_bytes:
                # 有上傳 predict.csv → 走 pre-split 模式,test_csv = predict.csv
                jobs.append({
                    "label": "原始", "source": "raw",
                    "train_csv_bytes": train_csv,
                    "test_csv_bytes": predict_csv_bytes,
                    "train_file_name": dataset_name_for_run,
                    "options": base_options,
                    "predict_input_bytes": predict_csv_bytes,
                })
            else:
                jobs.append({
                    "label": "原始", "source": "raw",
                    "csv_bytes": train_csv,
                    "file_name": dataset_name_for_run,
                    "options": base_options,
                })

        if "preprocessed" in src_list:
            if not preprocessorId:
                raise HTTPException(status_code=400, detail="選了 'preprocessed' 但未提供 preprocessorId")
            pp_entry = storage.get_preprocessor(preprocessorId, user, db)
            if dataset_name_for_run == "(uploaded)":
                try:
                    ds_bundle = storage.get_dataset(datasetId, user, db, include_df=False)
                    dataset_name_for_run = ds_bundle.get("fileName") or "dataset.csv"
                except HTTPException:
                    pass
            X_train_df = pp_entry["X_train"]
            X_test_df = pp_entry["X_test"]
            y_train = pp_entry["y_train"]
            y_test = pp_entry["y_test"]
            pp_target = pp_entry["target"]
            pp_obj = pp_entry["preprocessor"]

            # 大寬矩陣 (高基數欄 One-Hot 後動輒數千欄) 直接 df.to_csv(檔案路徑) 串流寫到暫存檔,
            # 不要 .copy() / StringIO / .encode() — 那會把整份矩陣在記憶體裡多複製好幾份 → OOM
            # (7375 欄 × 3 萬列一份就 1.68 GiB)。寫完即 del 釋放,讓峰值只剩「載入的那一份」。
            # 暫存檔路徑直接傳給 run_pipeline (它支援路徑模式),省掉 bytes 來回。
            import tempfile as _tempfile
            _safe = bundle_safe_name(pp_target)

            def _df_to_temp_csv(frame, prefix):
                with _tempfile.NamedTemporaryFile(
                    mode="w", suffix=".csv", delete=False,
                    prefix=prefix, encoding="utf-8", newline="",
                ) as _tf:
                    _path = _tf.name
                frame.to_csv(_path, index=False)
                return _path

            # train:就地加上 target 欄 (剛 unpickle 出來、無人共用,可安全 mutate)
            X_train_df[pp_target] = list(y_train)
            train_csv_path = _df_to_temp_csv(X_train_df, f"daniel_{_safe}_train_")
            del X_train_df
            pp_entry["X_train"] = None

            # 決定 test CSV:有 predict_csv 時用使用者上傳的 (套用同一個 preprocessor)
            predict_input_for_job: bytes | None = None
            if predict_csv_bytes:
                import pandas as _pd
                try:
                    predict_df_raw = _pd.read_csv(_io.BytesIO(predict_csv_bytes))
                    # 預處理器 fit 時看的是不含 target 的 X,所以丟掉 target (若有)
                    predict_feat_raw = predict_df_raw.drop(columns=[pp_target], errors="ignore")
                    transformed = pp_obj.transform(predict_feat_raw)
                    if hasattr(transformed, "toarray"):
                        transformed = transformed.toarray()
                    # train/test 同一個 preprocessor → 欄位一致,用 X_test 的欄名即可
                    transformed_df = _pd.DataFrame(transformed, columns=list(X_test_df.columns))
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"套用預處理到 predict CSV 失敗 — 欄位需與原始訓練資料一致: {e}",
                    )
                test_csv_path = _df_to_temp_csv(transformed_df, f"daniel_{_safe}_test_")
                del transformed_df
                predict_input_for_job = predict_csv_bytes
            else:
                X_test_df[pp_target] = list(y_test)
                test_csv_path = _df_to_temp_csv(X_test_df, f"daniel_{_safe}_test_")
            del X_test_df
            pp_entry["X_test"] = None

            pp_options = dict(base_options)
            pp_options["target"] = pp_target

            jobs.append({
                "label": "預處理", "source": "preprocessed",
                "train_csv_path": train_csv_path,
                "test_csv_path": test_csv_path,
                "train_file_name": f"{_safe}.csv",
                "options": pp_options,
                "preprocessorId": preprocessorId,
                "predict_input_bytes": predict_input_for_job,
                # run_pipeline 不會刪 caller 給的路徑 → 由 worker 的 finally 收尾刪除
                "_temp_paths": [train_csv_path, test_csv_path],
            })

    if not jobs:
        raise HTTPException(status_code=400, detail="沒有產生任何訓練任務")

    multi = len(jobs) > 1

    # 建 TrainingRun (running 狀態,完成後再 update)
    pipeline_run_id = storage.create_training_run(
        dataset_id=datasetId or "uploaded",
        dataset_name=dataset_name_for_run,
        engine="pipeline",
        target=target or "(auto)",
        task_type="classification",
        sources=src_list if from_store else ["upload"],
        options={
            "metric": metric, "fast": fast, "timeSeries": timeSeries,
            "skipDl": skipDl, "noNas": noNas, "skipTabular": skipTabular,
            "timeLimit": timeLimit, "preprocessorId": preprocessorId,
            "hasPredictFile": predict_csv_bytes is not None,
        },
        user=user, db=db,
    )
    # 上傳的 predict.csv 存成 'input' artifact (auth 才有 run_id)
    if predict_csv_bytes and pipeline_run_id:
        storage.save_prediction_artifact(
            training_run_id=pipeline_run_id, kind="input",
            file_name=predict_csv_name or "predict_input.csv",
            content_bytes=predict_csv_bytes, user=user, db=db,
        )

    t_start = time.time()

    def event_stream():
        q: queue.Queue = queue.Queue()
        all_results: list[dict[str, Any]] = []
        error_msg: str | None = None
        cancel_token = threading.Event()   # 前端 SSE 斷線時 set,worker 偵測後 terminate subprocess

        def prefix(ev, label):
            if not multi:
                return ev
            if ev.get("type") == "log":
                return {**ev, "msg": f"[{label}] {ev.get('msg', '')}"}
            if ev.get("type") == "progress":
                return {**ev, "step": f"[{label}] {ev.get('step', '')}", "source": label}
            return ev

        def _build_submission_csv(predict_input_bytes: bytes, predictions: list, target_name: str) -> bytes:
            """從 predict_input 拿第一欄當 ID,接上預測欄位,輸出 submission.csv bytes。"""
            import pandas as _pd
            predict_df = _pd.read_csv(_io.BytesIO(predict_input_bytes))
            sub = _pd.DataFrame()
            # 第一欄當 ID (常見 Kaggle 格式)
            id_col = predict_df.columns[0]
            sub[id_col] = predict_df[id_col].values[:len(predictions)]
            sub[target_name] = predictions[:len(sub)]
            out = _io.StringIO()
            sub.to_csv(out, index=False)
            return out.getvalue().encode("utf-8-sig")

        def worker():
            nonlocal error_msg
            try:
                for job in jobs:
                    label = job["label"]
                    on_prog = lambda ev, _label=label: q.put(prefix(ev, _label))
                    if cancel_token.is_set():
                        break
                    if "train_csv_path" in job:
                        # 路徑模式 (預處理大矩陣):CSV 已串流寫好,只傳路徑,不在記憶體扛 bytes
                        r = run_daniel_pipeline(
                            train_csv_path=job["train_csv_path"],
                            test_csv_path=job["test_csv_path"],
                            train_file_name=job["train_file_name"],
                            options=job["options"],
                            on_progress=on_prog,
                            cancel_token=cancel_token,
                        )
                    elif "train_csv_bytes" in job:
                        r = run_daniel_pipeline(
                            train_csv_bytes=job["train_csv_bytes"],
                            test_csv_bytes=job["test_csv_bytes"],
                            train_file_name=job["train_file_name"],
                            options=job["options"],
                            on_progress=on_prog,
                            cancel_token=cancel_token,
                        )
                    else:
                        r = run_daniel_pipeline(
                            csv_bytes=job["csv_bytes"],
                            file_name=job["file_name"],
                            options=job["options"],
                            on_progress=on_prog,
                            cancel_token=cancel_token,
                        )
                    r["dataSource"] = job["source"]
                    r["dataSourceLabel"] = label
                    if job.get("preprocessorId"):
                        r["preprocessorId"] = job["preprocessorId"]

                    # ── SHAP 支援：最佳 tabular 模型存 storage ─────────────────
                    if r.get("ok") and r.get("modelBytes"):
                        try:
                            import pickle as _pickle
                            import base64 as _b64
                            import pandas as _pd
                            _raw = _b64.b64decode(r["modelBytes"].encode("ascii"))
                            _m   = _pickle.loads(_raw)
                            _estimator  = _m["estimator"]
                            _X_test_np  = _m.get("X_test")          # np.ndarray
                            _n_feat     = _X_test_np.shape[1] if _X_test_np is not None else 0
                            _col_names  = [f"f{i}" for i in range(_n_feat)]
                            _X_test_df  = (
                                _pd.DataFrame(_X_test_np, columns=_col_names)
                                if _X_test_np is not None else _pd.DataFrame()
                            )
                            _pbundle = {
                                "type": "pipeline_best",
                                "name": _m.get("model_tag", "pipeline_best"),
                                "featureNames": _col_names,
                                "featureSet": _m.get("feature_set"),
                                "dataSource": job["source"],
                            }
                            _mid = storage.save_model(
                                bundle=_pbundle, estimator=_estimator, scaler=None,
                                X_test_df=_X_test_df, user=user, db=db,
                                preprocessor_id=job.get("preprocessorId"),
                            )
                            r["bestModelId"] = _mid
                            print(f"[Pipeline] 最佳模型存入 storage → modelId={_mid}", flush=True)
                        except Exception as _me:
                            import traceback as _tb
                            print(f"[Pipeline] save_model 失敗（不影響分數）: {_me}", flush=True)
                            _tb.print_exc()
                    r.pop("modelBytes", None)   # 大 blob 不需傳進 SSE

                    # Option B:有 predict_input 且 pipeline 成功 → 寫 submission artifact
                    predict_input = job.get("predict_input_bytes")
                    if (predict_input and r.get("ok") and r.get("predictions") and pipeline_run_id):
                        try:
                            submission_bytes = _build_submission_csv(
                                predict_input, r["predictions"], r.get("target") or "prediction",
                            )
                            kind = f"submission_{job['source']}"
                            storage.save_prediction_artifact(
                                training_run_id=pipeline_run_id, kind=kind,
                                file_name=f"submission_{job['source']}.csv",
                                content_bytes=submission_bytes, user=user, db=db,
                            )
                            r["submissionAvailable"] = True
                            r["submissionKind"] = kind
                        except Exception as e:
                            q.put({"type": "log", "level": "warning",
                                   "msg": f"[{label}] submission CSV 產生失敗: {e}"})

                    all_results.append(r)
            except Exception as e:
                error_msg = str(e)
            finally:
                # 清掉路徑模式 (預處理) 產生的暫存 CSV — run_pipeline 不刪 caller 的檔
                import os as _os
                for _job in jobs:
                    for _p in _job.get("_temp_paths") or []:
                        try: _os.unlink(_p)
                        except Exception: pass
                q.put(None)

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()

        try:
            while True:
                ev = q.get()
                if ev is None:
                    break
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                # 旁路把 progress / log 寫進 DB,給跨分頁 polling 看 (throttled 內部 2s 一次)
                try:
                    if ev.get("type") == "progress":
                        storage.update_training_progress(
                            pipeline_run_id, user, db,
                            pct=ev.get("pct"), step=ev.get("step"),
                        )
                    elif ev.get("type") == "log":
                        storage.update_training_progress(
                            pipeline_run_id, user, db,
                            log_line={"msg": ev.get("msg", ""), "level": ev.get("level", "info")},
                        )
                except Exception: pass
        except (GeneratorExit, asyncio.CancelledError):
            # 前端 SSE 斷線 (取消訓練 / 關分頁) → 告知 worker 立刻收尾
            cancel_token.set()
            try: storage.finish_training_run(pipeline_run_id, user, db,
                                             status="failed", error_msg="使用者取消",
                                             elapsed_sec=round(time.time() - t_start, 2))
            except Exception: pass
            raise

        elapsed = round(time.time() - t_start, 2)
        if error_msg:
            storage.finish_training_run(pipeline_run_id, user, db,
                                        status="failed", error_msg=error_msg,
                                        elapsed_sec=elapsed)
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg}, ensure_ascii=False)}\n\n"
        else:
            # 整理 results_summary (per-source 分數 + bestScore)
            summary = {
                "perSource": [{
                    "source": r.get("dataSource"),
                    "label": r.get("dataSourceLabel"),
                    "bestScore": r.get("bestScore"),
                    "scoreBlend": r.get("scoreBlend"),
                    "scoreStack": r.get("scoreStack"),
                    "accuracy": r.get("accuracy"),
                    "f1": r.get("f1"),
                    "metric": r.get("metric"),
                    "elapsedSec": r.get("elapsedSec"),
                    "submissionAvailable": r.get("submissionAvailable", False),
                    "submissionKind": r.get("submissionKind"),
                } for r in all_results],
                "datasetName": dataset_name_for_run,
            }
            has_predictions = any(r.get("submissionAvailable") for r in all_results)
            storage.finish_training_run(
                pipeline_run_id, user, db, status="completed",
                results_summary=summary, has_predictions=has_predictions,
                elapsed_sec=elapsed,
            )
            yield f"data: {json.dumps({'type': 'done', 'results': all_results, 'runId': pipeline_run_id}, ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================================
# TRAINING RUNS — 給 dashboard 最近實驗篩選用
# ============================================================
@app.get("/api/training-runs")
def list_training_runs_endpoint(
    datasetId: str | None = None,
    target: str | None = None,
    engine: str | None = None,
    limit: int = 50,
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
) -> dict[str, Any]:
    """列出 user 的訓練紀錄,可依 datasetId / target / engine 篩選,新→舊排序。
    Guest 走 in-memory 路徑(沒 DB)→ 直接回空 list,前端就 fallback 用 localStorage 的歷史。"""
    return {"runs": storage.list_training_runs(
        user, db, dataset_id=datasetId, target=target, engine=engine, limit=limit,
    )}


@app.get("/api/training-runs/{run_id}")
def get_training_run_endpoint(
    run_id: str,
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
) -> dict[str, Any]:
    return storage.get_training_run(run_id, user, db)


@app.get("/api/training-runs/{run_id}/progress")
def get_training_progress_endpoint(
    run_id: str,
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
) -> dict[str, Any]:
    """輕量 polling endpoint — 給跨分頁/裝置看訓練即時進度。
    回 progressPct / currentStep / latestLog / lastSeenAt / status。
    若 run 不存在或非該 user → 404。"""
    data = storage.get_training_progress(run_id, user, db)
    if data is None:
        raise HTTPException(status_code=404, detail="training run 不存在")
    return data


# ============================================================
# MODELS — 給登入時一次性還原前端 leaderboard / insights / what-if
# ============================================================
@app.get("/api/models")
def list_models_endpoint(
    trainingRunId: str | None = None,
    datasetId: str | None = None,
    limit: int = 200,
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
) -> dict[str, Any]:
    """列出 user 的所有 sklearn 模型 — 輕量版,bundle 內含 metrics/featureImportance/testTrue/Pred,
    不含 estimator pickle (那要用 SHAP/batch predict 時前端送 modelId 後端再載)。"""
    return {"models": storage.list_models(
        user, db, training_run_id=trainingRunId, dataset_id=datasetId, limit=limit,
    )}


@app.delete("/api/models/{model_id}")
def delete_model_endpoint(
    model_id: str,
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
) -> dict[str, Any]:
    """刪除指定模型（需為模型擁有者）。"""
    return storage.delete_model(model_id, user, db)


# 下載 pipeline 的 submission CSV
@app.get("/api/train/pipeline/runs/{run_id}/submission")
def pipeline_run_submission_endpoint(
    run_id: str,
    kind: str = "submission_raw",  # submission_raw / submission_preprocessed / submission_upload
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
):
    """下載某次 pipeline 訓練的預測結果 CSV。"""
    artifact = storage.get_prediction_artifact(run_id, kind, user, db)
    if not artifact:
        raise HTTPException(status_code=404, detail=f"找不到 {kind} 預測結果")
    return Response(
        content=artifact["contentBytes"],
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{artifact["fileName"]}"'},
    )


def bundle_safe_name(s: str) -> str:
    """把字串清成檔名安全:只留英數 + 底線。"""
    import re
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(s))[:40] or "preprocessed"


# ============================================================
# 2b. TRAIN (SSE) — 即時推送進度與 log
# ============================================================
@app.post("/api/train/stream")
def train_stream_endpoint(
    req: TrainRequest,
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
):
    t_start = time.time()
    run_id = storage.create_training_run(
        dataset_id=req.datasetId, dataset_name="(loading)",
        engine="sklearn", target=req.target,
        task_type=req.options.get("taskType", "auto") if req.options else "auto",
        sources=req.sources, options=req.options or {},
        user=user, db=db,
    )

    def event_stream():
        q: queue.Queue = queue.Queue()
        result_box: dict[str, Any] = {"results": None, "dataset_name": None, "error": None}
        cancel_token = threading.Event()   # 前端 SSE 斷線時 set,worker 在演算法之間 cooperative check

        def worker():
            try:
                results, ds_name = _run_sources(
                    req, user, db, on_progress=lambda ev: q.put(ev),
                    cancel_token=cancel_token,
                )
                result_box["results"] = results
                result_box["dataset_name"] = ds_name
            except HTTPException as he:
                result_box["error"] = str(he.detail)
            except Exception as e:
                result_box["error"] = str(e)
            finally:
                q.put(None)

        threading.Thread(target=worker, daemon=True).start()

        try:
            while True:
                ev = q.get()
                if ev is None: break
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                # 旁路把 progress / log 寫進 DB
                try:
                    if ev.get("type") == "progress":
                        storage.update_training_progress(
                            run_id, user, db,
                            pct=ev.get("pct"), step=ev.get("step"),
                        )
                    elif ev.get("type") == "log":
                        storage.update_training_progress(
                            run_id, user, db,
                            log_line={"msg": ev.get("msg", ""), "level": ev.get("level", "info")},
                        )
                except Exception: pass
        except (GeneratorExit, asyncio.CancelledError):
            # 前端 SSE 斷線 (取消按鈕 / 關分頁) → 告訴 worker 在下一個演算法前停下
            cancel_token.set()
            try: storage.finish_training_run(run_id, user, db, status="failed",
                                             error_msg="使用者取消",
                                             elapsed_sec=round(time.time() - t_start, 2))
            except Exception: pass
            raise

        if result_box["error"]:
            storage.finish_training_run(run_id, user, db, status="failed",
                                        error_msg=result_box["error"],
                                        elapsed_sec=round(time.time() - t_start, 2))
            yield f"data: {json.dumps({'type': 'error', 'message': result_box['error']}, ensure_ascii=False)}\n\n"
        elif not result_box["results"]:
            err = "訓練結束但沒收到任何結果 (內部錯誤,請重試)"
            storage.finish_training_run(run_id, user, db, status="failed", error_msg=err,
                                        elapsed_sec=round(time.time() - t_start, 2))
            yield f"data: {json.dumps({'type': 'error', 'message': err}, ensure_ascii=False)}\n\n"
        else:
            # 存模型階段:用 commit=False 把 30 筆 INSERT 壓成 1 次 commit (省 N-1 次 round-trip 到 Supabase),
            # 同時每 add 一個就 yield 一次,讓前端看到進度 + 保活 SSE 連線。
            bundles = []
            results = result_box["results"]
            total = len(results)
            for i, (bundle, estimator, scaler, X_test_df) in enumerate(results):
                pct = 90 + int(6 * (i + 1) / max(total, 1))  # 90~96% 用於 add 階段
                model_label = bundle.get("name") or bundle.get("type") or "model"
                yield f"data: {json.dumps({'type': 'log', 'msg': f'準備儲存 {i+1}/{total} ({model_label})...', 'level': 'muted'}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'progress', 'pct': pct, 'step': f'準備儲存 {i+1}/{total}'}, ensure_ascii=False)}\n\n"
                try:
                    model_id = storage.save_model(
                        bundle=bundle, estimator=estimator, scaler=scaler, X_test_df=X_test_df,
                        user=user, db=db,
                        preprocessor_id=bundle.get("preprocessorId"),
                        dataset_id=req.datasetId,
                        training_run_id=run_id,
                        hyperparameters=bundle.get("hyperparameters", {}),
                        commit=False,  # ← 關鍵:不要每個都 commit,跑完一起 commit
                    )
                    bundle["id"] = model_id
                except Exception as e:
                    import traceback as _tb
                    print(f"[event_stream] save_model 第 {i+1}/{total} 筆 add 失敗 (algo={bundle.get('type')}): {type(e).__name__}: {e}", flush=True)
                    _tb.print_exc()
                    bundle["id"] = None
                    bundle["_storeError"] = f"{type(e).__name__}: {e}"
                    yield f"data: {json.dumps({'type': 'log', 'msg': f'✗ 準備第 {i+1} 筆失敗: {e}', 'level': 'warning'}, ensure_ascii=False)}\n\n"
                bundles.append(bundle)

            # 一次性 commit 所有 add — 比一筆一筆 commit 快數十倍
            yield f"data: {json.dumps({'type': 'log', 'msg': f'⬆️ 一次上傳 {total} 個模型到 DB (Supabase)...', 'level': 'info'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'progress', 'pct': 97, 'step': '上傳到 DB'}, ensure_ascii=False)}\n\n"
            commit_t = time.time()
            try:
                db.commit()
                yield f"data: {json.dumps({'type': 'log', 'msg': f'✓ 全部上傳完成 ({round(time.time() - commit_t, 1)}s)', 'level': 'success'}, ensure_ascii=False)}\n\n"
            except Exception as e:
                import traceback as _tb
                print(f"[event_stream] bulk commit 失敗: {type(e).__name__}: {e}", flush=True)
                _tb.print_exc()
                try: db.rollback()
                except Exception: pass
                # 失敗的話前端 bundle 的 id 都失效 (沒寫進 DB)
                for b in bundles:
                    b["id"] = None
                    b.setdefault("_storeError", f"bulk commit 失敗: {type(e).__name__}: {e}")
                yield f"data: {json.dumps({'type': 'log', 'msg': f'✗ DB commit 失敗: {e} (前端仍可顯示,但不能跑 SHAP)', 'level': 'warning'}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type': 'log', 'msg': '更新訓練紀錄...', 'level': 'muted'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'progress', 'pct': 99, 'step': '更新訓練紀錄'}, ensure_ascii=False)}\n\n"
            try:
                storage.finish_training_run(
                    run_id, user, db, status="completed",
                    results_summary={
                        "topModels": [{"id": b.get("id"), "name": b.get("name"),
                                       "score": b.get("metrics", {}).get("testScore", 0.0)}
                                      for b in bundles[:5]],
                        "datasetName": result_box["dataset_name"],
                    },
                    model_ids=[b.get("id") for b in bundles if b.get("id")],
                    elapsed_sec=round(time.time() - t_start, 2),
                )
            except Exception as e:
                print(f"[event_stream] finish_training_run 失敗 (繼續送 done): {type(e).__name__}: {e}", flush=True)
                try: db.rollback()
                except Exception: pass

            yield f"data: {json.dumps({'type': 'progress', 'pct': 100, 'step': '完成'}, ensure_ascii=False)}\n\n"

            # ★ 改:分批送 model 事件,每個 < 20 KB,避免「30 個 bundle 黏成 500KB 大 chunk」
            #   被 SSE / browser buffer 切斷導致前端 JSON.parse 失敗 → 變成 models=[]。
            #   每個 model 一個事件,最後送小小的 'done' sentinel 告訴前端「結束了,共 N 個」。
            for b in bundles:
                try:
                    # 先 sanitize NaN/Inf → null,不然 JS JSON.parse 會炸 (NaN 不是合法 JSON)
                    safe_b = storage.sanitize_for_json(b)
                    model_payload = json.dumps({'type': 'model', 'bundle': safe_b},
                                               ensure_ascii=False, default=str)
                except Exception as e:
                    print(f"[event_stream] 模型 {b.get('id')} JSON 化失敗,改送 minimal: {e}", flush=True)
                    mini = {
                        "id": b.get("id"), "name": b.get("name"),
                        "type": b.get("type"),
                        "dataSource": b.get("dataSource"),
                        "metrics": storage.sanitize_for_json(b.get("metrics") or {}),
                        "trainTime": b.get("trainTime", 0),
                        "_minimal": True,
                    }
                    model_payload = json.dumps({'type': 'model', 'bundle': mini},
                                               ensure_ascii=False, default=str)
                yield f"data: {model_payload}\n\n"

            # 最後的 sentinel — 小小一個,前端用來知道「全部收完了」(帶 runId 給新 UI 跳結果)
            yield f"data: {json.dumps({'type': 'done', 'count': len(bundles), 'runId': run_id}, ensure_ascii=False)}\n\n"

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
def visualize_endpoint(
    req: VisualizeRequest,
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
) -> dict[str, Any]:
    entry = storage.get_model(req.modelId, user, db)
    return visualize.run(entry["bundle"], req.chartType, req.options)


# ============================================================
# 3b. VISUALIZE (SHAP) — 隊友 AutoMLVisualizer,回傳 Plotly figure JSON
# ============================================================
class ShapRequest(BaseModel):
    modelId: str
    sampleIndex: int = 0
    targetFeature: str | None = None
    maxSamples: int = 50  # SHAP 算太多會很慢 — 對 Tree/Linear 模型再大都秒級,但對 Permutation
                          # fallback (SVC/Voting/Stacking/KNN) 一個樣本就要 3 秒,200 個會跑 10 分鐘。
                          # 50 在大部分情況下圖夠穩,使用者要更穩可在系統設定調回 100~200。


@app.post("/api/visualize/shap")
def visualize_shap_endpoint(
    req: ShapRequest,
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
) -> dict[str, Any]:
    if AutoMLVisualizer is None:
        raise HTTPException(
            status_code=500,
            detail="AutoMLVisualizer 未啟用 — 請安裝 shap + plotly: pip install shap plotly",
        )
    entry = storage.get_model(req.modelId, user, db)

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
def predict_endpoint(
    req: PredictRequest,
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
) -> dict[str, Any]:
    entry = storage.get_model(req.modelId, user, db)

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
    # Pipeline 最佳模型儲存時 scaler=None (模型內部已含前處理)
    x_norm = scaler.transform(x) if scaler is not None else x
    pred = estimator.predict(x_norm)[0]
    # numpy types 不是 JSON serializable
    if hasattr(pred, "item"):
        pred = pred.item()

    # 分類任務額外回傳機率 (讓 What-If UI 可以顯示信心度)
    proba: list[float] | None = None
    classes: list | None = None
    if hasattr(estimator, "predict_proba"):
        try:
            proba = estimator.predict_proba(x_norm)[0].tolist()
            if hasattr(estimator, "classes_"):
                classes = [c.item() if hasattr(c, "item") else c for c in estimator.classes_]
        except Exception:
            pass

    return {"prediction": pred, "proba": proba, "classes": classes}


# ============================================================
# 4c. MODEL INFO — 回傳特徵名稱 + 統計值,供前端 What-If 初始化用
# ============================================================
@app.get("/api/model/{model_id}/info")
def model_info_endpoint(
    model_id: str,
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
) -> dict[str, Any]:
    # Returns feature names, means, stds, min, max for What-If slider initialization.
    # Works for Pipeline models (scaler=None) by computing stats from X_test.
    entry = storage.get_model(model_id, user, db)
    feature_names: list[str] = entry["featureNames"]
    scaler = entry["scaler"]
    n = len(feature_names)
    if n == 0:
        return {"featureNames": [], "featureMeans": [], "featureStds": [],
                "featureMin": [], "featureMax": [], "taskType": "regression"}
    if scaler is not None and hasattr(scaler, "mean_"):
        means = scaler.mean_.tolist()
        stds  = scaler.scale_.tolist()
    else:
        x_test = entry.get("X_test_df")
        if x_test is not None and len(x_test) > 0:
            import pandas as _pd
            if not isinstance(x_test, _pd.DataFrame):
                x_test = _pd.DataFrame(x_test)
            means = x_test.mean().tolist()
            stds  = x_test.std(ddof=1).fillna(1.0).tolist()
        else:
            means = [0.0] * n
            stds  = [1.0] * n
    feat_min = [m - 3 * s for m, s in zip(means, stds)]
    feat_max = [m + 3 * s for m, s in zip(means, stds)]
    bundle = entry.get("bundle", {})
    task_type = bundle.get("taskType", "regression")
    return {
        "featureNames": feature_names,
        "featureMeans": means,
        "featureStds":  stds,
        "featureMin":   feat_min,
        "featureMax":   feat_max,
        "taskType":     task_type,
        "targetName":   bundle.get("targetName", ""),
        "modelName":    bundle.get("name", model_id),
    }

# ============================================================
# 4b. PREDICT (BATCH)
# ============================================================
@app.post("/api/predict/batch")
async def predict_batch_endpoint(
    modelId: str = Form(...),
    file: UploadFile = File(...),
    sampleFile: Optional[UploadFile] = File(None),
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
):
    import pandas as pd
    entry = storage.get_model(modelId, user, db)
    estimator     = entry["estimator"]
    scaler        = entry["scaler"]
    feature_names = entry["featureNames"]
    preprocessor_id = entry.get("preprocessorId")

    # --- parse uploaded CSV ---
    try:
        raw = await file.read()
        feat_df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"CSV \u89e3\u6790\u5931\u6557: {e}")

    # --- apply preprocessor if any ---
    if preprocessor_id:
        try:
            pp_entry = storage.get_preprocessor(preprocessor_id)
            preprocessor = pp_entry["pipeline"]
            pp_target    = pp_entry.get("target", "")
            cols = [c for c in feat_df.columns if c != pp_target]
            feat_df[cols] = preprocessor.transform(feat_df[cols])
        except Exception as e:
            raise HTTPException(400, f"\u5957\u7528\u9810\u8655\u7406\u5931\u6557 \u2014 test.csv \u7684\u6b04\u4f4d\u9700\u8207\u8a13\u7df4\u8cc7\u6599\u7684\u539f\u59cb\u6b04\u4f4d\u4e00\u81f4: {e}")

    # --- select & order features ---
    missing = [f for f in feature_names if f not in feat_df.columns]
    if missing:
        raise HTTPException(400,
            f"CSV \u7f3a\u5c11 {len(missing)} \u500b\u6a21\u578b\u9700\u8981\u7684\u7279\u5fb5\u6b04\u4f4d: {missing}")

    X_t = feat_df[feature_names].values.astype(float)
    X   = scaler.transform(X_t) if scaler is not None else X_t

    try:
        preds = estimator.predict(X)
    except Exception as e:
        raise HTTPException(500, f"\u6279\u6b21\u9810\u6e2c\u5931\u6557: {e}")

    # --- build output ---
    if sampleFile is not None:
        try:
            sample_raw = await sampleFile.read()
            sub_df = pd.read_csv(io.BytesIO(sample_raw))
        except Exception as e:
            raise HTTPException(400, f"\u7bc4\u672c submission \u89e3\u6790\u5931\u6557: {e}")
        if len(sub_df.columns) < 2:
            raise HTTPException(400, "\u7bc4\u672c submission \u81f3\u5c11\u9700\u8981 2 \u6b04 (ID \u6b04 + \u9810\u6e2c\u6b04)")
        id_col   = sub_df.columns[0]
        pred_col = sub_df.columns[1]
        if id_col not in feat_df.columns:
            raise HTTPException(400,
                f"test.csv \u7f3a\u5c11\u7bc4\u672c\u8981\u6c42\u7684 ID \u6b04\u4f4d\u300c{id_col}\u300d")
        out_df = pd.DataFrame({id_col: feat_df[id_col], pred_col: preds})
        out_name = "submission.csv"
    else:
        out_df   = feat_df[feature_names].copy()
        out_df["prediction"] = preds
        out_name = file.filename.replace(".csv", "_predicted.csv") if file.filename else "_predicted.csv"

    csv_bytes = out_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )
