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
import os
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

# 前端入口 (for Claude in Chrome testing)
import pathlib as _pl
from fastapi.responses import HTMLResponse as _HTMLResponse
@app.get("/ui", response_class=_HTMLResponse, include_in_schema=False)
def _serve_ui():
    return (_pl.Path(__file__).parent.parent / "index.html").read_text(encoding="utf-8")



def _infer_task_type(y_series) -> str:
    """跟 _runner_entry._detect_task 同一條 heuristic — 用在 create_training_run
    前先把 task_type 推對,進行中也顯示正確的 metric/欄位。

    object / bool → classification;
    數值且 nunique <= 50 且佔比 < 30% → classification;
    其它 → regression。

    抓不到 / series 為 None → fallback "classification" (跟舊行為一致)。
    """
    try:
        import pandas as _pd
        if y_series is None:
            return "classification"
        s = y_series if isinstance(y_series, _pd.Series) else _pd.Series(y_series)
        if s.dtype == object or s.dtype == bool:
            return "classification"
        n = len(s)
        if n == 0:
            return "classification"
        n_unique = s.nunique(dropna=True)
        return "classification" if (n_unique <= 50 and n_unique / n < 0.30) else "regression"
    except Exception:
        return "classification"


def _safe_json_payload(obj) -> str:
    """SSE event 用的 NaN-safe JSON serializer。
    Python 內建 json.dumps 預設 allow_nan=True,會輸出 `NaN` literal — 那是無效 JSON,
    前端 JSON.parse() 會靜默 try/catch → 事件被吃掉 → 看起來像「sent nothing」。
    這裡先 allow_nan=False 試,炸了就遞迴把 NaN/Inf 換成 None 再 dump。
    """
    try:
        return json.dumps(obj, ensure_ascii=False, default=str, allow_nan=False)
    except ValueError:
        import math as _math
        def _scrub(v):
            if isinstance(v, float) and (_math.isnan(v) or _math.isinf(v)):
                return None
            if isinstance(v, list):
                return [_scrub(x) for x in v]
            if isinstance(v, dict):
                return {k: _scrub(x) for k, x in v.items()}
            return v
        return json.dumps(_scrub(obj), ensure_ascii=False, default=str)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    # async def 而非 def:health 不能被「同步重計算 endpoint (batch_prep / SHAP)」卡在 threadpool 後面。
    # 改成 async 走 event loop,即使所有 worker thread 都被占用,health 仍然秒回。
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


@app.post("/api/preprocess/transform")
async def preprocess_transform_endpoint(
    datasetId: str = Form(...),
    target: str = Form(...),
    testSize: float = Form(0.2),
    # 新版 daniel 已內部接管,留著當 schema 紀錄用,不影響 transform 行為
    useMice: bool = Form(False),
    useMiSelection: bool = Form(False),
    miThreshold: float = Form(0.01),
    # 時序模式 — 勾起後預處理改 chronological split (取最後 X% 當 holdout,no shuffle),
    # 避免隨機切讓未來資料混進 train → 時序模型 leak。預設 False 維持原 sklearn 隨機切。
    timeSeries: bool = Form(False),
    # 對抗驗證測試集 — 可選 Kaggle 風 test.csv。傳了會啟動 daniel 的
    # adversarial validation,偵測 train/test 分佈漂移的「間諜特徵」並剔除
    adversarialTestFile: UploadFile | None = File(None),
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
) -> dict[str, Any]:
    """跑完整的 preprocess_for_training,把 fitted preprocessor 存起來。
    multipart 介面 — 支援可選的對抗驗證測試集上傳。
    """
    bundle = storage.get_dataset(datasetId, user, db, include_df=True)
    df = bundle["df"]
    if target not in df.columns:
        raise HTTPException(status_code=400, detail=f"target '{target}' 不在欄位中")

    # 對抗驗證測試集 — 若有上傳就讀進 DataFrame 餵給 preprocess_for_training
    adv_test_df = None
    if adversarialTestFile is not None:
        try:
            import pandas as _pd
            adv_bytes = await adversarialTestFile.read()
            adv_test_df = _pd.read_csv(io.BytesIO(adv_bytes))
            print(f"[adversarial] 對抗驗證測試集載入 {adv_test_df.shape}", flush=True)
        except Exception as e:
            raise HTTPException(status_code=400,
                detail=f"對抗驗證測試集 CSV 解析失敗: {e}")

    try:
        # 注意:daniel 最新版 preprocess_for_training 拿掉了 use_mice / use_mi_selection /
        # mi_threshold 三個 flag — MICE 跟 MI selection 改成內部自動依資料特性決定。
        # 若有 adv_test_df,enable_adv_val=True 才會跑;沒給就跳過。
        X_train, X_test, y_train, y_test, fitted = preprocess_for_training(
            df, target, test_size=testSize,
            test_data_source=adv_test_df,
            enable_adv_val=(adv_test_df is not None),
            is_time_series=timeSeries,
        )
    except Exception as e:
        # 把完整 traceback 印到 stderr,前端只看 e.__class__ + message 沒辦法 debug
        # 哪一個 sklearn step 拋的 (StandardScaler / PowerTransformer / MIFeatureSelector...)
        import traceback as _tb
        _tb.print_exc()
        raise HTTPException(status_code=500, detail=f"transform 失敗: {e}")

    # 新版 preprocess_for_training 回傳雙軌 dict；
    # 取 tree 軌作為代表（feature_names / preview / shape 都用 tree 軌）
    X_train_tree = X_train["tree"] if isinstance(X_train, dict) else X_train
    X_test_tree  = X_test["tree"]  if isinstance(X_test,  dict) else X_test
    fitted_tree  = fitted["tree"]  if isinstance(fitted,  dict) else fitted

    feature_names = list(X_train_tree.columns)
    preprocessor_id = storage.save_preprocessor(
        preprocessor=fitted_tree, target=target, dataset_id=datasetId,
        feature_names=feature_names,
        X_train=X_train_tree, X_test=X_test_tree, y_train=y_train, y_test=y_test,
        user=user, db=db, test_size=testSize,
        use_mice=useMice,
        use_mi_selection=useMiSelection,
        mi_threshold=miThreshold,
    )

    # Router 分類預覽 (用於前端顯示哪些欄位被分到哪一桶)
    scan_df = df.drop(columns=[target])
    router = AutoRouter(categorical_threshold=50, text_length_threshold=20)
    feature_groups = router.fit_predict(scan_df)

    audit = run_data_audit(df, target)

    # 預覽前 10 列,float 轉乾淨 JSON
    preview_rows = X_train_tree.head(10).fillna(0).values.tolist()
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
        "trainSize": int(len(X_train_tree)),
        "testSize": int(len(X_test_tree)),
        "originalFeatureCount": int(df.shape[1] - 1),
        "transformedFeatureCount": int(X_train_tree.shape[1]),
        # v3 透明度:實際自動觸發的進階特徵工程 + MI 篩選結果
        "appliedFeatureSteps": getattr(fitted_tree, "applied_feature_steps_", []),
        "miSelection": getattr(fitted_tree, "mi_selection_", None),
        # 對抗驗證結果 — 只有 user 傳了 test.csv 才會非 None
        # 結構:{auc_mean, auc_std, verdict, message, features_to_drop, feature_importances, ...}
        "adversarialValidation": getattr(fitted_tree, "adversarial_validation_", None),
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
    # 為了在 create_training_run 之前推 task_type:三個 source 分支都會在這存一份 target 欄
    # 的 sample (pandas Series),最後丟給 _infer_task_type 就拿到 classification/regression。
    target_col_sample = None

    if direct_upload:
        raw = await file.read()
        dataset_name_for_run = file.filename or "uploaded.csv"
        # 抽 target 欄推 task_type — 失敗 (CSV 壞 / target 不存在) 就吃掉,讓 fallback 走預設值
        try:
            import pandas as _pd_infer
            _df_infer = _pd_infer.read_csv(_io.BytesIO(raw))
            if target and target in _df_infer.columns:
                target_col_sample = _df_infer[target]
            del _df_infer
        except Exception:
            pass
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
            # raw source 直接從 df 抽 target → 推 task_type
            if target_col_sample is None and target and target in df.columns:
                target_col_sample = df[target]
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

            # 【y_test=None 的兩個來源】
            #   1. 對抗驗證 path:user 上傳 test.csv (無 target) 觸發
            #   2. test_size=0 path:user 滑桿拉到 0% 不留 holdout
            # 兩種情況都沒「有標籤 holdout」可給 daniel,改走 daniel single-CSV 模式 (mode 1) —
            # daniel 內部會自己對 train_csv 跑 80/20。下面組 job 時看到 y_test=None → 不寫 test CSV。
            # preprocessed source 沒有原 df,但 y_train 就是 target 欄;直接拿來推 task_type
            if target_col_sample is None and y_train is not None:
                import pandas as _pd_infer2
                target_col_sample = _pd_infer2.Series(y_train)

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
            test_csv_path = None    # None = 沒有有標籤 holdout,走 daniel mode 1 自切
            if predict_csv_bytes:
                import pandas as _pd
                try:
                    predict_df_raw = _pd.read_csv(_io.BytesIO(predict_csv_bytes))
                    # 預處理器 fit 時看的是不含 target 的 X,所以丟掉 target (若有)
                    predict_feat_raw = predict_df_raw.drop(columns=[pp_target], errors="ignore")
                    transformed = pp_obj.transform(predict_feat_raw)
                    if hasattr(transformed, "toarray"):
                        transformed = transformed.toarray()
                    # train/test 同一個 preprocessor → 欄位一致,用 X_train 的欄名(避開 X_test 可能空的 case)
                    _cols_ref = list(X_train_df.columns) if X_train_df is not None else list(X_test_df.columns)
                    # X_train_df 此時包含 target 欄(line 773 加的),要剔除
                    _cols_ref = [c for c in _cols_ref if c != pp_target]
                    transformed_df = _pd.DataFrame(transformed, columns=_cols_ref)
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"套用預處理到 predict CSV 失敗 — 欄位需與原始訓練資料一致: {e}",
                    )
                test_csv_path = _df_to_temp_csv(transformed_df, f"daniel_{_safe}_test_")
                del transformed_df
                predict_input_for_job = predict_csv_bytes
            elif y_test is not None and X_test_df is not None and len(X_test_df) > 0:
                # 有 holdout 才寫 test CSV → daniel 走 mode 2 (pre-split)
                X_test_df[pp_target] = list(y_test)
                test_csv_path = _df_to_temp_csv(X_test_df, f"daniel_{_safe}_test_")
            else:
                # y_test=None 或 X_test 空 → 不寫 test CSV → 下面 worker 看到 test_csv_path is None
                # 就改用 csv_path 餵 daniel mode 1,daniel 內部自己 80/20 切
                print(f"[train_pipeline] 沒有有標籤 holdout,改走 daniel single-CSV mode (mode 1)", flush=True)
            if X_test_df is not None:
                del X_test_df
            pp_entry["X_test"] = None

            pp_options = dict(base_options)
            pp_options["target"] = pp_target

            # 暫存檔清單動態組 — daniel mode 1 沒 test_csv_path
            _temp_paths = [train_csv_path]
            if test_csv_path:
                _temp_paths.append(test_csv_path)

            jobs.append({
                "label": "預處理", "source": "preprocessed",
                "train_csv_path": train_csv_path,
                "test_csv_path": test_csv_path,   # None → worker 走 daniel mode 1 (用 csv_path)
                "train_file_name": f"{_safe}.csv",
                "options": pp_options,
                "preprocessorId": preprocessorId,
                "predict_input_bytes": predict_input_for_job,
                # run_pipeline 不會刪 caller 給的路徑 → 由 worker 的 finally 收尾刪除
                "_temp_paths": _temp_paths,
            })

    if not jobs:
        raise HTTPException(status_code=400, detail="沒有產生任何訓練任務")

    multi = len(jobs) > 1

    # 建 TrainingRun (running 狀態,完成後再 update)
    # task_type 從 target 欄推;subprocess 跑完 finish_training_run 還會用真正的 result 再覆寫一次,
    # 但訓練 *進行中* hydration 用 r.taskType 推 metric label 就不會錯 (回歸顯示 R² 不是 Accuracy)
    inferred_task_type = _infer_task_type(target_col_sample)
    pipeline_run_id = storage.create_training_run(
        dataset_id=datasetId or "uploaded",
        dataset_name=dataset_name_for_run,
        engine="pipeline",
        target=target or "(auto)",
        task_type=inferred_task_type,
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
                        if job.get("test_csv_path"):
                            # daniel mode 2:有 holdout 的 test CSV
                            r = run_daniel_pipeline(
                                train_csv_path=job["train_csv_path"],
                                test_csv_path=job["test_csv_path"],
                                train_file_name=job["train_file_name"],
                                options=job["options"],
                                on_progress=on_prog,
                                cancel_token=cancel_token,
                            )
                        else:
                            # daniel mode 1:單 CSV,daniel 內部自切 80/20
                            # (preprocess test_size=0 或對抗驗證 path 走到這)
                            r = run_daniel_pipeline(
                                csv_path=job["train_csv_path"],
                                file_name=job["train_file_name"],
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

                    # ── Ensemble bundle 持久化 (給 batch predict / Insights / re-login 用) ─────
                    print(f"[Pipeline] [{label}] subprocess 回傳 ok={r.get('ok')} "
                          f"ensembleBundlePath={r.get('ensembleBundlePath')!r}", flush=True)
                    if r.get("ok") and r.get("ensembleBundlePath"):
                        try:
                            import pandas as _pd
                            bundle_path = r["ensembleBundlePath"]
                            if os.path.exists(bundle_path):
                                with open(bundle_path, "rb") as _bf:
                                    bundle_bytes = _bf.read()
                                size_mb = len(bundle_bytes) / 1024 / 1024
                                print(f"[Pipeline] ensemble bundle = {size_mb:.1f}MB → 寫入 DB", flush=True)
                                # Insights / Leaderboard 顯示用的 bundle (不含 estimator)
                                _r_task = r.get("taskType", "classification")
                                _is_reg = (_r_task == "regression")
                                _ens_bundle = {
                                    "type": "daniel_pipeline_ensemble",
                                    "name": f"[{label}] Pipeline Ensemble",
                                    "dataSource": job["source"],
                                    "dataSourceLabel": label,
                                    "taskType": _r_task,
                                    "target": r.get("target"),
                                    "targetName": r.get("target"),
                                    "featureNames": r.get("featureNames", []),
                                    "featureImportance": r.get("featureImportance", []),
                                    "testTrue": r.get("testTrueDecoded"),
                                    "testPred": r.get("testPredDecoded"),
                                    "classes": r.get("classes", []) if not _is_reg else [],
                                    "canPredict": True,
                                    # U2:基模型 OOF 排名 — 排行榜可摺疊區塊用,hydration 後也看得到
                                    "perModel": r.get("perModel", []),
                                    "metrics": {
                                        "testScore": r.get("bestScore"),
                                        "testScoreLabel": (r.get("metric") or
                                                           ("R²" if _is_reg else "F1")).upper(),
                                        "scoreBlend": r.get("scoreBlend"),
                                        "scoreStack": r.get("scoreStack"),
                                        # 分類指標 (回歸時為 None)
                                        "testAccuracy": r.get("accuracy"),
                                        "f1": r.get("f1"),
                                        # 回歸指標 (分類時為 None)
                                        "testR2": r.get("r2"),
                                        "testRMSE": r.get("rmse"),
                                        "testMAE": r.get("mae"),
                                    },
                                }
                                # 把 X_test 小切 50 筆包成 DataFrame,給 SHAP 當 background
                                _xs = r.get("xTestSample")
                                _fn = r.get("featureNames") or []
                                _x_test_df = _pd.DataFrame()
                                if _xs and _fn and len(_xs) > 0:
                                    try:
                                        _x_test_df = _pd.DataFrame(_xs, columns=_fn)
                                    except Exception as _xe:
                                        print(f"[Pipeline] xTestSample → DataFrame 失敗: {_xe}", flush=True)

                                # estimator_pkl_bytes:bundle 已是 pickle bytes,直接給 save_model
                                # 一定要傳 training_run_id + dataset_id,不然 hydration
                                # 的 modelsByRun[r.id] 找不到 → 前端 fallback 到 placeholder
                                # 把 storage size 帶進 bundle JSON,前端 hydration 時就能顯示
                                # 「模型大小: 1982MB (File)」而不是只看到一個神祕大檔
                                _ens_bundle["estimatorBytes"] = int(len(bundle_bytes))
                                _ens_bundle["estimatorMb"] = round(size_mb, 1)

                                # Item 1:訓練時 SHAP 預計算 — 對最強 tabular + 最強 DL
                                # 各跑 3 張 plotly (global / waterfall / dependence),序列化
                                # 進 bundle JSON。Insights 頁就能秒開不用再等 SHAP 即時跑。
                                try:
                                    q.put({"type": "log", "level": "info",
                                           "msg": f"[{label}] 計算 SHAP (tabular + DL 各最佳)..."})
                                    import pickle as _pickle_shap
                                    # bundle pickle 內參考 src.preprocess / src.ensemble 等模組,
                                    # 主 API 預設不在 sys.path → unpickle 會 No module named 'src',
                                    # 補上 sys.path 才能順利 loads
                                    _ensure_pipeline_path_for_unpickle()
                                    _bundle_obj = _pickle_shap.loads(bundle_bytes)
                                    _shap_plots = _compute_shap_for_ensemble_bundle(
                                        _bundle_obj,
                                        per_model=r.get("perModel", []),
                                        feature_names=r.get("featureNames", []),
                                        x_test_sample=r.get("xTestSample", []),
                                    )
                                    if _shap_plots:
                                        _ens_bundle["shapPlots"] = _shap_plots
                                        _ens_bundle["bestTabularModel"] = (_shap_plots.get("tabular") or {}).get("modelTag")
                                        _ens_bundle["bestDLModel"]     = (_shap_plots.get("dl") or {}).get("modelTag")
                                        _families_done = [k for k in _shap_plots.keys()]
                                        q.put({"type": "log", "level": "success",
                                               "msg": f"[{label}] SHAP 完成 ({', '.join(_families_done)}),已存進 bundle"})
                                    else:
                                        q.put({"type": "log", "level": "warning",
                                               "msg": f"[{label}] SHAP 預計算回空,Insights 頁仍可即時算"})
                                    del _bundle_obj   # 釋放大物件
                                except Exception as _se:
                                    import traceback as _stb
                                    print(f"[Pipeline] SHAP 預計算失敗 (不影響模型存檔): {_se}", flush=True)
                                    _stb.print_exc()
                                    q.put({"type": "log", "level": "warning",
                                           "msg": f"[{label}] SHAP 預計算失敗,Insights 頁會 fallback 到即時算: {str(_se)[:120]}"})

                                # Heartbeat:大 bundle 寫盤期間 SSE 會靜默 10-30 秒,
                                # 沒這個 log 前端會以為 hang 住或斷線。
                                q.put({"type": "log", "level": "info",
                                       "msg": f"[{label}] 持久化 ensemble bundle ({size_mb:.0f}MB) 到{'檔案系統' if size_mb > 50 else 'DB'}..."})
                                _mid = storage.save_model(
                                    bundle=_ens_bundle, estimator=None, scaler=None,
                                    X_test_df=_x_test_df, user=user, db=db,
                                    preprocessor_id=job.get("preprocessorId"),
                                    dataset_id=datasetId,
                                    training_run_id=pipeline_run_id,
                                    estimator_pkl_bytes=bundle_bytes,
                                )
                                r["bestModelId"] = _mid
                                r["estimatorMb"] = round(size_mb, 1)
                                print(f"[Pipeline] ensemble 存入 storage → modelId={_mid} "
                                      f"(run={pipeline_run_id})", flush=True)
                                # xTestSample 是 50 列特徵矩陣,前端用不到 → 拋掉
                                # featureImportance / testTrue/Pred 留著,讓「剛訓練完」就看得到圖表
                                # (這兩個欄已經在 _run_regression / _run_classification 內 cap 上限,
                                # featureImportance ≤ 200 對、testTrue/Pred ≤ 5000 列,SSE 不會爆)
                                r.pop("xTestSample", None)
                            else:
                                print(f"[Pipeline] ensembleBundlePath 不存在: {bundle_path}", flush=True)
                        except Exception as _me:
                            import traceback as _tb
                            print(f"[Pipeline] ensemble save_model 失敗 (不影響分數): {_me}", flush=True)
                            _tb.print_exc()
                            # CRITICAL:save_model 失敗時 session 是 pending-rollback 狀態,
                            # 不 rollback 後續 finish_training_run 用同 session 會炸 PendingRollbackError
                            # → SSE stream crash → 瀏覽器 "network error"。一定要主動 rollback。
                            try:
                                db.rollback()
                                print(f"[Pipeline] db.rollback() 完成,session 恢復可用", flush=True)
                            except Exception as _re:
                                print(f"[Pipeline] db.rollback() 也失敗: {_re}", flush=True)
                    elif r.get("ok"):
                        # subprocess 成功但沒回 bundle 路徑 — 通常是 _dump_ensemble_bundle 拒絕 persist
                        # (某 config 來自快取沒有 fold artifacts → bundle 沒被 dump)
                        print(f"[Pipeline] [{label}] 沒拿到 ensembleBundlePath → "
                              f"skip ensemble save (前端會 fallback 到 placeholder,需重訓才有 batch predict)",
                              flush=True)
                    r.pop("ensembleBundlePath", None)   # 不需傳進 SSE
                    r.pop("modelBytes", None)           # legacy field,保險起見也清掉

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
                    "taskType": r.get("taskType", "classification"),
                    "bestScore": r.get("bestScore"),
                    "scoreBlend": r.get("scoreBlend"),
                    "scoreStack": r.get("scoreStack"),
                    # 分類指標
                    "accuracy": r.get("accuracy"),
                    "f1": r.get("f1"),
                    # 回歸指標 (TS regression 訓的會有)
                    "rmse": r.get("rmse"),
                    "r2": r.get("r2"),
                    "mae": r.get("mae"),
                    "metric": r.get("metric"),
                    "elapsedSec": r.get("elapsedSec"),
                    "submissionAvailable": r.get("submissionAvailable", False),
                    "submissionKind": r.get("submissionKind"),
                } for r in all_results],
                "datasetName": dataset_name_for_run,
            }
            has_predictions = any(r.get("submissionAvailable") for r in all_results)
            # 從第一個成功 r 推 task_type (TS regression 跟 classification 的入口共用 endpoint)
            inferred_task = next(
                (r.get("taskType") for r in all_results if r.get("taskType")),
                None,
            )
            storage.finish_training_run(
                pipeline_run_id, user, db, status="completed",
                results_summary=summary, has_predictions=has_predictions,
                elapsed_sec=elapsed, task_type=inferred_task,
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


# ============================================================
# 2c. TRAIN (Autogluon engine) — SSE endpoint
# ============================================================
# 跟 pipeline 走同樣 SSE pattern,只是 worker 改成跑 autogluon_runner
@app.post("/api/train/autogluon/stream")
async def train_autogluon_stream_endpoint(
    # 模式 A:直接上傳 CSV
    file: UploadFile | None = File(None),
    # 模式 B:從 stored dataset (autogluon MVP 只支援 raw source — 內建自家 preprocessing)
    datasetId: str | None = Form(None),
    sources: str | None = Form(None),
    # 通用 options
    target: str | None = Form(None),
    timeLimit: float = Form(0),    # 0 = autogluon default (沒上限)
    preset: str = Form("medium_quality"),
    timeSeries: bool = Form(False),  # True → chronological split (取最後 20% 當 holdout,不 shuffle)
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
):
    """Autogluon engine 訓練 — 跟 pipeline 同 SSE pattern (LogEvent + ProgressEvent + done)。

    preset 對應 autogluon presets:
      best_quality / high_quality / good_quality / medium_quality / experimental_quality
    沒傳 timeLimit (或傳 0) → 用 autogluon 自己的 default (一般是 None = 不限時)。
    """
    from api.train.autogluon_runner import run_autogluon
    import io as _io

    direct_upload = file is not None
    from_store = datasetId is not None
    if not direct_upload and not from_store:
        raise HTTPException(status_code=400, detail="需提供 file 或 datasetId")

    # 解析 sources (跟 pipeline endpoint 同邏輯,簡化版)
    src_list: list[str] = []
    if sources:
        try: src_list = json.loads(sources)
        except Exception:
            raise HTTPException(status_code=400, detail="sources 必須是 JSON list")
    if not isinstance(src_list, list):
        raise HTTPException(status_code=400, detail="sources 必須是 list")

    base_options = {"target": target, "timeLimit": timeLimit, "preset": preset, "timeSeries": bool(timeSeries)}
    jobs: list[dict[str, Any]] = []
    dataset_name_for_run = "(uploaded)"
    target_col_sample = None

    if direct_upload:
        raw = await file.read()
        dataset_name_for_run = file.filename or "uploaded.csv"
        try:
            import pandas as _pd_infer
            _df = _pd_infer.read_csv(_io.BytesIO(raw))
            if target and target in _df.columns:
                target_col_sample = _df[target]
            del _df
        except Exception: pass
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
            if target_col_sample is None and target and target in df.columns:
                target_col_sample = df[target]
            buf = _io.StringIO()
            df.to_csv(buf, index=False)
            jobs.append({
                "label": "原始", "source": "raw",
                "csv_bytes": buf.getvalue().encode("utf-8"),
                "file_name": dataset_name_for_run,
                "options": base_options,
            })
        # 預處理 source autogluon MVP 暫不接(autogluon 內建自己的 preprocessing,雙重處理沒意義)
        if "preprocessed" in src_list and not jobs:
            raise HTTPException(status_code=400,
                detail="Autogluon engine 暫不支援 'preprocessed' source — autogluon 內建自己的 preprocessing,直接用 raw source 即可。")

    if not jobs:
        raise HTTPException(status_code=400, detail="沒有產生任何訓練任務")

    inferred_task = _infer_task_type(target_col_sample)
    run_id = storage.create_training_run(
        dataset_id=datasetId or "uploaded",
        dataset_name=dataset_name_for_run,
        engine="autogluon",
        target=target or "(auto)",
        task_type=inferred_task,
        sources=src_list if from_store else ["upload"],
        options={"preset": preset, "timeLimit": timeLimit},
        user=user, db=db,
    )

    t_start = time.time()
    cancel_token = threading.Event()
    q: queue.Queue = queue.Queue()
    all_results: list[dict] = []
    error_msg: str | None = None
    multi = len(jobs) > 1

    def _prefix(ev, label):
        """單 source 不 prefix;多 source 把 label 塞進 log/progress 給前端對齊。"""
        if not multi:
            return ev
        if ev.get("type") == "log":
            return {**ev, "msg": f"[{label}] {ev.get('msg', '')}"}
        if ev.get("type") == "progress":
            return {**ev, "step": f"[{label}] {ev.get('step', '')}", "source": label}
        return ev

    def worker():
        nonlocal error_msg
        try:
            for job in jobs:
                if cancel_token.is_set(): break
                label = job["label"]
                on_prog = lambda ev, _label=label: q.put(_prefix(ev, _label))
                r = run_autogluon(
                    csv_bytes=job.get("csv_bytes"),
                    file_name=job.get("file_name"),
                    options=job["options"],
                    on_progress=on_prog,
                    cancel_token=cancel_token,
                )
                r["dataSource"] = job["source"]
                r["dataSourceLabel"] = label
                # 把 autogluon bundle (tar.gz) 寫進 file blob → 之後 hydration 可載回
                if r.get("ok") and r.get("bundlePath") and os.path.exists(r["bundlePath"]):
                    try:
                        with open(r["bundlePath"], "rb") as _bf:
                            bundle_bytes = _bf.read()
                        size_mb = len(bundle_bytes) / 1024 / 1024
                        q.put({"type": "log", "level": "info",
                               "msg": f"[{label}] 持久化 autogluon bundle ({size_mb:.0f}MB) "
                                      f"到{'檔案系統' if size_mb > 50 else 'DB'}..."})
                        _ens_bundle = {
                            "type": "autogluon_model",
                            "name": f"[{label}] Autogluon ({r.get('bestModel', '?')})",
                            "dataSource": job["source"],
                            "dataSourceLabel": label,
                            "taskType": r.get("taskType", "classification"),
                            "target": r.get("target"),
                            "targetName": r.get("target"),
                            "featureNames": r.get("featureNamesRaw") or [],   # 原始欄名 (autogluon 內部自己處理但 SHAP 需要)
                            "feature_names_raw": r.get("featureNamesRaw") or [],
                            "testTrue": r.get("testTrueDecoded"),
                            "testPred": r.get("testPredDecoded"),
                            "xTestSample": r.get("xTestSample"),     # SHAP 用,50 列原始特徵
                            "canPredict": True,    # 已實作 autogluon 批次預測 + SHAP
                            "perModel": r.get("perModel", []),
                            "leaderboardRaw": r.get("leaderboardRaw", []),
                            "metrics": {
                                "testScore": r.get("bestScore"),
                                "testScoreLabel": r.get("metric", "Score").upper(),
                                "testAccuracy": r.get("accuracy"),
                                "f1": r.get("f1"),
                                "testR2": r.get("r2"),
                                "testRMSE": r.get("rmse"),
                                "testMAE": r.get("mae"),
                            },
                            "estimatorBytes": int(len(bundle_bytes)),
                            "estimatorMb": round(size_mb, 1),
                            "presetUsed": r.get("presetUsed"),
                            # 訓練時間明確存進 bundle,hydration 後前端能讀到 (不然 m.trainTime 會掉)
                            "trainTimeMs": int(float(r.get("elapsedSec") or 0) * 1000),
                        }

                        # 【SHAP precompute】訓練完就算 SHAP,寫進 bundle.shapPlots,
                        # Insights 頁秒開,不用每次按下分析等 PermutationExplainer 跑 5-15 分。
                        # 跟 daniel 的 _ens_bundle["shapPlots"] 寫法對齊,前端用同一條 renderer。
                        try:
                            q.put({"type": "log", "level": "info",
                                   "msg": f"[{label}] 計算 SHAP (precompute,Insights 頁秒開)..."})
                            _shap_plots = _compute_shap_plots_for_autogluon_bundle(
                                bundle_bytes=bundle_bytes,
                                x_test_sample=r.get("xTestSample") or [],
                                feature_names=r.get("featureNamesRaw") or [],
                                task_type=r.get("taskType", "classification"),
                                model_name=_ens_bundle["name"],
                                oof_score=r.get("bestScore"),
                                cache_key=None,   # 還沒 save_model,沒 model_id;helper 內部 tmp unpack 一次
                            )
                            if _shap_plots:
                                _ens_bundle["shapPlots"] = _shap_plots
                                _ens_bundle["bestTabularModel"] = _ens_bundle["name"]
                                q.put({"type": "log", "level": "success",
                                       "msg": f"[{label}] SHAP precompute 完成,已存進 bundle"})
                            else:
                                q.put({"type": "log", "level": "warning",
                                       "msg": f"[{label}] SHAP precompute 失敗,Insights 頁會 fallback 即時算"})
                        except Exception as _se:
                            import traceback as _stb
                            print(f"[Autogluon] SHAP precompute 失敗 (不影響模型存檔): {_se}", flush=True)
                            _stb.print_exc()
                            q.put({"type": "log", "level": "warning",
                                   "msg": f"[{label}] SHAP precompute 失敗,Insights 頁會 fallback: {str(_se)[:120]}"})

                        _mid = storage.save_model(
                            bundle=_ens_bundle, estimator=None, scaler=None,
                            X_test_df=None, user=user, db=db,
                            preprocessor_id=None, dataset_id=datasetId,
                            training_run_id=run_id,
                            estimator_pkl_bytes=bundle_bytes,
                        )
                        r["bestModelId"] = _mid
                        r["estimatorMb"] = round(size_mb, 1)
                        # 順手刪 tarball (已經進 file blob 了)
                        try: os.unlink(r["bundlePath"])
                        except Exception: pass
                    except Exception as _me:
                        import traceback as _tb
                        print(f"[Autogluon] save_model 失敗 (不影響分數): {_me}", flush=True)
                        _tb.print_exc()
                        try: db.rollback()
                        except Exception: pass
                r.pop("bundlePath", None)
                all_results.append(r)
        except Exception as e:
            error_msg = str(e)
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    async def event_stream():
        nonlocal error_msg
        try:
            while True:
                ev = q.get()
                if ev is None: break
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                try:
                    if ev.get("type") == "progress":
                        storage.update_training_progress(run_id, user, db,
                            pct=ev.get("pct"), step=ev.get("step"))
                    elif ev.get("type") == "log":
                        storage.update_training_progress(run_id, user, db,
                            log_line={"msg": ev.get("msg", ""), "level": ev.get("level", "info")})
                except Exception: pass
        except (GeneratorExit, asyncio.CancelledError):
            cancel_token.set()
            try: storage.finish_training_run(run_id, user, db,
                                             status="failed", error_msg="使用者取消",
                                             elapsed_sec=round(time.time() - t_start, 2))
            except Exception: pass
            raise

        elapsed = round(time.time() - t_start, 2)
        if error_msg:
            storage.finish_training_run(run_id, user, db,
                                        status="failed", error_msg=error_msg, elapsed_sec=elapsed)
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg}, ensure_ascii=False)}\n\n"
        else:
            summary = {"perSource": [{
                "source": r.get("dataSource"), "label": r.get("dataSourceLabel"),
                "taskType": r.get("taskType", "classification"),
                "bestScore": r.get("bestScore"),
                "accuracy": r.get("accuracy"), "f1": r.get("f1"),
                "rmse": r.get("rmse"), "r2": r.get("r2"), "mae": r.get("mae"),
                "metric": r.get("metric"), "elapsedSec": r.get("elapsedSec"),
                "bestModel": r.get("bestModel"), "preset": r.get("presetUsed"),
            } for r in all_results], "datasetName": dataset_name_for_run}
            inferred = next((r.get("taskType") for r in all_results if r.get("taskType")), None)
            storage.finish_training_run(run_id, user, db,
                status="success", elapsed_sec=elapsed,
                results_summary=summary, task_type=inferred)
            # 【NaN safety】all_results 可能含 NaN (House Prices 之類有缺失的資料集)。
            # 直接 json.dumps 預設輸出 NaN literal (無效 JSON) → 前端 JSON.parse 靜默 catch,
            # finalResults 永遠不會被 set → 拋「未收到 autogluon 結果」。
            # 用 _safe_json_payload helper 一次性把 NaN/Inf 換成 None。
            yield f"data: {_safe_json_payload({'type': 'done', 'runId': run_id, 'results': all_results})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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


class _EnsembleSHAPAdapter:
    """讓 AutoMLVisualizer 把 ensemble bundle 當成單一 sklearn 模型 — 暴露 predict / predict_proba。
    SHAP 會走 PermutationExplainer (model-agnostic),只要 predict / predict_proba 能跑就能算,
    只是會比樹模型慢很多 (要 replay 整個 ensemble: fold-averaged → blender + stacker)。

    回歸 bundle:沒有 predict_proba,只暴露 predict (連續值,已 inverse-transform 回原 y 尺度)。
    分類 bundle:predict 走 argmax(predict_proba)。
    """

    def __init__(self, bundle: dict):
        self.bundle = bundle
        task_type = bundle.get("task_type", "classification")
        self.is_regression = (task_type == "regression"
                              or int(bundle.get("n_classes", 0)) <= 1)

    def predict(self, X):
        X = np.asarray(X)
        if self.is_regression:
            return _ensemble_replay(self.bundle, X, return_proba=False)
        return _ensemble_replay(self.bundle, X, return_proba=False)

    def predict_proba(self, X):
        if self.is_regression:
            raise AttributeError("regression ensemble has no predict_proba")
        return _ensemble_replay(self.bundle, np.asarray(X), return_proba=True)


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
    _ensure_pipeline_path_for_unpickle()
    entry = storage.get_model(req.modelId, user, db)

    estimator = entry.get("estimator")
    X_test_df = entry.get("X_test_df")
    feature_names = entry.get("featureNames", [])
    bundle_meta = entry.get("bundle") or {}

    # Daniel ensemble → 包 adapter,讓 AutoMLVisualizer 用 Permutation 算 SHAP
    is_ensemble = bundle_meta.get("type") == "daniel_pipeline_ensemble"
    is_autogluon = bundle_meta.get("type") == "autogluon_model"

    # Item 1C:若 bundle 已預計算 SHAP 持久化,優先直接回(秒級;不用算)
    # autogluon 也走這條 — 訓練時 worker 已用 _compute_shap_plots_for_autogluon_bundle 把
    # shapPlots 寫進 bundle。沒寫成功才 fallback 到下面的 live compute。
    if (is_ensemble or is_autogluon) and isinstance(bundle_meta.get("shapPlots"), dict) and bundle_meta["shapPlots"]:
        _sp = bundle_meta["shapPlots"]
        return {
            "modelId":         req.modelId,
            "source":          "precomputed",   # 前端顯示「✓ Persistent」徽章
            "shapPlots":       _sp,             # {"tabular": {...}, "dl": {...}} 各含 global/waterfall/dependence
            "bestTabularModel": bundle_meta.get("bestTabularModel"),
            "bestDLModel":     bundle_meta.get("bestDLModel"),
            "featureNames":    (_sp.get("tabular") or {}).get("featureNames", []) or
                               (_sp.get("dl") or {}).get("featureNames", []),
        }

    # Autogluon model 沒 precomputed shapPlots → fallback live 算 (舊 bundle 沒 precompute 過)
    if is_autogluon:
        return _compute_shap_for_autogluon(req.modelId, entry, bundle_meta)

    if is_ensemble:
        if not (isinstance(estimator, dict) and estimator.get("version") == 1):
            _est_status = entry.get("estimatorStatus", "empty")
            if _est_status == "file_missing":
                _msg = "ensemble file blob 已不存在 (model_blobs/ 被清掉或換機器了),SHAP 不可用。請重訓。"
            else:
                _msg = "ensemble bundle 在 DB 沒有 estimator (舊 placeholder 或 pickle 失敗),SHAP 不可用。請重訓。"
            raise HTTPException(status_code=400, detail=_msg)
        # ensemble 的 featureNames 用 bundle 內的 raw 特徵名
        feature_names = estimator.get("feature_names_raw", feature_names)
        # Item 5 防禦性清洗:預處理 source 訓的舊 bundle 可能有 'num_pipeline__SalePrice'
        # 這種 sklearn ColumnTransformer 前綴。daniel 新版會自動 split('__')[-1],這裡
        # 補一層保險讓 SHAP 圖顯示乾淨名(如 'SalePrice')。OHE 欄會帶 '_NoRidge' 後綴,
        # 這部分無解(訓練時就拼起來了),只能再做 mapping(本輪不做)。
        feature_names = [str(n).split("__")[-1] for n in (feature_names or [])]
        estimator = _EnsembleSHAPAdapter(estimator)

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
    _ensure_pipeline_path_for_unpickle()
    entry = storage.get_model(req.modelId, user, db)

    estimator = entry["estimator"]
    scaler = entry["scaler"]
    feature_names = entry["featureNames"]
    preprocessor_id = entry.get("preprocessorId")
    bundle_meta = entry.get("bundle") or {}

    # Daniel ensemble:走 fold-averaged → blender + stacker flow
    if bundle_meta.get("type") == "daniel_pipeline_ensemble":
        if not (isinstance(estimator, dict) and estimator.get("version") == 1 and "configs" in estimator):
            _est_status = entry.get("estimatorStatus", "empty")
            if _est_status == "file_missing":
                _msg = "此 ensemble 模型的 file blob 已不存在 (model_blobs/ 被清掉或換機器了),無法預測。請重訓。"
            else:
                _msg = "此 ensemble 模型在 DB 沒有 estimator (舊 placeholder 或 pickle 失敗),無法預測。請重訓。"
            raise HTTPException(status_code=410, detail=_msg)
        if len(req.features) != len(feature_names):
            raise HTTPException(status_code=400,
                detail=f"features 長度 {len(req.features)} 不符 (應為 {len(feature_names)})")
        return _predict_single_ensemble(estimator, req.features, feature_names,
                                       preprocessor_id, user, db)

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


def _predict_single_ensemble(bundle: dict, features: list, feature_names: list,
                             preprocessor_id: Optional[str], user, db) -> dict:
    """單筆 ensemble 預測 — 給 What-If simulator 用。

    feature_names 已經是 ensemble 訓練時看到的欄位順序:
      - raw source ensemble → 原始欄名
      - preprocessed source ensemble → transformed 欄名 (前端 What-If slider 顯示這些)
    """
    import pandas as pd

    # 單筆 N×1 → DataFrame (column 名跟 model 訓練時看到的對齊)
    one_row_df = pd.DataFrame([features], columns=feature_names)
    X_raw = one_row_df.values.astype(np.float32)

    task_type = bundle.get("task_type", "classification")
    is_reg = task_type == "regression" or int(bundle.get("n_classes", 0)) <= 1

    if is_reg:
        # 回歸:回連續值 + 5 fold 的標準差當「不確定度」,前端可畫 ±1σ 區間。
        # 直接收每 fold 預測值 (不走 blender/stacker,只是純不確定度估計)。
        y_center = bundle.get("y_center")
        y_scale  = bundle.get("y_scale")
        per_fold_preds = []
        for cfg_entry in bundle["configs"]:
            folds = cfg_entry.get("folds") or []
            for fold in folds:
                fb = fold.get("fb")
                X_fb = fb.transform(X_raw) if fb is not None else X_raw
                if "model" in fold:
                    p = fold["model"].predict(X_fb)
                else:
                    p = _predict_dl_fold(fold, X_fb, 1)
                p = np.asarray(p, dtype=float).ravel()
                # 逆 RobustScaler 還原回原 y 尺度
                if y_center is not None and y_scale is not None:
                    p = p * float(y_scale) + float(y_center)
                per_fold_preds.append(float(p[0]))
        # 走 ensemble blender/stacker 拿正式 prediction (跟 batch predict 一致)
        pred_arr = _ensemble_replay(bundle, X_raw, return_proba=False)
        val = float(np.asarray(pred_arr).ravel()[0])
        std = float(np.std(per_fold_preds)) if len(per_fold_preds) >= 2 else 0.0
        return {
            "prediction": val,
            "proba": None,
            "classes": None,
            # 回歸專屬欄位:讓 What-If UI 顯示 ±1σ 信心區間,以及單 fold 預測分布
            "predictionStd": std,
            "predictionCI68": [val - std, val + std],            # ±1σ ≈ 68% CI
            "predictionCI95": [val - 1.96 * std, val + 1.96 * std],
            "foldPredictions": per_fold_preds,
            "isRegression": True,
        }

    # 分類
    proba_arr = _ensemble_replay(bundle, X_raw, return_proba=True)
    proba_row = np.asarray(proba_arr)[0]
    pred_idx = int(np.argmax(proba_row)) if proba_row.ndim >= 1 else int(proba_row)

    le = bundle.get("label_encoder")
    n_classes = int(bundle.get("n_classes", 0))
    if le is not None:
        pred_label = le.inverse_transform([pred_idx])[0]
        if hasattr(pred_label, "item"):
            pred_label = pred_label.item()
        classes = [c.item() if hasattr(c, "item") else c for c in le.classes_]
    else:
        pred_label = pred_idx
        classes = list(range(n_classes))

    return {
        "prediction": pred_label,
        "proba": proba_row.tolist() if hasattr(proba_row, "tolist") else [float(proba_row)],
        "classes": classes,
    }


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
def _ensure_pipeline_path_for_unpickle():
    """Daniel ensemble bundle pickle 內參考 src.preprocess / src.ensemble 等模組,
    主 API 預設不在 sys.path,unpickle 會 ModuleNotFoundError → 在這先補上。
    idempotent — 多次呼叫只 insert 一次。"""
    _pipe = os.path.join(os.path.dirname(__file__), "train", "pipeline")
    if _pipe not in sys.path:
        sys.path.insert(0, _pipe)


# 訓練時 SHAP 預先計算 — Item 1
# ============================================================
# 對 ensemble bundle 內 OOF 最強的 tabular + DL 基模型各跑 SHAP,
# 各回 3 張 plotly figure JSON。失敗一個 family 不影響另一個。
# 取代「按下分析才即時算」的延遲 — 訓練完就存進 bundle,Insights 頁秒開。
# ============================================================
_SHAP_TABULAR_NAMES = {"lgbm", "xgb", "catboost", "rf", "extra_trees", "logreg", "knn", "ridge"}
_SHAP_DL_NAMES      = {"mlp", "cnn1d", "resnet1d", "tcn", "transformer", "patchtst", "tsnet"}
_SHAP_ALL_NAMES     = sorted(_SHAP_TABULAR_NAMES | _SHAP_DL_NAMES, key=len, reverse=True)


def _shap_family_of_tag(tag: str) -> Optional[str]:
    """從 fold tag(例如 'reg_lgbm_raw_stat_c0_xxxxxx_yrs')還原家族。"""
    t = tag[4:] if tag.startswith("reg_") else tag
    for n in _SHAP_ALL_NAMES:
        if t.startswith(n + "_"):
            return "tabular" if n in _SHAP_TABULAR_NAMES else "dl"
    return None


def _compute_shap_plots_for_autogluon_bundle(
    bundle_bytes: bytes,
    x_test_sample: list,
    feature_names: list,
    task_type: str,
    model_name: str = "Autogluon",
    oof_score: Optional[float] = None,
    cache_key: Optional[str] = None,
) -> Optional[dict]:
    """純函式版 — 給定 bundle bytes 跟 x_test_sample 就算 SHAP 出來。
    給「訓練時 precompute」跟「on-demand SHAP endpoint」共用。

    cache_key:有的話會走 _AG_UNPACK_CACHE,沒給就臨時 unpack 一次。

    回傳 {"tabular": {modelTag, oofScore, featureNames, global, waterfall, dependence}} 或 None (失敗)。
    跟 daniel 的 _compute_shap_for_ensemble_bundle 回傳格式對齊,前端用同一條渲染器。
    """
    import pandas as pd
    import plotly.io as pio
    try:
        from api.visualize import AutoMLVisualizer
    except Exception as e:
        print(f"[autogluon_shap_precompute] AutoMLVisualizer 無法 import: {e}", flush=True)
        return None
    if not x_test_sample or not feature_names:
        print("[autogluon_shap_precompute] xTestSample / featureNames 為空,跳過", flush=True)
        return None

    # 解 tar.gz (走 cache 若有 model_id)
    try:
        if cache_key:
            predictor, _ = _load_autogluon_predictor(cache_key, bundle_bytes)
        else:
            # 臨時 unpack — 訓練完就算的場景,還沒有 model_id
            import tarfile as _tar, tempfile as _tf, io as _io
            unpack_root = _tf.mkdtemp(prefix="ag_unpack_precompute_")
            with _tar.open(fileobj=_io.BytesIO(bundle_bytes), mode="r:gz") as tar:
                tar.extractall(unpack_root)
            predictor_dir = os.path.join(unpack_root, "autogluon")
            if not os.path.isdir(predictor_dir):
                subdirs = [os.path.join(unpack_root, d) for d in os.listdir(unpack_root)
                           if os.path.isdir(os.path.join(unpack_root, d))]
                if subdirs:
                    predictor_dir = subdirs[0]
            from autogluon.tabular import TabularPredictor
            predictor = TabularPredictor.load(predictor_dir,
                require_version_match=False, require_py_version_match=False)
    except Exception as e:
        import traceback as _tb
        print(f"[autogluon_shap_precompute] load predictor 失敗: {e}", flush=True)
        _tb.print_exc()
        return None

    is_reg = task_type == "regression"

    # mixed-type 編碼/解碼 (同 _compute_shap_for_autogluon 內邏輯)
    X_raw = pd.DataFrame(x_test_sample[:50], columns=feature_names)
    _decode_maps: dict[str, list] = {}
    X_encoded = X_raw.copy()
    for col in X_encoded.columns:
        s = X_encoded[col]
        if s.dtype == object or pd.api.types.is_string_dtype(s) or pd.api.types.is_categorical_dtype(s):
            uniques = list(pd.unique(s.fillna("__NA__")))
            _decode_maps[col] = uniques
            mapping = {v: i for i, v in enumerate(uniques)}
            X_encoded[col] = s.fillna("__NA__").map(mapping).astype(float)
        else:
            X_encoded[col] = pd.to_numeric(s, errors="coerce")
            if X_encoded[col].isna().any():
                _med = X_encoded[col].median()
                X_encoded[col] = X_encoded[col].fillna(_med if pd.notna(_med) else 0.0)

    class _AGWrap:
        def predict(_self, X):
            import pandas as _pd
            df_enc = _pd.DataFrame(X, columns=feature_names) if not isinstance(X, _pd.DataFrame) else X.copy()
            df_in = df_enc.copy()
            for col, uniques in _decode_maps.items():
                if col not in df_in.columns:
                    continue
                idx = df_in[col].round().clip(0, len(uniques) - 1).astype(int)
                df_in[col] = idx.map(lambda i: uniques[i] if 0 <= i < len(uniques) else "__NA__")
                df_in[col] = df_in[col].replace("__NA__", None)
            if is_reg:
                return predictor.predict(df_in).values
            try:
                proba = predictor.predict_proba(df_in)
                if proba.shape[1] == 2:
                    return proba.iloc[:, 1].values
                return proba.values
            except Exception:
                return predictor.predict(df_in).values

    try:
        viz = AutoMLVisualizer(_AGWrap(), X_encoded, output_dir=None)
        target_feat = feature_names[0] if feature_names else "f0"
        fig_global = viz.generate_beeswarm_plot(return_fig=True)
        fig_wf     = viz.generate_waterfall_plot(sample_index=0, return_fig=True)
        fig_dep    = viz.generate_dependence_plot(target_feature=target_feat, return_fig=True)
    except Exception as e:
        import traceback as _tb
        print(f"[autogluon_shap_precompute] SHAP 計算失敗: {e}", flush=True)
        _tb.print_exc()
        return None

    def _fig_to_json(fig):
        return json.loads(pio.to_json(fig))

    return {
        "tabular": {
            "modelTag": model_name,
            "oofScore": oof_score,
            "featureNames": feature_names,
            "global":     _fig_to_json(fig_global),
            "waterfall":  _fig_to_json(fig_wf),
            "dependence": _fig_to_json(fig_dep),
        }
    }


def _compute_shap_for_autogluon(model_id: str, entry: dict, bundle_meta: dict) -> dict:
    """Autogluon model 的 SHAP — unpack tar.gz → 包 predictor.predict 走 PermutationExplainer。

    回傳格式跟 daniel ensemble 對齊 (前端用同一條 _renderShapPrecomputed 渲染):
      {modelId, source='live_autogluon', shapPlots: {tabular: {global/waterfall/dependence}}, ...}
    """
    import pandas as pd
    import plotly.io as pio
    try:
        from api.visualize import AutoMLVisualizer
    except Exception as e:
        raise HTTPException(500, f"AutoMLVisualizer 無法 import: {e}")

    bundle_bytes = entry.get("_estimator_bytes")
    if not bundle_bytes:
        _est_status = entry.get("estimatorStatus", "empty")
        _msg = ("Autogluon file blob 已不存在,SHAP 不可用。請重訓。"
                if _est_status == "file_missing" else
                "Autogluon bundle bytes 不存在(舊紀錄),SHAP 不可用。請重訓。")
        raise HTTPException(410, _msg)

    x_test_sample = bundle_meta.get("xTestSample") or []
    feature_names = bundle_meta.get("featureNames") or bundle_meta.get("feature_names_raw") or []
    if not x_test_sample or not feature_names:
        raise HTTPException(400,
            "此 Autogluon 模型未保存 xTestSample / featureNames(舊紀錄)。請重訓即可。")

    # 解 tar.gz + load TabularPredictor
    try:
        predictor, _ = _load_autogluon_predictor(model_id, bundle_bytes)
    except HTTPException:
        raise
    except Exception as e:
        import traceback as _tb
        _tb.print_exc()
        raise HTTPException(500, f"Autogluon predictor load 失敗: {e}")

    # 包成 predictor.predict 給 AutoMLVisualizer 的 PermutationExplainer 用
    task_type = bundle_meta.get("taskType", "classification")
    is_reg = task_type == "regression"

    # 【mixed-type SHAP 修法】House Prices 那種真實表格資料含 string 類別欄(MSZoning / Neighborhood 等)。
    # shap.Explainer 內部 PermutationExplainer 對 X 做算術 (例如 mean / 減法) → 對 str 拋
    # "unsupported operand type(s) for -: 'str' and 'str'"。
    # 解法:把 string 欄轉成 int code (透明對 SHAP),predict wrapper 內再 decode 回原字串給 predictor。
    X_raw = pd.DataFrame(x_test_sample[:50], columns=feature_names)
    _decode_maps: dict[str, list] = {}   # col_name -> [unique_values],int code 對應 index
    X_encoded = X_raw.copy()
    for col in X_encoded.columns:
        s = X_encoded[col]
        # 用 pandas.api.types 判:object / string / category → 編碼
        if s.dtype == object or pd.api.types.is_string_dtype(s) or pd.api.types.is_categorical_dtype(s):
            uniques = list(pd.unique(s.fillna("__NA__")))
            _decode_maps[col] = uniques
            mapping = {v: i for i, v in enumerate(uniques)}
            X_encoded[col] = s.fillna("__NA__").map(mapping).astype(float)
        else:
            # 數值欄 NaN 填中位數 (SHAP 也吃不了 NaN)
            X_encoded[col] = pd.to_numeric(s, errors="coerce")
            if X_encoded[col].isna().any():
                _med = X_encoded[col].median()
                X_encoded[col] = X_encoded[col].fillna(_med if pd.notna(_med) else 0.0)

    class _AGWrap:
        """讓 shap.Explainer 看 numeric encoded X;內部 decode 回原字串給 predictor。"""
        def predict(_self, X):
            import pandas as _pd
            df_enc = _pd.DataFrame(X, columns=feature_names) if not isinstance(X, _pd.DataFrame) else X.copy()
            # decode:int code → 原字串 (對 _decode_maps 有登記的欄做反向查表)
            df_in = df_enc.copy()
            for col, uniques in _decode_maps.items():
                if col not in df_in.columns:
                    continue
                # round + clip 避免 SHAP perturbation 後產出非整數值,出界 idx 全當 __NA__
                idx = df_in[col].round().clip(0, len(uniques) - 1).astype(int)
                df_in[col] = idx.map(lambda i: uniques[i] if 0 <= i < len(uniques) else "__NA__")
                # __NA__ sentinel 換回 NaN 給 autogluon predictor (它會走自己的 NaN 處理)
                df_in[col] = df_in[col].replace("__NA__", None)
            if is_reg:
                return predictor.predict(df_in).values
            try:
                proba = predictor.predict_proba(df_in)
                if proba.shape[1] == 2:
                    return proba.iloc[:, 1].values
                return proba.values
            except Exception:
                return predictor.predict(df_in).values

    X_df = X_encoded   # SHAP 看到的版本

    try:
        viz = AutoMLVisualizer(_AGWrap(), X_df, output_dir=None)
        target_feat = feature_names[0] if feature_names else "f0"
        fig_global = viz.generate_beeswarm_plot(return_fig=True)
        fig_wf     = viz.generate_waterfall_plot(sample_index=0, return_fig=True)
        fig_dep    = viz.generate_dependence_plot(target_feature=target_feat, return_fig=True)
    except Exception as e:
        import traceback as _tb
        _tb.print_exc()
        raise HTTPException(500, f"Autogluon SHAP 計算失敗: {e}")

    def _fig_to_json(fig):
        return json.loads(pio.to_json(fig))

    tabular_plots = {
        "modelTag": bundle_meta.get("name", "Autogluon"),
        "oofScore": (bundle_meta.get("metrics") or {}).get("testScore"),
        "featureNames": feature_names,
        "global":     _fig_to_json(fig_global),
        "waterfall":  _fig_to_json(fig_wf),
        "dependence": _fig_to_json(fig_dep),
    }

    # source="precomputed" 是前端 _renderShapPrecomputed 那條 path 的觸發 flag
    # (autogluon 雖然不是訓練時 precompute,但結構跟 daniel ensemble 對齊 → 共用渲染器)。
    # 如果改 "live_autogluon" 前端會 fallthrough 到舊版單組 SHAP 分支,讀不到 res.global → undefined.data 炸。
    return {
        "modelId":     model_id,
        "source":      "precomputed",
        "shapPlots":   {"tabular": tabular_plots},   # 沒有 DL 家族,前端 sp.dl 那段會自動隱藏
        "bestTabularModel": bundle_meta.get("name", "Autogluon"),
        "bestDLModel": None,
        "featureNames": feature_names,
    }


def _build_readable_feature_names(fb, base_names: list, total_dims: int) -> list:
    """把 FeatureBuilder 展開後的 X_fb 欄,從 f0/f1/... 換成可讀名稱。

    Daniel FeatureBuilder.transform 對 ts_tabular_fft / raw_stat / poly2 等 feature_set 會 hstack 多段:
      [X_scaled (= base_names)] + [ts_win 13×F] + [fft ~50] + [stat 全域 20 + 局部 4×n_segments]

    - 前 len(base_names) 欄 = 直接對應 preprocessor 輸出名 (例如 "MSSubClass", "MSZoning_RL")
    - ts_win 段:每 F 欄一組 — diff / lag1 / lag2 / roll3_mean / roll3_std / roll3_max / roll3_min /
                              roll5_mean / roll5_std / roll5_max / roll5_min / ema3 / ema5
    - fft 段:全域聚合統計 (mean / std / max / argmax / 8 band%) + 頻譜形狀 (centroid/spread/rolloff/entropy) + top20
    - stat 段:全域 20 維 (mean/std/q5/q25/q75/q95/min/max/skew/kurt/median/abs_mean/abs_max/
                          pos_ratio/ptp/var/energy/rms/mean_abs_diff/zcr) + 局部 n_segments × 4

    對應不上的(未知 feature_set 或長度算不準)→ 給 generic 但帶 section 標籤的名字,
    例如 "ts_win_idx42" 比死硬 "f867" 更可懂。
    """
    F = len(base_names)
    if total_dims == F:
        return list(base_names)

    feature_set = getattr(fb, "feature_set", "") if fb is not None else ""
    n_segments = int(getattr(fb, "n_segments", 8)) if fb is not None else 8
    names = list(base_names)  # 前 F 欄一律是 base_names

    def _gen_ts_win(F: int) -> list:
        ops = ["diff", "lag1", "lag2",
               "roll3_mean", "roll3_std", "roll3_max", "roll3_min",
               "roll5_mean", "roll5_std", "roll5_max", "roll5_min",
               "ema3", "ema5"]  # 13 段,跟 _ts_window_features 的 parts 順序對齊
        out = []
        for op in ops:
            for b in base_names:
                out.append(f"{op}({b})")
        return out  # 13 × F

    def _gen_fft() -> list:
        # _fft_features 的順序:mean / std / max / argmax / band0% ~ band7% (8)
        # / centroid / spread / rolloff / entropy / top0 ~ top19 (20)
        return (
            ["fft_mean", "fft_std", "fft_max", "fft_argmax"]
            + [f"fft_band{i}_pct" for i in range(8)]
            + ["fft_centroid", "fft_spread", "fft_rolloff", "fft_entropy"]
            + [f"fft_top{i}" for i in range(20)]
        )  # 4 + 8 + 4 + 20 = 36

    def _gen_stat() -> list:
        # _stat_features 順序:全域 20 維 + 局部 n_segments × 4
        global_names = ["stat_mean", "stat_std", "stat_q5", "stat_q25", "stat_q75", "stat_q95",
                        "stat_min", "stat_max", "stat_skew", "stat_kurt", "stat_median",
                        "stat_abs_mean", "stat_abs_max", "stat_pos_ratio", "stat_ptp", "stat_var",
                        "stat_energy", "stat_rms", "stat_mean_abs_diff", "stat_zcr"]
        local_names = []
        for kind in ("mean", "std", "max", "min"):
            for i in range(n_segments):
                local_names.append(f"stat_{kind}_seg{i}")
        return global_names + local_names  # 20 + n_segments × 4

    # 依 feature_set 決定後續段的順序
    if feature_set == "raw":
        pass
    elif feature_set == "signal":
        # X_scaled + row_norm (F)
        names += [f"l2norm({b})" for b in base_names]
    elif feature_set in ("pca64", "svd64", "kpca32"):
        # 後段是降維,沒原始欄對應 → reduced_{i}
        names += [f"{feature_set}_d{i}" for i in range(total_dims - F)]
    elif feature_set == "poly2":
        # X_scaled + interactions (數量不定,依 top-k 變數)
        names += [f"poly2_inter{i}" for i in range(total_dims - F)]
    elif feature_set == "ts_tabular":
        names += _gen_ts_win(F)
        names += _gen_stat()
    elif feature_set == "ts_tabular_fft":
        names += _gen_ts_win(F)
        names += _gen_fft()
        names += _gen_stat()
    elif feature_set == "raw_stat":
        names += _gen_stat()
        # 後面可能還有 kmeans cluster 距離(若有開 use_kmeans)
    elif feature_set == "raw_stat_fft":
        names += _gen_stat()
        names += _gen_fft()
        # 後面可能還有 kmeans cluster 距離

    # 若我們算出來的 names 仍跟 total_dims 對不上(未知 feature_set / 算錯 / kmeans 等)
    # 剩餘欄用 generic 但帶 feature_set 前綴
    if len(names) < total_dims:
        extra = total_dims - len(names)
        names += [f"{feature_set or 'extra'}_d{i}" for i in range(extra)]
    elif len(names) > total_dims:
        names = names[:total_dims]
    return names


def _map_transformed_to_original(tx_cols: list, base_names: list) -> tuple:
    """把 transformed 欄(含 OneHot 展開 + FeatureBuilder 衍生)反查回原始 CSV 欄。

    回傳:
      mapped_names: 跟 tx_cols 等長,每個位置是「應該歸屬到的原始 CSV 欄名」
      ohe_prefixes: 偵測到的 OneHot 來源欄(原始 CSV 名),供 caller debug

    邏輯:
      ① 偵測 base_names (前 75 個 = preprocessor 輸出) 中的 OneHot 群組:
         - 多個欄共用 "Prefix_" 前綴 + 該 Prefix 本身不存在於 base_names → OHE 來源
         - 例:MSZoning_RL/RM/FV 三個都有 "MSZoning_" 前綴,且沒有獨立的 "MSZoning" → MSZoning 是 OHE 來源
         - LotArea 沒 underscore → 不是 OHE,保持 LotArea
         - 1stFlrSF 雖有底線但 "1stFlrSF" 本身就在 base_names → 不算 OHE
      ② 對 tx_cols 的每個名字:
         - 如果是 op(X) 包裝的衍生欄 → 提出 X,套用 OHE 映射,結果不再加包裝(讓 op 跟 OHE 都聚合到 X 的原始欄)
         - 純全域衍生欄如 stat_mean_seg2 / fft_band3_pct → 保留(不屬於任何單一原始欄)
         - 直接的 base name → 套用 OHE 映射
    """
    import re
    from collections import defaultdict

    # ── ① 偵測 OneHot 來源欄
    standalone = set(base_names)
    prefix_groups = defaultdict(list)
    for n in base_names:
        if "_" in n:
            prefix = n.split("_", 1)[0]
            prefix_groups[prefix].append(n)
    ohe_prefixes = {
        p for p, group in prefix_groups.items()
        if len(group) >= 2 and p not in standalone
    }

    def _strip_to_original(name: str) -> str:
        """套用 OHE prefix 映射,回傳原始 CSV 欄名(若無法判定就回傳原 name)。"""
        if name in ohe_prefixes:
            return name  # 不應該發生
        if "_" in name:
            prefix = name.split("_", 1)[0]
            if prefix in ohe_prefixes:
                return prefix
        return name

    # 全域衍生欄前綴(無法歸屬到單一原始欄,保留)
    GLOBAL_STAT_PREFIXES = ("stat_", "fft_", "l2norm", "pca64", "svd64", "kpca32",
                             "poly2_", "ts_tabular_", "raw_stat_", "extra_d")

    mapped = []
    for name in tx_cols:
        # 全域衍生欄 → 保留
        if any(name.startswith(p) for p in GLOBAL_STAT_PREFIXES):
            mapped.append(name)
            continue
        # op(X) 包裝的衍生欄 → 提出 X 再映射(讓所有 diff/lag/roll/ema 都聚合到原始)
        m = re.match(r'^([a-zA-Z0-9_]+)\((.+)\)$', name)
        if m:
            inner = m.group(2)
            mapped.append(_strip_to_original(inner))
            continue
        # 直接 base name
        mapped.append(_strip_to_original(name))

    return mapped, ohe_prefixes


def _aggregate_shap_by_original(viz, tx_cols: list, base_names: list):
    """SHAP 值按原始 CSV 欄聚合,直接 in-place 改 viz.X_test 跟 _get_shap_matrix。

    monkey-patch viz 的 X_test.columns 跟 _get_shap_matrix 回傳值,讓 generate_beeswarm_plot /
    generate_waterfall_plot / generate_dependence_plot 都看到聚合後的版本。
    """
    import pandas as pd
    from collections import OrderedDict

    mapped_names, ohe_prefixes = _map_transformed_to_original(tx_cols, base_names)
    if ohe_prefixes:
        print(f"[shap_precompute] OHE 來源欄偵測: {sorted(ohe_prefixes)}", flush=True)

    # 依 mapped_names 分組(保持首次出現順序)
    groups: "OrderedDict[str, list[int]]" = OrderedDict()
    for i, m in enumerate(mapped_names):
        groups.setdefault(m, []).append(i)

    if len(groups) == len(tx_cols):
        # 沒有任何聚合(都一對一) → 不動 viz,直接 return
        return

    shap_mat = viz._get_shap_matrix()  # (n_samples, n_tx_features)
    agg_names = list(groups.keys())
    agg_shap = np.zeros((shap_mat.shape[0], len(agg_names)), dtype=shap_mat.dtype)
    for j, (orig_col, indices) in enumerate(groups.items()):
        # 同源欄的 SHAP 值相加 — 對 global importance 跟 waterfall 都正確
        agg_shap[:, j] = shap_mat[:, indices].sum(axis=1)

    # 聚合後 X_test 拿 group 內第一個原欄的值當代表(僅用於 plot label / hover,
    # 不影響 SHAP 值本身)。
    agg_X = pd.DataFrame(
        {n: viz.X_test.iloc[:, groups[n][0]].values for n in agg_names}
    )

    viz.X_test = agg_X
    # 用 closure 鎖住聚合後矩陣;_get_shap_matrix 改成 lambda 回固定值
    viz._get_shap_matrix = lambda _v=agg_shap: _v
    print(
        f"[shap_precompute] 聚合 transformed→original: {len(tx_cols)} → {len(agg_names)} 欄",
        flush=True,
    )


def _compute_shap_for_ensemble_bundle(bundle: dict, per_model: list,
                                       feature_names: list, x_test_sample: list) -> dict:
    """對 ensemble 內最強的 tabular + DL 基模型各跑 3 張 SHAP plotly。

    Args:
        bundle: 已 unpickle 的 ensemble bundle dict (含 configs / blender / stacker)
        per_model: SSE 回的 perModel list (含 tag + oofScore),index 對應 bundle["configs"]
        feature_names: 訓練時的特徵名 (已 split('__')[-1] 清乾淨)
        x_test_sample: 50 列 raw X 樣本 (從 subprocess r["xTestSample"])

    Returns:
        {"tabular": {modelTag, oofScore, global, waterfall, dependence},
         "dl":      {modelTag, oofScore, global, waterfall, dependence}}
        找不到該家族 / 計算失敗的 key 缺。
    """
    import pandas as pd
    import plotly.io as pio
    try:
        from api.visualize import AutoMLVisualizer
    except Exception as e:
        print(f"[shap_precompute] AutoMLVisualizer 無法 import: {e}", flush=True)
        return {}

    # 按 OOF 分數選每家族第一名
    best: dict = {"tabular": (None, -float("inf")), "dl": (None, -float("inf"))}
    for i, m in enumerate(per_model or []):
        score = m.get("oofScore")
        if score is None:
            continue
        fam = _shap_family_of_tag(m.get("tag", ""))
        if fam and score > best[fam][1]:
            best[fam] = (i, float(score))

    if best["tabular"][0] is None and best["dl"][0] is None:
        print("[shap_precompute] 找不到任何 tabular / DL 基模型,跳過", flush=True)
        return {}

    # X sample → DataFrame(feature_names 對齊)
    if not x_test_sample:
        print("[shap_precompute] x_test_sample 空,跳過", flush=True)
        return {}
    X_arr = np.asarray(x_test_sample[:50])
    if X_arr.ndim != 2:
        return {}
    # feature_names 可能跟 X_arr 欄數不一致(舊 bundle),fallback 用 f0/f1/...
    if len(feature_names) == X_arr.shape[1]:
        cols = list(feature_names)
    else:
        cols = [f"f{i}" for i in range(X_arr.shape[1])]
    X_sample_df = pd.DataFrame(X_arr, columns=cols)

    def _fig_to_json(fig):
        return json.loads(pio.to_json(fig))

    results: dict = {}
    for family in ("tabular", "dl"):
        idx, score = best[family]
        if idx is None:
            continue
        try:
            config_entry = bundle["configs"][idx]
            folds = config_entry.get("folds") or []
            if not folds:
                print(f"[shap_precompute] {family} idx={idx} fold list 空", flush=True)
                continue
            fold = folds[0]  # 第一 fold 的 model 當代表

            # FeatureBuilder transform 一次,讓 X 對齊 model 訓練時的維度
            fb = fold.get("fb")
            if fb is not None:
                try:
                    X_fb = fb.transform(X_arr)
                except Exception as e:
                    print(f"[shap_precompute] {family} fb.transform 失敗: {e}", flush=True)
                    continue
            else:
                X_fb = X_arr

            # feature names 對齊 transformed 後的維度(FeatureBuilder 會展開/壓縮)
            # 不再 fallback "f0/f1/..." — 改成依 feature_set 結構生衍生欄的可讀名稱
            tx_cols = _build_readable_feature_names(fb, cols, X_fb.shape[1])
            X_fb_df = pd.DataFrame(X_fb, columns=tx_cols)

            if family == "tabular":
                model = fold.get("model")
                if model is None:
                    print(f"[shap_precompute] tabular fold 沒 'model' key,跳過", flush=True)
                    continue
                viz = AutoMLVisualizer(model, X_fb_df, output_dir=None)
            else:
                # DL:包成有 .predict 的 wrapper 讓 AutoMLVisualizer 走 PermutationExplainer
                n_classes = max(1, int(bundle.get("n_classes", 1)))
                class _DLWrap:
                    def predict(_self, X):
                        Xn = X.values if hasattr(X, "values") else np.asarray(X)
                        return _predict_dl_fold(fold, Xn, n_classes)
                viz = AutoMLVisualizer(_DLWrap(), X_fb_df, output_dir=None)

            # 【SHAP 聚合到原始 CSV 欄】把 transformed 欄 (含 OneHot + diff/lag/fft/stat) 反向歸位:
            # - MSZoning_RL / RM / FV 三欄 → 聚合成 MSZoning (一個 bar,SHAP 值相加)
            # - diff(LotArea) / lag1(LotArea) / roll3_mean(LotArea) → 全部聚合到 LotArea
            # - stat_mean_seg2 / fft_band3_pct 全域聚合特徵 → 無對應原始欄,保留
            # 修法在 viz 完成 SHAP 計算之後,monkey-patch viz.X_test + _get_shap_matrix,
            # 讓後續 beeswarm / waterfall / dependence 都看到聚合後的版本。
            try:
                _aggregate_shap_by_original(viz, tx_cols, cols)
            except Exception as _agg_e:
                print(f"[shap_precompute] {family} 聚合失敗,fallback transformed 欄: {_agg_e}", flush=True)

            # 三張圖 — 聚合後 viz.X_test.columns 已換成原始 CSV 欄名
            agg_cols = list(viz.X_test.columns)
            target_feat = agg_cols[0] if agg_cols else "f0"
            fig_global = viz.generate_beeswarm_plot(return_fig=True)
            fig_wf     = viz.generate_waterfall_plot(sample_index=0, return_fig=True)
            fig_dep    = viz.generate_dependence_plot(target_feature=target_feat, return_fig=True)

            results[family] = {
                "modelTag": (per_model[idx] or {}).get("tag"),
                "oofScore": (per_model[idx] or {}).get("oofScore"),
                "featureNames": agg_cols,   # 給前端 dependence 切換用 — 已是聚合後的原始 CSV 欄名
                "global":     _fig_to_json(fig_global),
                "waterfall":  _fig_to_json(fig_wf),
                "dependence": _fig_to_json(fig_dep),
            }
            print(f"[shap_precompute] ✓ {family} ({per_model[idx].get('tag')}, OOF={score:.4f})", flush=True)
        except Exception as e:
            print(f"[shap_precompute] {family} 失敗 (不影響其他): {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

    return results


def _predict_dl_fold(fold: dict, X_fb: "np.ndarray", n_classes: int) -> "np.ndarray":
    """重建 DL 模型 → load_state_dict → 推論。
    分類:回 N×C 機率矩陣 (softmax)。
    回歸:回 N 維 (model.forward 直接輸出,沒有 activation)。"""
    import torch
    _ensure_pipeline_path_for_unpickle()
    from src.train import _build_dl_model
    model = _build_dl_model(fold["model_name"], fold["arch_params"],
                            fold["in_features"], n_classes)
    model.load_state_dict(fold["state_dict"])
    model.eval()
    X_t = torch.tensor(X_fb, dtype=torch.float32)
    with torch.no_grad():
        out = model(X_t)
        if n_classes == 1:
            # 回歸:回 1D 預測值 (n_classes=1)
            arr = out.squeeze(-1).cpu().numpy()
        else:
            arr = torch.softmax(out, dim=1).cpu().numpy()
    return arr


def _ensemble_replay(bundle: dict, X_raw: "np.ndarray", *, return_proba: bool = True,
                     on_progress=None, cancel_token=None):
    """重播 ensemble bundle:各 config 跑 5 fold 平均 → blender / stacker。

    分類 (return_proba=True):回 N×C 機率矩陣 (取 stacker 優先,失敗回退 blender)。
    回歸:回 N 維值 (已 inverse-transform 回原 y 尺度)。

    on_progress(event: dict) → None:每完成一個 config / 進到 blender/stacker 階段都會呼叫;
                                     event 結構見下面 dict literal,讓 endpoint 對應發 SSE event。
    cancel_token:threading.Event。is_set() 為 True 時下次 config 迭代會 raise InterruptedError,
                  讓上層 endpoint 把 CSV 結果丟掉直接回 cancel 狀態。
    """
    def _emit(event):
        if on_progress is not None:
            try: on_progress(event)
            except Exception: pass

    def _check_cancel():
        if cancel_token is not None and cancel_token.is_set():
            raise InterruptedError("使用者取消批次預測")

    n_classes = int(bundle.get("n_classes", 0))
    task_type = bundle.get("task_type", "classification")
    is_reg = task_type == "regression" or n_classes <= 1

    total_cfgs = sum(1 for c in bundle["configs"] if c.get("folds"))
    _emit({"phase": "start", "totalConfigs": total_cfgs,
           "nClasses": n_classes, "taskType": "regression" if is_reg else "classification",
           "nSamples": int(X_raw.shape[0])})

    all_test_preds = []
    done = 0
    for cfg_entry in bundle["configs"]:
        folds = cfg_entry.get("folds") or []
        if not folds:
            continue
        _check_cancel()
        tag = cfg_entry.get("tag", "?")
        _emit({"phase": "config_start", "tag": tag, "done": done, "total": total_cfgs,
               "nFolds": len(folds)})
        t0 = time.time()
        fold_preds = []
        for fold in folds:
            _check_cancel()
            fb = fold.get("fb")
            try:
                X_fb = fb.transform(X_raw) if fb is not None else X_raw
            except Exception as e:
                raise HTTPException(500, f"FeatureBuilder.transform 失敗 ({cfg_entry.get('tag')}): {e}")
            if "model" in fold:
                # tabular — 回歸用 predict,分類用 predict_proba
                pred = fold["model"].predict(X_fb) if is_reg else fold["model"].predict_proba(X_fb)
            else:
                # DL — _predict_dl_fold 內部依 n_classes 自動分流
                pred = _predict_dl_fold(fold, X_fb, max(1, n_classes))
            fold_preds.append(np.asarray(pred))
        all_test_preds.append(np.mean(fold_preds, axis=0))
        done += 1
        elapsed = round(time.time() - t0, 1)
        _emit({"phase": "config_done", "tag": tag, "done": done, "total": total_cfgs,
               "elapsedSec": elapsed})

    if not all_test_preds:
        raise HTTPException(500, "ensemble bundle 無 fold 可推論")

    _check_cancel()
    _emit({"phase": "ensemble_combine", "done": done, "total": total_cfgs})

    blender = bundle["blender"]
    stacker = bundle["stacker"]

    if is_reg:
        # 回歸:blender/stacker 一律走 .predict (沒有 _proba 概念)
        # stacker 通常較強 → 失敗回退 blender
        try:
            pred_scaled = stacker.predict(all_test_preds, X_orig=X_raw)
        except Exception:
            pred_scaled = blender.predict(all_test_preds)
        # RobustScaler inverse: y = y_scaled * IQR + median
        y_center = bundle.get("y_center")
        y_scale  = bundle.get("y_scale")
        arr = np.asarray(pred_scaled, dtype=float).ravel()
        if y_center is not None and y_scale is not None:
            arr = arr * float(y_scale) + float(y_center)
        return arr

    # 分類
    if return_proba:
        try:
            return stacker.predict_proba(all_test_preds, X_orig=X_raw)
        except Exception:
            try:
                return blender.predict_proba(all_test_preds)
            except Exception:
                return blender.predict(all_test_preds)
    # 純 predict 走 argmax
    try:
        proba = stacker.predict_proba(all_test_preds, X_orig=X_raw)
    except Exception:
        try:
            proba = blender.predict_proba(all_test_preds)
        except Exception:
            proba = blender.predict(all_test_preds)
    proba = np.asarray(proba)
    return proba.argmax(axis=1) if proba.ndim == 2 else proba.astype(int)


# ============================================================
# Autogluon helpers — unpack tar.gz bundle + load TabularPredictor
# ============================================================
_AG_UNPACK_CACHE: dict[str, str] = {}    # modelId → 已 unpack 的 dir (避免每次 predict 都重 unpack)

def _load_autogluon_predictor(model_id: str, bundle_bytes: bytes):
    """把 tar.gz bytes 解到 temp dir,load TabularPredictor 回來。

    Cache by model_id — 同一個 model 反覆 predict / SHAP 不用每次解壓。
    回傳 (predictor, predictor_dir)。
    """
    import tarfile as _tar
    import tempfile as _tf
    import io as _io

    if model_id in _AG_UNPACK_CACHE:
        cached_dir = _AG_UNPACK_CACHE[model_id]
        if os.path.isdir(cached_dir):
            try:
                from autogluon.tabular import TabularPredictor as _TP
                return _TP.load(cached_dir, require_version_match=False, require_py_version_match=False), cached_dir
            except Exception:
                pass   # cache 壞了,重 unpack
            _AG_UNPACK_CACHE.pop(model_id, None)

    # Unpack
    unpack_root = _tf.mkdtemp(prefix=f"ag_unpack_{model_id}_")
    with _tar.open(fileobj=_io.BytesIO(bundle_bytes), mode="r:gz") as tar:
        tar.extractall(unpack_root)
    predictor_dir = os.path.join(unpack_root, "autogluon")
    if not os.path.isdir(predictor_dir):
        # Fallback:找解出來的第一個 dir
        subdirs = [os.path.join(unpack_root, d) for d in os.listdir(unpack_root)
                   if os.path.isdir(os.path.join(unpack_root, d))]
        if subdirs:
            predictor_dir = subdirs[0]

    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        raise HTTPException(500, "autogluon 未安裝 — pip install autogluon.tabular")
    predictor = TabularPredictor.load(predictor_dir, require_version_match=False, require_py_version_match=False)
    _AG_UNPACK_CACHE[model_id] = predictor_dir
    return predictor, predictor_dir


async def _predict_batch_autogluon(model_id: str, entry: dict, bundle_meta: dict,
                                    file: UploadFile, sample_file: Optional[UploadFile]):
    """Autogluon batch predict — unpack tar.gz → TabularPredictor.predict()。"""
    import pandas as pd

    # estimator_pkl_bytes 在 guest path 是 unpickled 物件(bytes 不會自動 loads),
    # 在 DB path 是 _read_blob 回來的 bytes。但 autogluon bundle 是 tar.gz,不是 pickle,
    # 所以這裡要從 entry 直接拿 raw bytes — entry["estimator"] 是 None (沒 pickle),
    # 真實 bytes 要去 storage 撈。
    bundle_bytes = entry.get("_estimator_bytes")
    if not bundle_bytes:
        _est_status = entry.get("estimatorStatus", "empty")
        _msg = ("Autogluon model 的 file blob 已不存在 (model_blobs/ 被清掉),無法批次預測。請重訓。"
                if _est_status == "file_missing"
                else "Autogluon bundle bytes 不存在(舊紀錄或 save 失敗),請重訓。")
        raise HTTPException(410, _msg)

    try:
        predictor, _ = _load_autogluon_predictor(model_id, bundle_bytes)
    except HTTPException:
        raise
    except Exception as e:
        import traceback as _tb
        print(f"[autogluon_batch_predict] load predictor 失敗: {e}", flush=True)
        _tb.print_exc()
        raise HTTPException(500, f"Autogluon predictor load 失敗: {e}")

    # Parse CSV
    try:
        raw = await file.read()
        test_df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"CSV 解析失敗: {e}")

    # Drop target column if present (predict 不需要 label)
    target_col = bundle_meta.get("target") or bundle_meta.get("targetName")
    feat_df = test_df.drop(columns=[target_col], errors="ignore") if target_col else test_df

    try:
        preds = predictor.predict(feat_df)
    except Exception as e:
        import traceback as _tb
        _tb.print_exc()
        raise HTTPException(500, f"Autogluon predict 失敗: {e}")

    # 輸出 CSV — 跟 sklearn batch predict 一樣的格式
    sample_bytes = None
    if sample_file is not None:
        try:
            sample_bytes = await sample_file.read()
        except Exception:
            pass
    out_name = (file.filename.replace(".csv", "_predicted.csv")
                if file.filename else "predicted.csv")
    if sample_bytes:
        try:
            sample_df = pd.read_csv(io.BytesIO(sample_bytes))
            if len(sample_df.columns) >= 2:
                # 第一欄當 ID(取 test_df 的同名欄),第二欄放預測
                id_col = sample_df.columns[0]
                pred_col = sample_df.columns[1]
                out_df = pd.DataFrame({
                    id_col: test_df[id_col] if id_col in test_df.columns else range(len(preds)),
                    pred_col: preds.values if hasattr(preds, "values") else list(preds),
                })
                out_name = "submission.csv"
            else:
                out_df = test_df.copy()
                out_df["prediction"] = preds.values if hasattr(preds, "values") else list(preds)
        except Exception:
            out_df = test_df.copy()
            out_df["prediction"] = preds.values if hasattr(preds, "values") else list(preds)
    else:
        out_df = test_df.copy()
        out_df["prediction"] = preds.values if hasattr(preds, "values") else list(preds)

    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    from fastapi.responses import Response
    return Response(
        content=csv_bytes, media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )


async def _predict_batch_ensemble(bundle: dict, file: UploadFile,
                                  sample_file: Optional[UploadFile],
                                  preprocessor_id: Optional[str] = None,
                                  user=None, db=None):
    """ensemble bundle 的批次預測 — replay 訓練時的 fold-averaged → blender + stacker flow。

    若 ensemble 訓練時用 'preprocessed' source (preprocessor_id 有值),
    先把 raw test CSV 套同樣的 ColumnTransformer 才會跟訓練 schema 對得上。
    """
    import pandas as pd
    _ensure_pipeline_path_for_unpickle()

    # 1. 讀 CSV (原始欄位)
    try:
        raw = await file.read()
        feat_df_raw = pd.read_csv(io.BytesIO(raw))   # 原樣保留供 ID + 沒範本時 output
    except Exception as e:
        raise HTTPException(400, f"CSV 解析失敗: {e}")

    # 2. 若有 preprocessor → 先套 transform → 對齊 ensemble 訓練時看到的 schema
    target_col = bundle.get("target", "")
    if preprocessor_id and user is not None and db is not None:
        print(f"[batch_prep] preprocessor_id={preprocessor_id} test_rows={len(feat_df_raw)} cols={len(feat_df_raw.columns)}", flush=True)
        try:
            pp_entry = storage.get_preprocessor(preprocessor_id, user, db)
            preprocessor = pp_entry["preprocessor"]
            pp_target = pp_entry.get("target", "")
            print(f"[batch_prep] preprocessor type={type(preprocessor).__name__}, pp_target='{pp_target}'", flush=True)

            # 對齊裝甲:test.csv 缺的欄補 NaN,多的欄丟掉;sklearn 預設 transform 對欄序敏感
            X_in = feat_df_raw.drop(columns=[pp_target], errors="ignore").copy()
            # 抓 preprocessor fit 時看到的欄(若拿不到就跳過對齊,直接 transform 賭一賭)
            fit_cols = None
            if hasattr(preprocessor, "feature_names_in_"):
                fit_cols = list(preprocessor.feature_names_in_)
            elif hasattr(preprocessor, "_columns"):
                fit_cols = list(preprocessor._columns)
            if fit_cols:
                missing_in_test = [c for c in fit_cols if c not in X_in.columns]
                extra_in_test = [c for c in X_in.columns if c not in fit_cols]
                if missing_in_test:
                    print(f"[batch_prep] test 缺 {len(missing_in_test)} 個 train 看過的欄,補 NaN: {missing_in_test[:5]}...", flush=True)
                    for c in missing_in_test:
                        X_in[c] = np.nan
                if extra_in_test:
                    print(f"[batch_prep] test 多 {len(extra_in_test)} 個 train 沒看過的欄,丟掉: {extra_in_test[:5]}...", flush=True)
                # 重排成 train 看到的順序
                X_in = X_in[fit_cols]

            print(f"[batch_prep] X_in shape: {X_in.shape}, 正在 transform...", flush=True)
            X_transformed = preprocessor.transform(X_in)
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()
            print(f"[batch_prep] X_transformed shape: {X_transformed.shape}", flush=True)

            try:
                _raw_names = (list(preprocessor.get_feature_names_out())
                              if hasattr(preprocessor, "get_feature_names_out")
                              else [f"f{i}" for i in range(X_transformed.shape[1])])
                # daniel 訓練時把 'pipeline_name__feature' split('__')[-1] 簡化成 'feature',
                # bundle.feature_names_raw 存的也是簡名 — 推論時必須做一樣的清理
                tx_names = [str(n).split("__")[-1] for n in _raw_names]
            except Exception:
                tx_names = [f"f{i}" for i in range(X_transformed.shape[1])]
            if len(tx_names) != X_transformed.shape[1]:
                tx_names = [f"f{i}" for i in range(X_transformed.shape[1])]
            feat_df = pd.DataFrame(X_transformed, columns=tx_names)
            print(f"[batch_prep] feat_df 完成,cols 範例: {list(feat_df.columns[:5])}", flush=True)
        except HTTPException:
            raise
        except Exception as e:
            # 完整 traceback 印到 server console,讓使用者抓得到根因;
            # 對前端只回精簡訊息,不暴露內部結構
            import traceback as _tb
            print(f"[batch_prep] preprocessor.transform 失敗:", flush=True)
            _tb.print_exc()
            raise HTTPException(400,
                f"套用預處理失敗 — {type(e).__name__}: {str(e)[:300]} "
                f"(完整 traceback 請看 server console)")
    else:
        feat_df = feat_df_raw

    # 3. 抽出 ensemble 需要的特徵 (對齊訓練時 _prepare_xy 的邏輯)
    cols_required = bundle.get("feature_names_raw", [])
    missing = [c for c in cols_required if c not in feat_df.columns]
    if missing:
        raise HTTPException(400,
            f"CSV 缺少 {len(missing)} 個模型需要的特徵欄位: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    X_raw = feat_df[cols_required].fillna(0).values.astype(np.float32)

    # 4. 重播 ensemble (回歸 vs 分類分流由 _ensemble_replay 內部處理)
    task_type = bundle.get("task_type", "classification")
    is_reg = task_type == "regression" or int(bundle.get("n_classes", 0)) <= 1

    if is_reg:
        # 回歸:回連續值,直接寫進 CSV
        preds = _ensemble_replay(bundle, X_raw, return_proba=False)
        preds = np.asarray(preds, dtype=float).ravel()
    else:
        # 分類:回 argmax 後再 inverse_transform 回原 label
        idx_arr = _ensemble_replay(bundle, X_raw, return_proba=False)
        idx_arr = np.asarray(idx_arr).astype(int).ravel()
        le = bundle.get("label_encoder")
        preds = le.inverse_transform(idx_arr) if le is not None else idx_arr

    # 7. submission / predicted CSV — ID 從 raw df 抓 (transformed df 沒這欄)
    if sample_file is not None:
        try:
            sample_raw = await sample_file.read()
            sub_df = pd.read_csv(io.BytesIO(sample_raw))
        except Exception as e:
            raise HTTPException(400, f"範本 submission 解析失敗: {e}")
        if len(sub_df.columns) < 2:
            raise HTTPException(400, "範本 submission 至少需要 2 欄 (ID 欄 + 預測欄)")
        id_col   = sub_df.columns[0]
        pred_col = sub_df.columns[1]
        if id_col not in feat_df_raw.columns:
            raise HTTPException(400, f"test.csv 缺少範本要求的 ID 欄位「{id_col}」")
        out_df = pd.DataFrame({id_col: feat_df_raw[id_col], pred_col: preds})
        out_name = "submission.csv"
    else:
        # 沒範本 → 在原始 raw df 加 prediction 欄,使用者一目了然
        out_df = feat_df_raw.copy()
        out_df["prediction"] = preds
        out_name = (file.filename.replace(".csv", "_predicted.csv")
                    if file.filename else "_predicted.csv")

    csv_bytes = out_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )


# ── 共用:把 daniel ensemble 的 batch predict 拆成「prep」+「replay」+「write CSV」三段 ───
async def _ensemble_batch_prep(bundle, file: UploadFile, sample_file,
                               preprocessor_id, user, db):
    """讀 CSV → (optional) 套 preprocessor → 對齊 feature_names_raw → X_raw + 一些 meta。"""
    import pandas as pd
    _ensure_pipeline_path_for_unpickle()
    try:
        raw = await file.read()
        feat_df_raw = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"CSV 解析失敗: {e}")

    if preprocessor_id and user is not None and db is not None:
        print(f"[batch_prep] preprocessor_id={preprocessor_id} test_rows={len(feat_df_raw)} cols={len(feat_df_raw.columns)}", flush=True)
        try:
            pp_entry = storage.get_preprocessor(preprocessor_id, user, db)
            preprocessor = pp_entry["preprocessor"]
            pp_target = pp_entry.get("target", "")

            # 優先用 daniel 的 preprocess_for_inference — 它有「特徵對齊裝甲」(missing → NaN,
            # 多餘 → drop,順序重排),比直接 .transform() 對未知/缺欄健壯很多。
            # cols_required 是訓練時 fit 出來的特徵名,用來給 inference 函式知道目標 schema。
            cols_required = bundle.get("feature_names_raw", [])
            X_in_raw = feat_df_raw.drop(columns=[pp_target], errors="ignore") if pp_target else feat_df_raw

            try:
                from api.preprocess import preprocess_for_inference
                print(f"[batch_prep] 嘗試 daniel preprocess_for_inference (對齊到 {len(cols_required)} 個目標欄)", flush=True)
                feat_df = preprocess_for_inference(
                    X_in_raw,                        # data_source
                    preprocessor,                    # fitted_preprocessor (tree-track)
                    training_features=cols_required, # 對齊目標
                )
                print(f"[batch_prep] preprocess_for_inference OK,shape={feat_df.shape}", flush=True)
            except Exception as _pf_e:
                # Fallback:走老路 .transform(),配特徵對齊裝甲手動補
                print(f"[batch_prep] preprocess_for_inference 不可用 / 失敗,fallback 直接 .transform(): {_pf_e}", flush=True)
                X_in = X_in_raw.copy()
                fit_cols = None
                if hasattr(preprocessor, "feature_names_in_"):
                    fit_cols = list(preprocessor.feature_names_in_)
                elif hasattr(preprocessor, "_columns"):
                    fit_cols = list(preprocessor._columns)
                if fit_cols:
                    missing_in_test = [c for c in fit_cols if c not in X_in.columns]
                    if missing_in_test:
                        print(f"[batch_prep] 對齊裝甲補 NaN ({len(missing_in_test)} 欄): {missing_in_test[:5]}...", flush=True)
                        for c in missing_in_test:
                            X_in[c] = np.nan
                    X_in = X_in[fit_cols]   # 重排 + drop extras
                X_transformed = preprocessor.transform(X_in)
                if hasattr(X_transformed, "toarray"):
                    X_transformed = X_transformed.toarray()
                try:
                    _raw_names = (list(preprocessor.get_feature_names_out())
                                  if hasattr(preprocessor, "get_feature_names_out")
                                  else [f"f{i}" for i in range(X_transformed.shape[1])])
                    tx_names = [str(n).split("__")[-1] for n in _raw_names]
                except Exception:
                    tx_names = [f"f{i}" for i in range(X_transformed.shape[1])]
                if len(tx_names) != X_transformed.shape[1]:
                    tx_names = [f"f{i}" for i in range(X_transformed.shape[1])]
                feat_df = pd.DataFrame(X_transformed, columns=tx_names)
                print(f"[batch_prep] fallback transform OK,shape={feat_df.shape}", flush=True)
        except HTTPException:
            raise
        except Exception as e:
            import traceback as _tb
            print(f"[batch_prep] preprocess_for_inference + fallback 都失敗:", flush=True)
            _tb.print_exc()
            raise HTTPException(400,
                f"套用預處理失敗 — {type(e).__name__}: {str(e)[:300]} "
                f"(完整 traceback 請看 server console)")
    else:
        feat_df = feat_df_raw

    cols_required = bundle.get("feature_names_raw", [])
    missing = [c for c in cols_required if c not in feat_df.columns]
    if missing:
        raise HTTPException(400,
            f"CSV 缺少 {len(missing)} 個模型需要的特徵欄位: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    X_raw = feat_df[cols_required].fillna(0).values.astype(np.float32)

    sample_bytes = None
    if sample_file is not None:
        try:
            sample_bytes = await sample_file.read()
        except Exception as e:
            raise HTTPException(400, f"範本 submission 解析失敗: {e}")

    return feat_df_raw, X_raw, sample_bytes


def _ensemble_finalize_csv(bundle, feat_df_raw, preds, sample_bytes, out_filename):
    """把 preds 寫成 CSV bytes。回 (csv_bytes, out_name)。"""
    import pandas as pd
    le = bundle.get("label_encoder")
    is_reg = bundle.get("task_type") == "regression" or int(bundle.get("n_classes", 0)) <= 1

    if not is_reg and le is not None and not isinstance(preds[0] if len(preds) else None, (str, bytes)):
        # 已經是 idx 數字 → 解回原 label
        try:
            preds = le.inverse_transform(np.asarray(preds).astype(int))
        except Exception:
            pass

    if sample_bytes is not None:
        sub_df = pd.read_csv(io.BytesIO(sample_bytes))
        if len(sub_df.columns) < 2:
            raise HTTPException(400, "範本 submission 至少需要 2 欄 (ID 欄 + 預測欄)")
        id_col   = sub_df.columns[0]
        pred_col = sub_df.columns[1]
        if id_col not in feat_df_raw.columns:
            raise HTTPException(400, f"test.csv 缺少範本要求的 ID 欄位「{id_col}」")
        out_df = pd.DataFrame({id_col: feat_df_raw[id_col], pred_col: preds})
        out_name = "submission.csv"
    else:
        out_df = feat_df_raw.copy()
        out_df["prediction"] = preds
        out_name = out_filename or "_predicted.csv"

    return out_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), out_name


# ── 新 SSE endpoint:邊跑邊 yield progress + 最終 base64 CSV + 支援 cancel ──────────
@app.post("/api/predict/batch/stream")
async def predict_batch_stream_endpoint(
    modelId: str = Form(...),
    file: UploadFile = File(...),
    sampleFile: Optional[UploadFile] = File(None),
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
):
    """SSE 版批次預測 — 邊推進度邊跑 ensemble replay。

    事件流 (text/event-stream):
      event: progress  data: {"phase": "config_start", "tag": "lgbm_raw_c0", "done": 0, "total": 5}
      event: progress  data: {"phase": "config_done",  "tag": "lgbm_raw_c0", "done": 1, "total": 5, "elapsedSec": 8.3}
      ...
      event: done      data: {"filename": "submission.csv", "csvBase64": "..."}
    瀏覽器斷線 (AbortController) → cancel_token.set() → 下次 config 迭代拋 InterruptedError → 收尾 cancelled 事件。
    """
    import threading, base64, asyncio
    _ensure_pipeline_path_for_unpickle()
    entry = storage.get_model(modelId, user, db)
    estimator       = entry["estimator"]
    preprocessor_id = entry.get("preprocessorId")
    bundle_meta     = entry.get("bundle") or {}

    if bundle_meta.get("type") != "daniel_pipeline_ensemble":
        # sklearn 等小模型:SSE 沒意義,直接回退到一次性 endpoint
        raise HTTPException(400, "SSE batch predict 只支援 daniel ensemble — 一般模型走 /api/predict/batch")
    if not (isinstance(estimator, dict) and estimator.get("version") == 1 and "configs" in estimator):
        _est_status = entry.get("estimatorStatus", "empty")
        _msg = ("此 ensemble 模型的 file blob 已不存在 (model_blobs/ 被清掉或換機器了)"
                if _est_status == "file_missing"
                else "此 ensemble 模型在 DB 沒有 estimator (舊 placeholder 或 pickle 失敗)")
        raise HTTPException(status_code=410, detail=_msg + ",無法批次預測。請重訓。")

    bundle = estimator
    feat_df_raw, X_raw, sample_bytes = await _ensemble_batch_prep(
        bundle, file, sampleFile, preprocessor_id, user, db,
    )

    q: queue.Queue = queue.Queue()
    cancel_token = threading.Event()
    result_box: dict[str, Any] = {"preds": None, "error": None}

    def worker():
        # 把每個 emit 同時吐到 stdout (flush=True),萬一 C 層 segfault 沒留 traceback,
        # 至少從 terminal 看得到「最後印的是哪個 fold/config」,就能定位崩在哪。
        def _on_progress(ev):
            try:
                _phase = ev.get("phase", "?")
                _tag = ev.get("tag", "")
                _done = ev.get("done", "")
                _total = ev.get("total", "")
                print(f"[predict_worker] phase={_phase} tag={_tag} {_done}/{_total}", flush=True)
            except Exception:
                pass
            q.put({"type": "progress", **ev})

        try:
            print(f"[predict_worker] start replay  bundle_configs={len(bundle.get('configs', []))} X_shape={X_raw.shape}", flush=True)
            preds = _ensemble_replay(
                bundle, X_raw, return_proba=False,
                on_progress=_on_progress,
                cancel_token=cancel_token,
            )
            print(f"[predict_worker] replay 完成 preds_len={len(preds) if preds is not None else 0}", flush=True)
            result_box["preds"] = np.asarray(preds).ravel()
        except InterruptedError as ie:
            print(f"[predict_worker] 取消: {ie}", flush=True)
            result_box["error"] = ("cancelled", str(ie))
        except HTTPException as he:
            print(f"[predict_worker] HTTPException: {he.status_code} {he.detail}", flush=True)
            result_box["error"] = ("http", f"{he.status_code}: {he.detail}")
        except BaseException as e:
            # 連 SystemExit / KeyboardInterrupt 都接,什麼例外都印 — 比 Exception 還寬
            import traceback
            print(f"[predict_worker] 💥 {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            result_box["error"] = ("error", f"{type(e).__name__}: {str(e)[:300]}")
        finally:
            print("[predict_worker] worker 結束,送 sentinel", flush=True)
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    # SSE heartbeat sentinel — q.get 超時時 _q_get_with_timeout 回傳這個,
    # event_stream 看到就送 `: heartbeat\n\n` comment 維持連線。
    # 不能用 None,因為 None 是 worker 結束的訊號。
    _HEARTBEAT = object()

    def _q_get_with_timeout(_q, _timeout: float):
        try:
            return _q.get(timeout=_timeout)
        except queue.Empty:
            return _HEARTBEAT

    async def event_stream():
        # 兩層 try:外層只擋 GeneratorExit/CancelledError(瀏覽器斷線觸發 cancel_token),
        # 內層擋一切 — 真的炸時也吐 'error' event 不讓 ASGI reset 連線(瀏覽器看到網路錯誤).
        try:
            try:
                loop = asyncio.get_event_loop()
                # 立刻送一個 heartbeat,讓 client 馬上收到 first byte,避免 fetch 在 headers 收完
                # 卻沒 body 的情況下被某些 proxy / 瀏覽器 timeout
                # 立刻送 first byte (真 event 不是 comment) — 確保 client / proxy 都認得「有資料來了」
                yield f"event: progress\ndata: {json.dumps({'phase': 'connected'}, ensure_ascii=False)}\n\n"
                while True:
                    # 阻塞最多 1 秒等 worker 推 event;沒事就送 heartbeat 保活
                    # 解決 uvicorn keep-alive 預設 5s timeout + 某些 fold predict 跑 > 5s 時連線被砍的問題
                    ev = await loop.run_in_executor(None, _q_get_with_timeout, q, 1.0)
                    if ev is _HEARTBEAT:
                        # SSE comment (`:` 開頭) 有些 proxy / 瀏覽器不認當「有資料」→ 改送真 progress event。
                        # client 看到 phase=heartbeat 會在 progress handler 走 else 分支(不更新 UI 也不會 error)
                        yield f"event: progress\ndata: {json.dumps({'phase': 'heartbeat'}, ensure_ascii=False)}\n\n"
                        continue
                    if ev is None:
                        break
                    yield f"event: progress\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"

                if result_box["error"]:
                    kind, msg = result_box["error"]
                    yield f"event: error\ndata: {json.dumps({'kind': kind, 'msg': msg}, ensure_ascii=False)}\n\n"
                    return

                # 完成 → 寫 CSV + base64 一次性傳給前端
                preds = result_box["preds"]
                out_name_default = (file.filename.replace(".csv", "_predicted.csv")
                                    if file.filename else "_predicted.csv")
                try:
                    csv_bytes, out_name = _ensemble_finalize_csv(
                        bundle, feat_df_raw, preds, sample_bytes, out_name_default,
                    )
                except HTTPException as he:
                    yield f"event: error\ndata: {json.dumps({'kind': 'http', 'msg': f'{he.status_code}: {he.detail}'}, ensure_ascii=False)}\n\n"
                    return
                except Exception as e:
                    import traceback as _tb
                    print("[SSE batch predict] _ensemble_finalize_csv 失敗:", flush=True)
                    _tb.print_exc()
                    yield f"event: error\ndata: {json.dumps({'kind': 'finalize', 'msg': f'{type(e).__name__}: {str(e)[:300]}'}, ensure_ascii=False)}\n\n"
                    return
                payload = {
                    "filename": out_name,
                    "csvBase64": base64.b64encode(csv_bytes).decode("ascii"),
                    "rowCount": int(len(preds)),
                }
                yield f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            except Exception as e:
                # event_stream 本身的 bug 也吐成 SSE error,不讓 ASGI 重置連線
                import traceback as _tb
                print("[SSE batch predict] event_stream 內部 bug:", flush=True)
                _tb.print_exc()
                try:
                    yield f"event: error\ndata: {json.dumps({'kind': 'stream', 'msg': f'{type(e).__name__}: {str(e)[:300]}'}, ensure_ascii=False)}\n\n"
                except Exception:
                    pass
        except (GeneratorExit, asyncio.CancelledError):
            cancel_token.set()
            raise

    # 【SSE 必備 headers】缺這幾個 client 會把 stream 當 normal HTTP 處理,
    # 讀完第一個 chunk 就斷線 → 後端 worker 繼續跑變孤兒,前端看 ERR_CONNECTION_RESET
    #   - Cache-Control: no-cache        proxies / browsers 不要 cache
    #   - X-Accel-Buffering: no          nginx 等 proxy 不要 buffer (即使本機沒 nginx,
    #                                     某些 antivirus / 瀏覽器擴充也會看這個 header)
    #   - Connection: keep-alive         明確要求保持連線
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/predict/batch")
async def predict_batch_endpoint(
    modelId: str = Form(...),
    file: UploadFile = File(...),
    sampleFile: Optional[UploadFile] = File(None),
    user = Depends(get_current_user),
    db: DbSession = Depends(get_db),
):
    import pandas as pd
    _ensure_pipeline_path_for_unpickle()
    entry = storage.get_model(modelId, user, db)
    estimator       = entry["estimator"]
    scaler          = entry["scaler"]
    feature_names   = entry["featureNames"]
    preprocessor_id = entry.get("preprocessorId")
    bundle_meta     = entry.get("bundle") or {}

    # Daniel ensemble:bundle.type 是真相 (estimator 可能因 pickle 失敗沒存進 DB)
    if bundle_meta.get("type") == "daniel_pipeline_ensemble":
        if not (isinstance(estimator, dict) and estimator.get("version") == 1 and "configs" in estimator):
            _est_status = entry.get("estimatorStatus", "empty")
            if _est_status == "file_missing":
                _msg = "此 ensemble 模型的 file blob 已不存在 (model_blobs/ 被清掉或換機器了),無法批次預測。請重訓。"
            else:
                _msg = "此 ensemble 模型在 DB 沒有 estimator (舊 placeholder 或 pickle 失敗),無法批次預測。請重訓。"
            raise HTTPException(status_code=410, detail=_msg)
        # preprocessor_id / user / db 傳下去 — preprocessed source 訓的 ensemble
        # 需要先把 raw test.csv 套同樣的 ColumnTransformer
        return await _predict_batch_ensemble(estimator, file, sampleFile,
                                             preprocessor_id=preprocessor_id,
                                             user=user, db=db)

    # Autogluon model:estimator 是 tar.gz bytes (entire autogluon predictor dir)
    if bundle_meta.get("type") == "autogluon_model":
        return await _predict_batch_autogluon(modelId, entry, bundle_meta, file, sampleFile)

    # --- parse uploaded CSV ---
    try:
        raw = await file.read()
        feat_df_raw = pd.read_csv(io.BytesIO(raw))   # \u4fdd\u7559\u539f\u59cb\u6b04\u4f4d\u4f9b ID + \u6c92\u7bc4\u672c\u6642 output
    except Exception as e:
        raise HTTPException(400, f"CSV \u89e3\u6790\u5931\u6557: {e}")

    # --- apply preprocessor if any (raw \u2192 transformed) ---
    # \u9810\u8655\u7406\u6a21\u578b\u8a13\u7df4\u6642\u7684 feature_names \u662f transformed \u5f8c\u7684\u540d\u7a31
    # (\u4f8b:cat_pipeline__workclass_Private)\u3002\u76f4\u63a5\u62ff\u539f\u59cb test.csv \u7684\u6b04\u4f4d\u9078\u4e0d\u5230,
    # \u5fc5\u9808\u7528 train \u6642\u7684 preprocessor \u628a raw \u2192 transformed \u518d select\u3002
    if preprocessor_id:
        try:
            pp_entry = storage.get_preprocessor(preprocessor_id, user, db)
            preprocessor = pp_entry["preprocessor"]
            pp_target = pp_entry.get("target", "")
            # \u4e1f\u6389 target \u6b04\u4f4d (test.csv \u53ef\u80fd\u6709\u4e5f\u53ef\u80fd\u6c92\u6709)
            X_in = feat_df_raw.drop(columns=[pp_target], errors="ignore")
            X_transformed = preprocessor.transform(X_in)
            # sparse \u2192 dense
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()
            # \u9084\u539f\u6210 DataFrame,column \u5c0d\u9f4a preprocessor \u7684 transformed names
            try:
                _raw_names = (list(preprocessor.get_feature_names_out())
                              if hasattr(preprocessor, "get_feature_names_out")
                              else [f"f{i}" for i in range(X_transformed.shape[1])])
                # daniel 訓練時把 'pipeline_name__feature' split('__')[-1] 簡化成 'feature',
                # bundle.feature_names_raw 存的也是簡名 — 推論時必須做一樣的清理,
                # 否則 cols_required 對不上 → 觸發「缺 N 個欄」假錯誤
                tx_names = [str(n).split("__")[-1] for n in _raw_names]
            except Exception:
                tx_names = [f"f{i}" for i in range(X_transformed.shape[1])]
            if len(tx_names) != X_transformed.shape[1]:
                tx_names = [f"f{i}" for i in range(X_transformed.shape[1])]
            feat_df = pd.DataFrame(X_transformed, columns=tx_names)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400,
                f"\u5957\u7528\u9810\u8655\u7406\u5931\u6557 \u2014 test.csv \u7684\u6b04\u4f4d\u9700\u8207\u8a13\u7df4\u6642\u7684\u300c\u539f\u59cb\u6b04\u4f4d\u300d\u4e00\u81f4: {e}")
    else:
        feat_df = feat_df_raw   # raw \u6a21\u578b\u76f4\u63a5\u7528\u539f\u59cb df

    # --- select & order features ---
    missing = [f for f in feature_names if f not in feat_df.columns]
    if missing:
        raise HTTPException(400,
            f"CSV \u7f3a\u5c11 {len(missing)} \u500b\u6a21\u578b\u9700\u8981\u7684\u7279\u5fb5\u6b04\u4f4d (\u524d 10 \u500b): {missing[:10]}")

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
        # ID \u5f9e\u539f\u59cb raw df \u6293 (transformed df \u6c92\u9019\u6b04)
        if id_col not in feat_df_raw.columns:
            raise HTTPException(400,
                f"test.csv \u7f3a\u5c11\u7bc4\u672c\u8981\u6c42\u7684 ID \u6b04\u4f4d\u300c{id_col}\u300d")
        out_df = pd.DataFrame({id_col: feat_df_raw[id_col], pred_col: preds})
        out_name = "submission.csv"
    else:
        # \u6c92\u7bc4\u672c \u2192 \u5728\u539f\u59cb raw df \u5f8c\u9762\u52a0 prediction \u6b04,\u4f7f\u7528\u8005\u4e00\u76ee\u4e86\u7136
        out_df = feat_df_raw.copy()
        out_df["prediction"] = preds
        out_name = file.filename.replace(".csv", "_predicted.csv") if file.filename else "_predicted.csv"

    csv_bytes = out_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )
