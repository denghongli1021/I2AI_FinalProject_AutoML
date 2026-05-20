# -*- coding: utf-8 -*-
"""
storage.py — 統一 storage 層,把 guest (in-memory dict) 跟 authed user (DB) 兩條路徑藏在後面。

對 endpoint 來說:
    bundle = storage.get_dataset(dataset_id, user, db)   # 不管 user 是不是 None,都 work
    storage.save_dataset(dataset_id, payload, user, db)

dispatch 規則:
    user is None  → 走 api/store.py 的 in-memory dict (重啟掉,符合「guest 不持久」)
    user 是物件   → 走 DB (重啟保留)

回傳格式 (對外):
    dataset bundle = {
        "id": str, "fileName": str, "df": DataFrame (lazy hydrate),
        "rowCount": int, "colCount": int, "headers": list,
        "response": dict (analysis + extras),
        "loadedAt": float, "createdAt": datetime,
    }
    preprocessor bundle = {
        "id": str, "preprocessor": ColumnTransformer, "target": str,
        "datasetId": str, "featureNames": list,
        "X_train", "X_test", "y_train", "y_test",
    }
    model bundle = {
        "id": str, "bundle": dict, "estimator": ..., "scaler": ...,
        "featureNames": list, "X_test_df": DataFrame, "preprocessorId": str|None,
        "hyperparameters": dict,
    }
"""
from __future__ import annotations

import io
import json
import pickle
import time
import uuid
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session

from api.auth.db import Dataset as DbDataset
from api.auth.db import Model as DbModel
from api.auth.db import PredictionArtifact as DbPredictionArtifact
from api.auth.db import Preprocessor as DbPreprocessor
from api.auth.db import TrainingRun as DbTrainingRun

# Guest (未登入) 的 in-memory store — 重啟就消失
from api.store import DATASETS, MODELS, PREPROCESSORS, get_owned, list_owned, stamp


# ============================================================
# Helpers
# ============================================================
def _is_authed(user) -> bool:
    return user is not None and getattr(user, "id", None) is not None


def _uid(user) -> Optional[int]:
    return user.id if _is_authed(user) else None


def _hydrate_dataset_df(ds: DbDataset) -> pd.DataFrame:
    """從 csv_blob 還原 DataFrame。每次叫都 parse 一次 (~10-50ms)。"""
    if ds.csv_blob is None:
        raise HTTPException(status_code=500, detail=f"dataset {ds.id} 沒有 csv_blob")
    return pd.read_csv(io.BytesIO(ds.csv_blob))


def _db_dataset_to_bundle(ds: DbDataset, include_df: bool = True) -> dict:
    """DB row → bundle dict (跟 in-memory 那邊長相一致,讓上層程式不用改 if/else)。"""
    response = json.loads(ds.analysis_json) if ds.analysis_json else {}
    if ds.extras_json:
        extras = json.loads(ds.extras_json)
        response.update(extras)
    bundle = {
        "id": ds.id,
        "fileName": ds.file_name,
        "rowCount": ds.row_count,
        "colCount": ds.col_count,
        "headers": json.loads(ds.headers_json) if ds.headers_json else [],
        "loadedAt": ds.created_at.timestamp() if ds.created_at else time.time(),
        "response": response,
    }
    if include_df:
        bundle["df"] = _hydrate_dataset_df(ds)
    return bundle


# ============================================================
# DATASET
# ============================================================
def save_dataset(
    df: pd.DataFrame,
    file_name: str,
    csv_bytes: bytes,
    response: dict,
    user,
    db: Session,
) -> str:
    """寫入新 dataset,回傳 dataset_id。"""
    dataset_id = str(uuid.uuid4())

    if not _is_authed(user):
        DATASETS[dataset_id] = stamp({
            "df": df,
            "fileName": file_name,
            "loadedAt": time.time(),
            "response": response,
        }, user)
        return dataset_id

    # DB path — 拆 analysis 跟 extras (extras 是非預期的大 key 例如 correlation)
    main_keys = {"rowCount", "colCount", "headers", "analysis", "preview", "stats"}
    analysis = {k: v for k, v in response.items() if k in main_keys}
    extras = {k: v for k, v in response.items() if k not in main_keys}

    db.add(DbDataset(
        id=dataset_id,
        user_id=user.id,
        file_name=file_name,
        row_count=response.get("rowCount", len(df)),
        col_count=response.get("colCount", df.shape[1]),
        headers_json=json.dumps(response.get("headers", list(df.columns)), ensure_ascii=False),
        csv_blob=csv_bytes,
        analysis_json=json.dumps(analysis, ensure_ascii=False, default=str),
        extras_json=json.dumps(extras, ensure_ascii=False, default=str) if extras else None,
    ))
    db.commit()
    return dataset_id


def get_dataset(dataset_id: str, user, db: Session, include_df: bool = True) -> dict:
    """讀單一 dataset,回 bundle。"""
    if not _is_authed(user):
        return get_owned(DATASETS, dataset_id, user, "dataset")
    ds = db.query(DbDataset).filter_by(id=dataset_id, user_id=user.id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="dataset 不存在")
    return _db_dataset_to_bundle(ds, include_df=include_df)


def list_datasets(user, db: Session) -> list[dict]:
    """列出當前 user 所有 dataset (metadata only,不含 df)。"""
    if not _is_authed(user):
        items = []
        for ds_id, entry in list_owned(DATASETS, user):
            resp = entry.get("response") or {}
            items.append({
                "id": ds_id,
                "fileName": entry.get("fileName"),
                "loadedAt": entry.get("loadedAt"),
                "rowCount": resp.get("rowCount", 0),
                "colCount": resp.get("colCount", 0),
                "headers": resp.get("headers", []),
            })
        items.sort(key=lambda x: x.get("loadedAt") or 0, reverse=True)
        return items

    rows = (db.query(DbDataset)
            .filter_by(user_id=user.id)
            .order_by(DbDataset.created_at.desc())
            .all())
    return [{
        "id": r.id,
        "fileName": r.file_name,
        "loadedAt": r.created_at.timestamp() if r.created_at else None,
        "rowCount": r.row_count,
        "colCount": r.col_count,
        "headers": json.loads(r.headers_json) if r.headers_json else [],
    } for r in rows]


def delete_dataset(dataset_id: str, user, db: Session) -> dict:
    """級聯刪除 dataset + 衍生的 preprocessor + model。"""
    if not _is_authed(user):
        # 確認存在
        get_owned(DATASETS, dataset_id, user, "dataset")
        # 找衍生
        pp_ids = [pid for pid, p in PREPROCESSORS.items()
                  if p.get("datasetId") == dataset_id and p.get("_owner") == stamp({}, user)["_owner"]]
        model_ids = [mid for mid, m in MODELS.items()
                     if m.get("preprocessorId") in pp_ids and m.get("_owner") == stamp({}, user)["_owner"]]
        DATASETS.pop(dataset_id, None)
        for pid in pp_ids:    PREPROCESSORS.pop(pid, None)
        for mid in model_ids: MODELS.pop(mid, None)
        return {"ok": True, "removed": {"dataset": dataset_id,
                                         "preprocessors": pp_ids, "models": model_ids}}

    # DB path — 先 check ownership (只有 dataset 主人才能刪)
    ds = db.query(DbDataset).filter_by(id=dataset_id, user_id=user.id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="dataset 不存在")
    # 子表 (preprocessors / models) 只用 dataset_id 過濾,**不加 user_id**:
    # 理論上同個 dataset_id 的子記錄一定屬於同個 user (因為 dataset_id 唯一),
    # 加 user_id 反而會漏掉早期 orphan 記錄,後續 FK 會擋住 dataset 刪除。
    pp_rows = db.query(DbPreprocessor).filter_by(dataset_id=dataset_id).all()
    pp_ids = [p.id for p in pp_rows]
    model_rows = (db.query(DbModel).filter(DbModel.preprocessor_id.in_(pp_ids)).all()
                  if pp_ids else [])
    # 也撈直接以 dataset_id 連的 model (raw source 訓練的)
    raw_model_rows = db.query(DbModel).filter_by(dataset_id=dataset_id).all()
    seen = {m.id for m in model_rows}
    for m in raw_model_rows:
        if m.id not in seen:
            model_rows.append(m); seen.add(m.id)
    model_ids = [m.id for m in model_rows]

    # 刪除順序:孩子 → 父。每批 flush 一次確保 FK 不會在 commit 階段炸。
    for r in model_rows: db.delete(r)
    db.flush()
    for r in pp_rows:    db.delete(r)
    db.flush()
    db.delete(ds)
    db.commit()
    return {"ok": True, "removed": {"dataset": dataset_id,
                                     "preprocessors": pp_ids, "models": model_ids}}


# ============================================================
# PREPROCESSOR
# ============================================================
def save_preprocessor(
    preprocessor,
    target: str,
    dataset_id: str,
    feature_names: list,
    X_train, X_test, y_train, y_test,
    user,
    db: Session,
    test_size: float = 0.2,
) -> str:
    pp_id = f"pp_{uuid.uuid4().hex[:8]}"
    if not _is_authed(user):
        PREPROCESSORS[pp_id] = stamp({
            "preprocessor": preprocessor,
            "target": target,
            "datasetId": dataset_id,
            "featureNames": feature_names,
            "createdAt": time.time(),
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
        }, user)
        return pp_id

    # DB path — preprocessor + train/test 全部 pickle 進 LargeBinary
    train_test_blob = pickle.dumps((X_train, X_test, y_train, y_test), protocol=pickle.HIGHEST_PROTOCOL)
    db.add(DbPreprocessor(
        id=pp_id,
        user_id=user.id,
        dataset_id=dataset_id,
        target=target,
        test_size=test_size,
        feature_names_json=json.dumps(feature_names, ensure_ascii=False),
        preprocessor_pkl=pickle.dumps(preprocessor, protocol=pickle.HIGHEST_PROTOCOL),
        train_test_pkl=train_test_blob,
    ))
    db.commit()
    return pp_id


def get_preprocessor(preprocessor_id: str, user, db: Session) -> dict:
    if not _is_authed(user):
        return get_owned(PREPROCESSORS, preprocessor_id, user, "preprocessor")
    pp = db.query(DbPreprocessor).filter_by(id=preprocessor_id, user_id=user.id).first()
    if pp is None:
        raise HTTPException(status_code=404, detail="preprocessor 不存在")
    X_train, X_test, y_train, y_test = pickle.loads(pp.train_test_pkl)
    return {
        "id": pp.id,
        "preprocessor": pickle.loads(pp.preprocessor_pkl),
        "target": pp.target,
        "datasetId": pp.dataset_id,
        "featureNames": json.loads(pp.feature_names_json) if pp.feature_names_json else [],
        "createdAt": pp.created_at.timestamp() if pp.created_at else None,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
    }


def list_preprocessors(user, db: Session) -> list[dict]:
    if not _is_authed(user):
        items = []
        for pid, entry in list_owned(PREPROCESSORS, user):
            ds_id = entry.get("datasetId")
            ds_bundle = DATASETS.get(ds_id) or {}
            if ds_bundle.get("_owner") != entry.get("_owner"):
                ds_bundle = {}
            items.append({
                "id": pid,
                "datasetId": ds_id,
                "fileName": ds_bundle.get("fileName"),
                "target": entry.get("target"),
                "featureCount": len(entry.get("featureNames", [])),
                "trainSize": int(len(entry["X_train"])) if entry.get("X_train") is not None else 0,
                "testSize": int(len(entry["X_test"])) if entry.get("X_test") is not None else 0,
                "createdAt": entry.get("createdAt"),
            })
        items.sort(key=lambda x: x.get("createdAt") or 0, reverse=True)
        return items

    pp_rows = (db.query(DbPreprocessor)
               .filter_by(user_id=user.id)
               .order_by(DbPreprocessor.created_at.desc())
               .all())
    out = []
    for r in pp_rows:
        ds = db.query(DbDataset).filter_by(id=r.dataset_id, user_id=user.id).first()
        # train/test size 從 pickle 反序列出來 (慢但只在 list 時),或省略 — 為了快只算長度
        try:
            X_tr, X_te, _, _ = pickle.loads(r.train_test_pkl)
            train_size = len(X_tr)
            test_size  = len(X_te)
        except Exception:
            train_size = test_size = 0
        out.append({
            "id": r.id,
            "datasetId": r.dataset_id,
            "fileName": ds.file_name if ds else None,
            "target": r.target,
            "featureCount": len(json.loads(r.feature_names_json) if r.feature_names_json else []),
            "trainSize": train_size,
            "testSize": test_size,
            "createdAt": r.created_at.timestamp() if r.created_at else None,
        })
    return out


# ============================================================
# MODEL (sklearn 引擎用 — pipeline 不存到這裡)
# ============================================================
def save_model(
    bundle: dict,
    estimator, scaler, X_test_df: pd.DataFrame,
    user, db: Session,
    *,
    preprocessor_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    training_run_id: Optional[str] = None,
    hyperparameters: Optional[dict] = None,
    commit: bool = True,
) -> str:
    """存 sklearn 模型 — bundle + estimator + scaler + X_test (給 SHAP) + hyperparams。

    commit=False:只 db.add() 不 commit,讓呼叫者 (event_stream) 在迴圈結束後一次 commit,
                  把 N 次 round-trip 壓成 1 次,大幅加速。
    """
    model_id = f"model_{uuid.uuid4().hex[:8]}"
    bundle["id"] = model_id

    if not _is_authed(user):
        MODELS[model_id] = stamp({
            "bundle": bundle,
            "estimator": estimator,
            "scaler": scaler,
            "featureNames": bundle.get("featureNames", []),
            "X_test_df": X_test_df,
            "preprocessorId": preprocessor_id,
            "hyperparameters": hyperparameters or {},
        }, user)
        return model_id

    # DB path — pickle 太大時(>20MB)就跳過,只存 bundle。
    _BLOB_MAX = 20 * 1024 * 1024  # 20 MB per blob

    def _safe_pickle(obj, label):
        """pickle 物件;太大或失敗就回 None 並印 warning。"""
        if obj is None:
            return None
        try:
            blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            if len(blob) > _BLOB_MAX:
                print(f"[save_model] {label} pickle 太大 ({len(blob)/1024/1024:.1f}MB > 20MB),跳過 — "
                      f"此模型不能跑 SHAP/predict,但洞察/排行榜顯示正常",
                      flush=True)
                return None
            return blob
        except Exception as e:
            print(f"[save_model] {label} pickle 失敗: {e} — 此模型不能跑 SHAP/predict", flush=True)
            return None

    estimator_blob = _safe_pickle(estimator, "estimator")
    scaler_blob    = _safe_pickle(scaler,    "scaler")
    # x_test_pkl: 預處理來源的 X_test 已經存在 preprocessor 表裡,N 個模型不要重複存
    # (raw 來源的 X_test 是該模型訓練時 train_test_split 切出來的,獨一無二,還是要存)
    if preprocessor_id:
        x_test_blob = None
    else:
        x_test_blob = _safe_pickle(X_test_df, "x_test")

    db.add(DbModel(
        id=model_id,
        user_id=user.id,
        dataset_id=dataset_id,
        preprocessor_id=preprocessor_id,
        training_run_id=training_run_id,
        algorithm=bundle.get("type") or bundle.get("algorithm") or "",
        name=bundle.get("name", ""),
        data_source=bundle.get("dataSource"),
        task_type=bundle.get("taskType", "classification"),
        target=bundle.get("targetName") or bundle.get("target"),
        bundle_json=json.dumps(sanitize_for_json(bundle), ensure_ascii=False, default=_json_default),
        hyperparameters_json=json.dumps(sanitize_for_json(hyperparameters or {}), ensure_ascii=False, default=_json_default),
        estimator_pkl=estimator_blob,
        scaler_pkl=scaler_blob,
        x_test_pkl=x_test_blob,
        feature_names_json=json.dumps(bundle.get("featureNames", []), ensure_ascii=False),
        test_score=bundle.get("metrics", {}).get("testScore", 0.0),
        train_time_ms=bundle.get("trainTime", 0),
    ))
    if commit:
        db.commit()
    return model_id


def list_models(user, db: Session, *, training_run_id: Optional[str] = None,
                dataset_id: Optional[str] = None, limit: int = 200) -> list[dict]:
    """列出 user 的所有 sklearn 模型 — 輕量版,只回 bundle (含 metrics/featureImportance/testTrue/Pred/means/stds/featureStats),
    不回 estimator/scaler/X_test pickle (那些重物,要用 batch predict / SHAP 時才會由 get_model 載出)。
    給前端登入時一次性還原 leaderboard / insights / what-if 用。"""
    if not _is_authed(user):
        items = []
        for mid, entry in list_owned(MODELS, user):
            bundle = entry.get("bundle", {})
            bundle["id"] = mid
            items.append({
                "id": mid,
                "bundle": bundle,
                "hyperparameters": entry.get("hyperparameters", {}),
                "preprocessorId": entry.get("preprocessorId"),
                "createdAt": None,
            })
        return items

    q = db.query(DbModel).filter_by(user_id=user.id)
    if training_run_id:
        q = q.filter_by(training_run_id=training_run_id)
    if dataset_id:
        q = q.filter_by(dataset_id=dataset_id)
    rows = q.order_by(DbModel.created_at.desc()).limit(limit).all()
    out = []
    for m in rows:
        bundle = json.loads(m.bundle_json) if m.bundle_json else {}
        bundle["id"] = m.id
        out.append({
            "id": m.id,
            "bundle": bundle,
            "hyperparameters": json.loads(m.hyperparameters_json) if m.hyperparameters_json else {},
            "preprocessorId": m.preprocessor_id,
            "datasetId": m.dataset_id,
            "trainingRunId": m.training_run_id,
            "algorithm": m.algorithm,
            "name": m.name,
            "dataSource": m.data_source,
            "testScore": m.test_score,
            "trainTimeMs": m.train_time_ms,
            "createdAt": m.created_at.timestamp() if m.created_at else None,
        })
    return out


def get_model(model_id: str, user, db: Session) -> dict:
    if not _is_authed(user):
        return get_owned(MODELS, model_id, user, "model")
    m = db.query(DbModel).filter_by(id=model_id, user_id=user.id).first()
    if m is None:
        raise HTTPException(status_code=404, detail="model 不存在")

    # X_test_df:預處理來源的模型不存自己的 pickle,從 preprocessor 撈
    x_test_df = pickle.loads(m.x_test_pkl) if m.x_test_pkl else None
    if x_test_df is None and m.preprocessor_id:
        pp = db.query(DbPreprocessor).filter_by(id=m.preprocessor_id, user_id=user.id).first()
        if pp and pp.train_test_pkl:
            try:
                _, X_test, _, _ = pickle.loads(pp.train_test_pkl)
                x_test_df = X_test
            except Exception as e:
                print(f"[get_model] 從 preprocessor {m.preprocessor_id} 撈 X_test 失敗: {e}", flush=True)

    return {
        "id": m.id,
        "bundle": json.loads(m.bundle_json) if m.bundle_json else {},
        "estimator": pickle.loads(m.estimator_pkl) if m.estimator_pkl else None,
        "scaler": pickle.loads(m.scaler_pkl) if m.scaler_pkl else None,
        "featureNames": json.loads(m.feature_names_json) if m.feature_names_json else [],
        "X_test_df": x_test_df,
        "preprocessorId": m.preprocessor_id,
        "hyperparameters": json.loads(m.hyperparameters_json) if m.hyperparameters_json else {},
    }


# ============================================================
# TRAINING RUN (兩個引擎共用)
# ============================================================
def create_training_run(
    *,
    dataset_id: str,
    dataset_name: str,
    engine: str,
    target: str,
    task_type: str,
    sources: list,
    options: dict,
    user, db: Session,
) -> Optional[str]:
    """開始一次訓練 — 寫 'running' 狀態到 DB。回傳 run_id。Guest 回 None。"""
    if not _is_authed(user):
        return None
    run_id = str(uuid.uuid4())
    db.add(DbTrainingRun(
        id=run_id,
        user_id=user.id,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        engine=engine,
        target=target,
        task_type=task_type,
        sources_json=json.dumps(sources, ensure_ascii=False),
        options_json=json.dumps(options, ensure_ascii=False, default=_json_default),
        status="running",
        started_at=datetime.utcnow(),
    ))
    db.commit()
    return run_id


def finish_training_run(
    run_id: Optional[str], user, db: Session,
    *,
    status: str = "completed",
    results_summary: Optional[dict] = None,
    model_ids: Optional[list] = None,
    has_predictions: bool = False,
    error_msg: Optional[str] = None,
    elapsed_sec: Optional[float] = None,
) -> None:
    if run_id is None or not _is_authed(user):
        return
    run = db.query(DbTrainingRun).filter_by(id=run_id, user_id=user.id).first()
    if run is None: return
    run.status = status
    run.finished_at = datetime.utcnow()
    run.elapsed_sec = elapsed_sec
    if results_summary is not None:
        run.results_summary_json = json.dumps(results_summary, ensure_ascii=False, default=_json_default)
    if model_ids is not None:
        run.model_ids_json = json.dumps(model_ids)
    run.has_predictions = has_predictions
    run.error_msg = error_msg
    # 進度欄位:完成時設 100%,失敗就保持當下進度。讓 frontend 看得到「跑到哪裡死的」
    run.last_seen_at = datetime.utcnow()
    if status == "completed":
        run.progress_pct = 100
        run.current_step = "完成"
    db.commit()


# ============================================================
# 即時進度 — SSE 訓練中旁路寫進 DB (throttled 每 2 秒最多一次)
# 給其他分頁 / 裝置 poll training_runs/{id}/progress 用
# ============================================================
_PROGRESS_THROTTLE: dict[str, float] = {}     # run_id → last write timestamp
_PROGRESS_LOG_BUF:  dict[str, list] = {}      # run_id → ring buffer of last log lines
_LOG_BUF_MAX = 30

def update_training_progress(
    run_id: Optional[str], user, db: Session,
    *,
    pct: Optional[int] = None,
    step: Optional[str] = None,
    log_line: Optional[dict] = None,    # {"msg": str, "level": str, "ts": str}
    force: bool = False,                # True 跳過 throttle (例如 first/last write)
) -> None:
    """寫 training run 即時進度。每 2 秒最多寫一次 (但 log buffer 仍會持續累積)。"""
    if run_id is None or not _is_authed(user):
        return
    # 先把 log 累積進 in-memory ring buffer (不寫 DB,讓 throttle 決定何時 flush)
    buf = _PROGRESS_LOG_BUF.setdefault(run_id, [])
    if log_line:
        buf.append(log_line)
        if len(buf) > _LOG_BUF_MAX:
            del buf[: len(buf) - _LOG_BUF_MAX]

    # Throttle: 距離上次寫 DB < 2 秒就跳過 (除非強制)
    now = time.time()
    last = _PROGRESS_THROTTLE.get(run_id, 0)
    if not force and now - last < 2.0:
        return
    _PROGRESS_THROTTLE[run_id] = now

    try:
        run = db.query(DbTrainingRun).filter_by(id=run_id, user_id=user.id).first()
        if run is None:
            return
        if pct is not None:
            run.progress_pct = max(0, min(100, int(pct)))
        if step is not None:
            run.current_step = str(step)[:120]
        if buf:
            run.latest_log_json = json.dumps(buf, ensure_ascii=False, default=_json_default)
        run.last_seen_at = datetime.utcnow()
        db.commit()
    except Exception as e:
        # 寫進度失敗不能影響訓練 — 印 log 就放掉
        print(f"[update_training_progress] {run_id}: {e}", flush=True)
        try: db.rollback()
        except Exception: pass


def get_training_progress(run_id: str, user, db: Session) -> Optional[dict]:
    """讀 training run 的即時進度 (給 polling endpoint 用,輕量,不抓 results_summary 等大欄位)。"""
    if not _is_authed(user):
        return None
    run = db.query(DbTrainingRun).filter_by(id=run_id, user_id=user.id).first()
    if run is None:
        return None
    log_lines = []
    if run.latest_log_json:
        try: log_lines = json.loads(run.latest_log_json)
        except Exception: pass
    return {
        "id": run.id,
        "status": run.status,
        "progressPct": run.progress_pct or 0,
        "currentStep": run.current_step,
        "latestLog": log_lines,
        "lastSeenAt": run.last_seen_at.timestamp() if run.last_seen_at else None,
        "startedAt": run.started_at.timestamp() if run.started_at else None,
        "finishedAt": run.finished_at.timestamp() if run.finished_at else None,
        "elapsedSec": run.elapsed_sec,
        "errorMsg": run.error_msg,
    }


def list_training_runs(
    user, db: Session,
    *,
    dataset_id: Optional[str] = None,
    target: Optional[str] = None,
    engine: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """列出 user 的訓練紀錄,可篩選 dataset/target/engine。最新排前。"""
    if not _is_authed(user):
        return []
    q = db.query(DbTrainingRun).filter_by(user_id=user.id)
    if dataset_id: q = q.filter_by(dataset_id=dataset_id)
    if target:     q = q.filter_by(target=target)
    if engine:     q = q.filter_by(engine=engine)
    rows = q.order_by(DbTrainingRun.started_at.desc()).limit(limit).all()
    return [{
        "id": r.id,
        "datasetId": r.dataset_id,
        "datasetName": r.dataset_name,
        "engine": r.engine,
        "target": r.target,
        "taskType": r.task_type,
        "sources": json.loads(r.sources_json) if r.sources_json else [],
        "options": json.loads(r.options_json) if r.options_json else {},
        "resultsSummary": json.loads(r.results_summary_json) if r.results_summary_json else None,
        "modelIds": json.loads(r.model_ids_json) if r.model_ids_json else [],
        "status": r.status,
        "errorMsg": r.error_msg,
        "hasPredictions": bool(r.has_predictions),
        "startedAt": r.started_at.timestamp() if r.started_at else None,
        "finishedAt": r.finished_at.timestamp() if r.finished_at else None,
        "elapsedSec": r.elapsed_sec,
    } for r in rows]


def get_training_run(run_id: str, user, db: Session) -> dict:
    if not _is_authed(user):
        raise HTTPException(status_code=404, detail="training run 不存在")
    r = db.query(DbTrainingRun).filter_by(id=run_id, user_id=user.id).first()
    if r is None:
        raise HTTPException(status_code=404, detail="training run 不存在")
    return {
        "id": r.id, "datasetId": r.dataset_id, "datasetName": r.dataset_name,
        "engine": r.engine, "target": r.target, "taskType": r.task_type,
        "sources": json.loads(r.sources_json) if r.sources_json else [],
        "options": json.loads(r.options_json) if r.options_json else {},
        "resultsSummary": json.loads(r.results_summary_json) if r.results_summary_json else None,
        "modelIds": json.loads(r.model_ids_json) if r.model_ids_json else [],
        "status": r.status, "errorMsg": r.error_msg,
        "hasPredictions": bool(r.has_predictions),
        "startedAt": r.started_at.timestamp() if r.started_at else None,
        "finishedAt": r.finished_at.timestamp() if r.finished_at else None,
        "elapsedSec": r.elapsed_sec,
    }


# ============================================================
# PREDICTION ARTIFACT (pipeline Option B 用)
# ============================================================
def save_prediction_artifact(
    training_run_id: str, kind: str, file_name: str, content_bytes: bytes,
    user, db: Session,
) -> str:
    if not _is_authed(user):
        # guest 沒 training run 概念,artifact 也跟著消散
        return ""
    aid = str(uuid.uuid4())
    db.add(DbPredictionArtifact(
        id=aid,
        training_run_id=training_run_id,
        user_id=user.id,
        kind=kind,
        file_name=file_name,
        content_blob=content_bytes,
    ))
    db.commit()
    return aid


def get_prediction_artifact(training_run_id: str, kind: str, user, db: Session) -> Optional[dict]:
    if not _is_authed(user):
        return None
    r = (db.query(DbPredictionArtifact)
         .filter_by(training_run_id=training_run_id, user_id=user.id, kind=kind)
         .order_by(DbPredictionArtifact.created_at.desc())
         .first())
    if r is None:
        return None
    return {"id": r.id, "fileName": r.file_name, "contentBytes": r.content_blob, "kind": r.kind}


# ============================================================
# Internal helpers
# ============================================================
def _json_default(o):
    """處理 numpy / pandas 物件序列化。"""
    import numpy as np
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.ndarray,)):  return o.tolist()
    if isinstance(o, datetime):       return o.isoformat()
    return str(o)


def sanitize_for_json(obj):
    """遞迴把 dict/list 裡的 NaN / Inf 換成 None (JS JSON.parse 不接受 NaN)。
    也順手把 numpy scalar 拆成 python 原生型,讓下游 json.dumps 不再依賴 default=。"""
    import math
    import numpy as np
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj
