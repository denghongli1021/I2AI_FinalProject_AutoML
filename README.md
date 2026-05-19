---
title: I2AI AutoML Backend
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: FastAPI backend for the I2AI AutoML project (sklearn + pipeline engine)
---

# I2AI AutoML

清大「人工智慧概論」期末專題 — 端對端 AutoML 平台。

## 架構
- **前端**: 靜態檔 → [GitHub Pages](https://denghongli1021.github.io/I2AI_FinalProject_AutoML/)
- **後端 (primary)**: FastAPI on HuggingFace Spaces (this repo, Docker SDK)
- **後端 (backup)**: FastAPI on Render free tier
- **DB**: Supabase Postgres

## 功能
- 數據集管理 + 預處理
- 多演算法訓練 (sklearn / pipeline)
- SHAP 解釋、What-If 模擬
- OAuth 登入 (Google / GitHub)

## 本地開發
```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload
```
