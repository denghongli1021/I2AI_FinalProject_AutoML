# =============================================================================
# I2AI AutoML — backend image for HuggingFace Spaces (Docker SDK)
# HF Spaces 規範:
#   - 容器要 expose port 7860 (預設)
#   - 寫入路徑只有 /data 持久,其他都重啟就消失
#   - 不能用 root 跑,加一個 user
# =============================================================================
FROM python:3.11-slim

# 系統相依 (lightgbm / catboost / psycopg2 都要 libgomp / postgres client lib)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 建非 root user (HF 規範)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /home/user/app

# 先 copy requirements 利用 docker layer cache
COPY --chown=user api/requirements.txt /home/user/app/api/requirements.txt
RUN pip install --no-cache-dir --user --upgrade pip && \
    pip install --no-cache-dir --user -r api/requirements.txt

# pipeline 用的額外套件 (CPU torch + optuna)
RUN pip install --no-cache-dir --user \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user \
        optuna>=3.5.0 \
        scipy>=1.10.0 \
        tqdm>=4.65.0

# 再 copy 程式碼
COPY --chown=user . /home/user/app

# HF Space 規定 port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
