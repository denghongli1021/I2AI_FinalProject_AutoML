// ===== API CLIENT — 串接後端 Python (FastAPI) =====
// 啟用方式:在「系統設定」頁面打開「使用 Python 後端 API」開關。
// 也可在 console 執行: ApiClient.setEnabled(true) 或修改 localStorage.useApi。

const ApiClient = {
  baseUrl: localStorage.getItem('apiBaseUrl') || 'https://i2ai-automl-api.onrender.com',
  enabled: localStorage.getItem('useApi') === 'true',

  setEnabled(v) {
    this.enabled = !!v;
    localStorage.setItem('useApi', this.enabled ? 'true' : 'false');
  },

  setBaseUrl(url) {
    this.baseUrl = url.replace(/\/$/, '');
    localStorage.setItem('apiBaseUrl', this.baseUrl);
  },

  async health() {
    try {
      const r = await fetch(`${this.baseUrl}/api/health`, { method: 'GET' });
      return r.ok;
    } catch (_) {
      return false;
    }
  },

  // ---- 0. DATASET LIST / GET — 登入後 (或 guest) 把後端記憶體裡的 datasets 還原回前端 ----
  async datasetList() {
    const r = await fetch(`${this.baseUrl}/api/dataset/list`, { method: 'GET' });
    if (!r.ok) throw new Error(`datasetList 失敗 (${r.status})`);
    const j = await r.json();
    return j.datasets || [];
  },

  async datasetGet(id) {
    const r = await fetch(`${this.baseUrl}/api/dataset/${encodeURIComponent(id)}`, { method: 'GET' });
    if (!r.ok) throw new Error(`datasetGet 失敗 (${r.status})`);
    return r.json();
  },

  async datasetDelete(id) {
    const r = await fetch(`${this.baseUrl}/api/dataset/${encodeURIComponent(id)}`, { method: 'DELETE' });
    if (!r.ok) throw new Error(`datasetDelete 失敗 (${r.status})`);
    return r.json();
  },

  async modelDelete(modelId) {
    const headers = {};
    if (this.token) headers['Authorization'] = `Bearer ${this.token}`;
    const r = await fetch(`${this.baseUrl}/api/models/${encodeURIComponent(modelId)}`, { method: 'DELETE', headers });
    if (!r.ok) throw new Error(`modelDelete 失敗 (${r.status}): ${await r.text()}`);
    return r.json();
  },

  // ---- 1. PREPROCESS ----
  async preprocess(file) {
    const fd = new FormData();
    fd.append('file', file);
    const r = await fetch(`${this.baseUrl}/api/preprocess`, { method: 'POST', body: fd });
    if (!r.ok) throw new Error(`preprocess 失敗 (${r.status}): ${await r.text()}`);
    return r.json();
  },

  // ---- 1b. PREPROCESS — 隊友模組 (audit / transform / inference) ----
  async preprocessAudit(payload) {
    const r = await fetch(`${this.baseUrl}/api/preprocess/audit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!r.ok) throw new Error(`audit 失敗 (${r.status}): ${await r.text()}`);
    return r.json();
  },

  async preprocessTransform(payload) {
    // 改用 multipart 是為了支援可選的 adversarialTestFile (Kaggle 風 test.csv)
    // 觸發 daniel preprocess_for_training 的 adversarial validation
    const fd = new FormData();
    fd.append('datasetId', String(payload.datasetId));
    fd.append('target', String(payload.target));
    if (payload.testSize != null) fd.append('testSize', String(payload.testSize));
    // useMice / useMiSelection 已被 daniel 新版內部接管,還是送過去當紀錄
    if (payload.useMice != null) fd.append('useMice', String(!!payload.useMice));
    if (payload.useMiSelection != null) fd.append('useMiSelection', String(!!payload.useMiSelection));
    if (payload.miThreshold != null) fd.append('miThreshold', String(payload.miThreshold));
    if (payload.adversarialTestFile) fd.append('adversarialTestFile', payload.adversarialTestFile);
    const r = await fetch(`${this.baseUrl}/api/preprocess/transform`, {
      method: 'POST',
      body: fd,   // 不要設 Content-Type, fetch 會自動加 multipart boundary
    });
    if (!r.ok) throw new Error(`transform 失敗 (${r.status}): ${await r.text()}`);
    return r.json();
  },

  async preprocessInference(payload) {
    const r = await fetch(`${this.baseUrl}/api/preprocess/inference`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!r.ok) throw new Error(`inference 失敗 (${r.status}): ${await r.text()}`);
    return r.json();
  },

  // 列出後端所有可用的 preprocessor
  async preprocessList() {
    const r = await fetch(`${this.baseUrl}/api/preprocess/list`, { method: 'GET' });
    if (!r.ok) throw new Error(`preprocess list 失敗 (${r.status}): ${await r.text()}`);
    return r.json();
  },

  // ---- 2. TRAIN ----
  // payload: { datasetId, target, features, algorithms, options }
  async train(payload) {
    const r = await fetch(`${this.baseUrl}/api/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!r.ok) throw new Error(`train 失敗 (${r.status}): ${await r.text()}`);
    return r.json();
  },

  // ---- 共用：取得帶 token 的 JSON headers ----
  _authJsonHeaders() {
    const h = { 'Content-Type': 'application/json' };
    if (typeof AuthClient !== 'undefined' && AuthClient.token)
      h['Authorization'] = `Bearer ${AuthClient.token}`;
    else if (this.token)
      h['Authorization'] = `Bearer ${this.token}`;
    return h;
  },

  // ---- 3. VISUALIZE ----
  // payload: { modelId, chartType, options }
  async visualize(payload) {
    const r = await fetch(`${this.baseUrl}/api/visualize`, {
      method: 'POST',
      headers: this._authJsonHeaders(),
      body: JSON.stringify(payload),
    });
    if (!r.ok) throw new Error(`visualize 失敗 (${r.status}): ${await r.text()}`);
    return r.json();
  },

  // ---- 3b. VISUALIZE (SHAP) — 隊友 AutoMLVisualizer ----
  // payload: { modelId, sampleIndex, targetFeature?, maxSamples? }
  async visualizeShap(payload) {
    // signal 從 payload 抽出 → AbortController 控制取消
    const { signal, ...body } = payload;
    const r = await fetch(`${this.baseUrl}/api/visualize/shap`, {
      method: 'POST',
      headers: this._authJsonHeaders(),
      body: JSON.stringify(body),
      signal,
    });
    if (!r.ok) throw new Error(`SHAP 失敗 (${r.status}): ${await r.text()}`);
    return r.json();
  },

  // ---- 4. PREDICT (What-If Simulator) ----
  // payload: { modelId, features: [v1, v2, ...] }   raw scale, order matches featureNames
  async predict(payload) {
    const r = await fetch(`${this.baseUrl}/api/predict`, {
      method: 'POST',
      headers: this._authJsonHeaders(),
      body: JSON.stringify(payload),
    });
    if (!r.ok) throw new Error(`predict 失敗 (${r.status}): ${await r.text()}`);
    return r.json();
  },

  // ---- 4c. MODEL INFO — 取得特徵名稱 + 統計值 (供 What-If 初始化) ----
  async get(path) {
    const headers = {};
    if (typeof AuthClient !== 'undefined' && AuthClient.token)
      headers['Authorization'] = `Bearer ${AuthClient.token}`;
    else if (this.token)
      headers['Authorization'] = `Bearer ${this.token}`;
    const r = await fetch(`${this.baseUrl}${path}`, { method: 'GET', headers });
    if (!r.ok) throw new Error(`GET ${path} 失敗 (${r.status}): ${await r.text()}`);
    return r.json();
  },

  // ---- 4b. PREDICT (BATCH) — 上傳 CSV,整批預測,回傳含預測欄的 CSV blob ----
  // sampleFile 選填:給了就照範本 submission 格式輸出
  async predictBatch(modelId, file, sampleFile = null) {
    const fd = new FormData();
    fd.append('modelId', modelId);
    fd.append('file', file);
    if (sampleFile) fd.append('sampleFile', sampleFile);
    const authHeaders = {};
    if (typeof AuthClient !== 'undefined' && AuthClient.token)
      authHeaders['Authorization'] = `Bearer ${AuthClient.token}`;
    else if (this.token)
      authHeaders['Authorization'] = `Bearer ${this.token}`;
    const r = await fetch(`${this.baseUrl}/api/predict/batch`, { method: 'POST', body: fd, headers: authHeaders });
    if (!r.ok) throw new Error(`批次預測失敗 (${r.status}): ${await r.text()}`);
    return r.blob();  // CSV 檔
  },

  // ---- 5. TRAIN (SSE streaming) ----
  // onEvent: callback({type, ...}) — type 可能是 'progress' | 'log' | 'done' | 'error'
  // 回傳 promise,完成時 resolve 為 done 事件中的 models 陣列
  async trainStream(payload, onEvent, signal) {
    // 帶 Authorization header — 不然後端會把 request 當 guest,結果寫到 in-memory dict 而不是 DB
    const headers = { 'Content-Type': 'application/json' };
    if (typeof AuthClient !== 'undefined' && AuthClient.token) {
      headers['Authorization'] = `Bearer ${AuthClient.token}`;
    }
    const r = await fetch(`${this.baseUrl}/api/train/stream`, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal,
    });
    if (!r.ok || !r.body) throw new Error(`trainStream 失敗 (${r.status}): ${await r.text()}`);

    const reader = r.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    const collectedModels = [];   // 從 'model' 事件累積
    let finalModels = null;        // 'done' 來臨時定型
    let legacyModels = null;       // 後端若還是用舊版單一 done 事件 (含整個 models 陣列) 的 fallback
    let errorMsg = null;

    const parseAndDispatch = (json) => {
      let ev;
      try { ev = JSON.parse(json); }
      catch (e) {
        console.warn('[trainStream] JSON.parse 失敗,skip:', e.message, json.slice(0, 100));
        return;
      }
      if (ev.type === 'model' && ev.bundle) {
        collectedModels.push(ev.bundle);
      } else if (ev.type === 'done') {
        // 新版:done 只是 sentinel,models 從 collectedModels 取
        // 舊版相容:若 done 帶 models 陣列,直接吃下來
        if (Array.isArray(ev.models)) legacyModels = ev.models;
        finalModels = legacyModels && legacyModels.length ? legacyModels : collectedModels;
      } else if (ev.type === 'error') {
        errorMsg = ev.message || 'unknown';
      }
      if (onEvent) onEvent(ev);
    };

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // SSE: events 以 \n\n 分隔,每行 "data: {json}"
      const parts = buffer.split('\n\n');
      buffer = parts.pop(); // keep last incomplete chunk
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith('data:')) continue;
        const json = line.slice(5).trim();
        if (!json) continue;
        parseAndDispatch(json);
      }
    }
    // 連線結束時 flush 殘留 buffer (有時最後一個 chunk 沒 \n\n 收尾)
    if (buffer.trim().startsWith('data:')) {
      parseAndDispatch(buffer.trim().slice(5).trim());
    }

    if (errorMsg) throw new Error(errorMsg);
    // 沒 done 但有 collectedModels 也算成功
    if (finalModels === null && collectedModels.length > 0) finalModels = collectedModels;
    return finalModels || [];
  },
};

// 把後端回傳的 dataset 灌進 DataEngine,讓現有的 render 邏輯可以直接用
function adoptApiDataset(apiDataset) {
  const headers = apiDataset.headers;
  const data = apiDataset.data;
  const columns = {};
  headers.forEach((h, i) => {
    columns[h] = data.map(row => row[i]);
  });

  const dataset = {
    id: apiDataset.id || Date.now(),
    fileName: apiDataset.fileName,
    headers,
    data,
    columns,
    rowCount: apiDataset.rowCount,
    colCount: apiDataset.colCount,
    analysis: apiDataset.analysis,
    loadedAt: new Date(),
    _fromApi: true,
    // 後端預先算好的進階指標 (DataEngine 的對應方法會優先用這些)
    _correlation: apiDataset.correlation || null,
    _healthScore: apiDataset.healthScore || null,
    _processingLog: apiDataset.processingLog || null,
  };

  const existIdx = DataEngine.datasets.findIndex(d => d.fileName === dataset.fileName);
  if (existIdx >= 0) DataEngine.datasets[existIdx] = dataset;
  else DataEngine.datasets.push(dataset);

  DataEngine.currentDataset = dataset;

  // 攔截 DataEngine 的進階方法,優先回傳後端結果
  const _origCorr = DataEngine.computeCorrelationMatrix.bind(DataEngine);
  const _origHealth = DataEngine.computeHealthScore.bind(DataEngine);
  const _origLog = DataEngine.generateProcessingLog.bind(DataEngine);
  DataEngine.computeCorrelationMatrix = function () {
    return this.currentDataset?._correlation || _origCorr();
  };
  DataEngine.computeHealthScore = function () {
    return this.currentDataset?._healthScore || _origHealth();
  };
  DataEngine.generateProcessingLog = function () {
    return this.currentDataset?._processingLog || _origLog();
  };

  return dataset;
}
