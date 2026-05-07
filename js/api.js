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

  // ---- 1. PREPROCESS ----
  async preprocess(file) {
    const fd = new FormData();
    fd.append('file', file);
    const r = await fetch(`${this.baseUrl}/api/preprocess`, { method: 'POST', body: fd });
    if (!r.ok) throw new Error(`preprocess 失敗 (${r.status}): ${await r.text()}`);
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

  // ---- 3. VISUALIZE ----
  // payload: { modelId, chartType, options }
  async visualize(payload) {
    const r = await fetch(`${this.baseUrl}/api/visualize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!r.ok) throw new Error(`visualize 失敗 (${r.status}): ${await r.text()}`);
    return r.json();
  },

  // ---- 4. PREDICT (What-If Simulator) ----
  // payload: { modelId, features: [v1, v2, ...] }   raw scale, order matches featureNames
  async predict(payload) {
    const r = await fetch(`${this.baseUrl}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!r.ok) throw new Error(`predict 失敗 (${r.status}): ${await r.text()}`);
    return r.json();
  },

  // ---- 5. TRAIN (SSE streaming) ----
  // onEvent: callback({type, ...}) — type 可能是 'progress' | 'log' | 'done' | 'error'
  // 回傳 promise,完成時 resolve 為 done 事件中的 models 陣列
  async trainStream(payload, onEvent) {
    const r = await fetch(`${this.baseUrl}/api/train/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!r.ok || !r.body) throw new Error(`trainStream 失敗 (${r.status}): ${await r.text()}`);

    const reader = r.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let finalModels = null;
    let errorMsg = null;

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
        try {
          const ev = JSON.parse(json);
          if (ev.type === 'done') finalModels = ev.models || [];
          else if (ev.type === 'error') errorMsg = ev.message || 'unknown';
          if (onEvent) onEvent(ev);
        } catch (e) { /* ignore parse errors */ }
      }
    }
    if (errorMsg) throw new Error(errorMsg);
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
