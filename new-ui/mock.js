// Mock data — realistic enough that the UI feels alive, but not so elaborate
// that we're maintaining a fake DB. Everything in one place so it's easy to find.

const MOCK = {
  // ---- Workspace + user ----
  user: { name: '陳同學', email: 'student@gapp.nthu.edu.tw', provider: 'google' },

  // ---- Sidebar counts ----
  counts: { datasets: 3, experiments: 12, models: 47 },

  // ---- API + system status ----
  apiHealth: { ok: true, latency: 42 },

  // ---- Datasets ----
  datasets: [
    { id: 'churn', name: '客戶資料 2024', rows: 12450, cols: 23, target: 'churn', task: 'classification', uploadedAt: '今天 14:02', size: '4.2 MB', health: 87 },
    { id: 'sales', name: '銷售紀錄 Q1-Q4', rows: 89200, cols: 18, target: 'revenue', task: 'regression', uploadedAt: '昨天', size: '28 MB', health: 73 },
    { id: 'reviews', name: '產品評價數據', rows: 34100, cols: 12, target: 'rating', task: 'regression', uploadedAt: '3天前', size: '11 MB', health: 91 },
  ],

  // ---- Columns of the active dataset (churn 2024) ----
  columns: [
    { name: 'customer_id', type: 'id',   missing: 0,    unique: 12450, role: 'id'      },
    { name: 'age',         type: 'int',  missing: 0,    unique: 62,    role: 'feature' },
    { name: 'gender',      type: 'cat',  missing: 12,   unique: 2,     role: 'feature' },
    { name: 'tenure',      type: 'int',  missing: 0,    unique: 73,    role: 'feature' },
    { name: 'monthly_charge', type:'float', missing: 30, unique: 1430, role:'feature'  },
    { name: 'total_charge', type: 'float', missing: 0,  unique: 9821,  role: 'feature' },
    { name: 'contract',    type: 'cat',  missing: 0,    unique: 3,     role: 'feature' },
    { name: 'payment',     type: 'cat',  missing: 0,    unique: 4,     role: 'feature' },
    { name: 'internet',    type: 'cat',  missing: 0,    unique: 3,     role: 'feature' },
    { name: 'phone',       type: 'cat',  missing: 0,    unique: 2,     role: 'feature' },
    { name: 'streaming',   type: 'cat',  missing: 0,    unique: 3,     role: 'feature' },
    { name: 'complaints',  type: 'int',  missing: 0,    unique: 8,     role: 'feature' },
    { name: 'last_login',  type: 'date', missing: 145,  unique: 980,   role: 'feature' },
    { name: 'churn',       type: 'cat',  missing: 0,    unique: 2,     role: 'target'  },
  ],

  // ---- Experiments (runs) ----
  experiments: [
    { id: 'exp-12', name: '客戶流失預測 v3', dataset: 'churn', task: 'classification',
      status: 'completed', startedAt: '2小時前', duration: '8m 12s',
      best: { algo: 'XGBoost', metric: 'F1', value: 0.942 },
      models: 7,
    },
    { id: 'exp-11', name: '銷售額預測 Q2', dataset: 'sales', task: 'regression',
      status: 'running', startedAt: '剛剛', duration: '3m 47s',
      progress: 0.47,
      best: { algo: 'LightGBM', metric: 'RMSE', value: 2418 },
      models: 5,
    },
    { id: 'exp-10', name: '產品推薦引擎', dataset: 'reviews', task: 'classification',
      status: 'completed', startedAt: '昨天 18:30', duration: '12m 04s',
      best: { algo: 'Ensemble', metric: 'AUC', value: 0.978 },
      models: 9,
    },
    { id: 'exp-09', name: '庫存需求預測', dataset: 'sales', task: 'regression',
      status: 'failed', startedAt: '2天前', duration: '1m 22s',
      error: '數據品質不足 — 缺失率 > 50%',
      models: 0,
    },
  ],

  // ---- Models (leaderboard for active experiment) ----
  models: [
    { id: 'm1', rank: 1, algo: 'XGBoost',        source: 'preprocessed', f1: 0.942, auc: 0.961, acc: 0.928, trainTime:  18, inferLatency: 2.1, tag: 'best' },
    { id: 'm2', rank: 2, algo: 'LightGBM',       source: 'preprocessed', f1: 0.938, auc: 0.957, acc: 0.924, trainTime:  12, inferLatency: 1.8, tag: 'fast' },
    { id: 'm3', rank: 3, algo: 'CatBoost',       source: 'preprocessed', f1: 0.935, auc: 0.953, acc: 0.920, trainTime:  45, inferLatency: 2.3, tag: '' },
    { id: 'm4', rank: 4, algo: 'Random Forest',  source: 'preprocessed', f1: 0.921, auc: 0.944, acc: 0.911, trainTime:  31, inferLatency: 5.4, tag: '' },
    { id: 'm5', rank: 5, algo: 'XGBoost',        source: 'raw',          f1: 0.917, auc: 0.940, acc: 0.908, trainTime:  16, inferLatency: 2.0, tag: '' },
    { id: 'm6', rank: 6, algo: 'Logistic Reg.',  source: 'preprocessed', f1: 0.882, auc: 0.911, acc: 0.876, trainTime:   2, inferLatency: 0.4, tag: 'baseline' },
    { id: 'm7', rank: 7, algo: 'KNN',            source: 'preprocessed', f1: 0.840, auc: 0.870, acc: 0.834, trainTime:   1, inferLatency: 8.2, tag: '' },
  ],

  // ---- Algorithms picker ----
  algorithms: [
    { id: 'xgb', name: 'XGBoost', kind: 'gbm', enabled: true },
    { id: 'lgbm', name: 'LightGBM', kind: 'gbm', enabled: true },
    { id: 'cat', name: 'CatBoost', kind: 'gbm', enabled: true },
    { id: 'rf', name: 'Random Forest', kind: 'tree', enabled: true },
    { id: 'et', name: 'Extra Trees', kind: 'tree', enabled: false },
    { id: 'lr', name: 'Logistic Reg.', kind: 'linear', enabled: true },
    { id: 'knn', name: 'KNN', kind: 'instance', enabled: false },
    { id: 'svm', name: 'SVM', kind: 'kernel', enabled: false },
  ],

  // ---- Feature importance (for insights page) ----
  featureImportance: [
    { feature: 'tenure',          gain: 0.187, shap: 0.211 },
    { feature: 'monthly_charge',  gain: 0.142, shap: 0.158 },
    { feature: 'contract',        gain: 0.131, shap: 0.144 },
    { feature: 'total_charge',    gain: 0.103, shap: 0.118 },
    { feature: 'complaints',      gain: 0.092, shap: 0.087 },
    { feature: 'internet',        gain: 0.067, shap: 0.071 },
    { feature: 'payment',         gain: 0.054, shap: 0.062 },
    { feature: 'age',             gain: 0.048, shap: 0.051 },
    { feature: 'last_login',      gain: 0.042, shap: 0.044 },
    { feature: 'streaming',       gain: 0.038, shap: 0.034 },
    { feature: 'phone',           gain: 0.021, shap: 0.013 },
    { feature: 'gender',          gain: 0.012, shap: 0.007 },
  ],

  // ---- Trend data for dashboard ----
  // 7 days of best F1 across all classification runs
  trendF1: [0.876, 0.891, 0.902, 0.918, 0.925, 0.937, 0.951],
  trendRMSE: [3420, 3210, 2980, 2841, 2710, 2618, 2418],

  // Task distribution
  taskMix: { classification: 32, regression: 11, timeseries: 4 },

  // SHAP waterfall for a single sample
  shapSample: {
    base: 0.32,
    contributions: [
      { feature: 'tenure = 8 months',       delta: +0.18 },
      { feature: 'contract = Month-to-Month', delta: +0.14 },
      { feature: 'monthly_charge = $89.5',  delta: +0.09 },
      { feature: 'complaints = 3',          delta: +0.07 },
      { feature: 'internet = Fiber Optic',  delta: +0.04 },
      { feature: 'payment = Electronic',    delta: +0.02 },
      { feature: 'age = 34',                delta: -0.03 },
      { feature: 'total_charge = $720',     delta: -0.05 },
    ],
    final: 0.78,
  },

  // ---- Training runs for Insights cascade (dataset → target → run) ----
  trainingRuns: [
    { id: 'run-12', datasetId: 'churn',   datasetName: '客戶資料 2024', target: 'churn',    task: 'classification', startedAt: '今天 14:32', bestModel: 'XGBoost',   bestScore: 0.942, scoreLabel: 'F1',   modelCount: 7, engine: 'sklearn' },
    { id: 'run-11', datasetId: 'churn',   datasetName: '客戶資料 2024', target: 'churn',    task: 'classification', startedAt: '昨天 16:08', bestModel: 'LightGBM',  bestScore: 0.918, scoreLabel: 'F1',   modelCount: 6, engine: 'pipeline' },
    { id: 'run-10', datasetId: 'sales',   datasetName: '銷售紀錄 Q1-Q4', target: 'revenue',  task: 'regression',     startedAt: '昨天 11:24', bestModel: 'LightGBM',  bestScore: 2418,  scoreLabel: 'RMSE', modelCount: 5, engine: 'sklearn' },
    { id: 'run-09', datasetId: 'reviews', datasetName: '產品評價數據',   target: 'rating',   task: 'regression',     startedAt: '前天',       bestModel: 'CatBoost',  bestScore: 0.42,  scoreLabel: 'RMSE', modelCount: 4, engine: 'sklearn' },
  ],

  // ---- Model comparison data for insights ----
  // 同一筆 run 內所有 model 的多 metric 比較
  modelCompare: [
    { algo: 'XGBoost',        source: 'preprocessed', f1: 0.942, auc: 0.961, acc: 0.928 },
    { algo: 'LightGBM',       source: 'preprocessed', f1: 0.938, auc: 0.957, acc: 0.924 },
    { algo: 'CatBoost',       source: 'preprocessed', f1: 0.935, auc: 0.953, acc: 0.920 },
    { algo: 'Random Forest',  source: 'preprocessed', f1: 0.921, auc: 0.944, acc: 0.911 },
    { algo: 'XGBoost',        source: 'raw',          f1: 0.917, auc: 0.940, acc: 0.908 },
    { algo: 'Logistic Reg.',  source: 'preprocessed', f1: 0.882, auc: 0.911, acc: 0.876 },
    { algo: 'KNN',            source: 'preprocessed', f1: 0.840, auc: 0.870, acc: 0.834 },
  ],

  // ---- Regression diagnostics (for pred-vs-actual scatter + residuals) ----
  // 模擬 200 個樣本的預測 vs 實際 + 殘差
  predScatter: (() => {
    const pts = [];
    for (let i = 0; i < 160; i++) {
      const actual = 2000 + Math.random() * 6000;
      const noise = (Math.random() - 0.5) * 800;
      const pred = actual + noise;
      pts.push({ actual, pred, residual: pred - actual });
    }
    return pts;
  })(),

  // ---- SHAP interaction dependence (2 特徵交互作用) ----
  // 模擬「tenure × monthly_charge」散點:x=tenure, y=shap value, color=monthly_charge
  shapInteraction: (() => {
    const pts = [];
    for (let i = 0; i < 120; i++) {
      const tenure = Math.random() * 72;
      const monthly = 18 + Math.random() * 102;
      // 早期客戶且月費高 → SHAP 大
      const shap = 0.4 - (tenure / 72) * 0.5 + (monthly / 120) * 0.2 + (Math.random() - 0.5) * 0.1;
      pts.push({ x: tenure, y: shap, color: monthly });
    }
    return pts;
  })(),

  // ---- Notifications ----
  notifications: [
    { kind: 'warn', title: '銷售數據 2024 有 12% 缺失值', body: '建議在預處理頁面補值或移除受影響的列', time: '10m' },
    { kind: 'info', title: '超參數優化完成 — 200 組 trials', body: '最佳組合: LightGBM · lr=0.025 · depth=6', time: '2h' },
    { kind: 'good', title: '客戶流失預測 v3 已部署', body: '/api/v1/predict/churn · p99 45ms', time: '昨天' },
  ],
};

window.MOCK = MOCK;
