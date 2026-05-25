// ===== MOCK DATA FOR AUTOML PLATFORM =====

const MOCK = {
  // Dataset columns for EDA table
  datasetColumns: [
    { name: 'customer_id', type: '整數', dtype: 'int64', missing: 0, outliers: 0, status: 'ok', dist: [5,8,12,15,18,20,18,15,12,8] },
    { name: 'age', type: '整數', dtype: 'int64', missing: 5.2, outliers: 1, status: 'filled', dist: [2,5,12,22,28,25,18,10,5,2] },
    { name: 'gender', type: '類別', dtype: 'category', missing: 0, outliers: 0, status: 'encoded', dist: [45,55,0,0,0,0,0,0,0,0] },
    { name: 'income', type: '浮點數', dtype: 'float64', missing: 2.1, outliers: 3, status: 'warning', dist: [8,15,22,28,18,12,8,4,2,1] },
    { name: 'tenure_days', type: '整數', dtype: 'int64', missing: 0, outliers: 0, status: 'ok', dist: [20,15,12,10,8,7,8,10,12,18] },
    { name: 'monthly_spending', type: '浮點數', dtype: 'float64', missing: 1.5, outliers: 2, status: 'filled', dist: [3,8,15,25,22,15,8,4,2,1] },
    { name: 'purchase_date', type: '日期', dtype: 'datetime', missing: 0, outliers: 0, status: 'decomposed', dist: [10,10,10,10,10,10,10,10,10,10] },
    { name: 'complaints', type: '整數', dtype: 'int64', missing: 0, outliers: 0, status: 'ok', dist: [60,25,10,3,2,0,0,0,0,0] },
    { name: 'contract_type', type: '類別', dtype: 'category', missing: 0, outliers: 0, status: 'encoded', dist: [35,40,25,0,0,0,0,0,0,0] },
    { name: 'satisfaction_score', type: '浮點數', dtype: 'float64', missing: 8.3, outliers: 0, status: 'filled', dist: [2,5,8,12,18,22,18,10,5,2] },
    { name: 'last_login_days', type: '整數', dtype: 'int64', missing: 0, outliers: 5, status: 'warning', dist: [30,20,15,10,8,5,4,3,3,2] },
    { name: 'churn', type: '二元', dtype: 'int64', missing: 0, outliers: 0, status: 'target', dist: [72,28,0,0,0,0,0,0,0,0] },
  ],

  // Leaderboard models
  leaderboardModels: [
    { name: 'LightGBM (Tuned)', f1: 0.942, auc: 0.981, accuracy: 0.953, time: '2m 18s', timeVal: 138, latency: '12ms', latencyVal: 12, tags: ['best', 'deployed'] },
    { name: 'XGBoost (Tuned)', f1: 0.938, auc: 0.978, accuracy: 0.949, time: '3m 45s', timeVal: 225, latency: '15ms', latencyVal: 15, tags: [] },
    { name: 'CatBoost', f1: 0.931, auc: 0.975, accuracy: 0.944, time: '4m 12s', timeVal: 252, latency: '18ms', latencyVal: 18, tags: [] },
    { name: 'Ensemble (Top 3)', f1: 0.948, auc: 0.984, accuracy: 0.958, time: '10m 15s', timeVal: 615, latency: '42ms', latencyVal: 42, tags: ['best'] },
    { name: 'Random Forest', f1: 0.912, auc: 0.962, accuracy: 0.928, time: '1m 30s', timeVal: 90, latency: '8ms', latencyVal: 8, tags: ['fastest'] },
    { name: 'Logistic Regression', f1: 0.873, auc: 0.921, accuracy: 0.895, time: '5s', timeVal: 5, latency: '2ms', latencyVal: 2, tags: ['fastest'] },
    { name: 'SVM (RBF)', f1: 0.889, auc: 0.945, accuracy: 0.908, time: '8m 22s', timeVal: 502, latency: '25ms', latencyVal: 25, tags: [] },
    { name: 'Neural Network', f1: 0.925, auc: 0.971, accuracy: 0.938, time: '15m 40s', timeVal: 940, latency: '35ms', latencyVal: 35, tags: [] },
  ],

  // Feature importance data
  featureImportance: [
    { name: 'tenure_days', importance: 0.182 },
    { name: 'monthly_spending', importance: 0.156 },
    { name: 'complaints', importance: 0.143 },
    { name: 'satisfaction_score', importance: 0.128 },
    { name: 'last_login_days', importance: 0.098 },
    { name: 'contract_type', importance: 0.087 },
    { name: 'age', importance: 0.072 },
    { name: 'income', importance: 0.058 },
    { name: 'age×income', importance: 0.042 },
    { name: 'tenure×frequency', importance: 0.034 },
  ],

  // SHAP waterfall data for different customers
  shapSamples: [
    {
      label: '客戶 #1024 (高風險)',
      baseValue: 0.28,
      prediction: 0.82,
      features: [
        { name: 'tenure_days = 45', value: 0.18, color: '#ef4444' },
        { name: 'complaints = 4', value: 0.15, color: '#ef4444' },
        { name: 'satisfaction = 2.1', value: 0.12, color: '#ef4444' },
        { name: 'last_login = 60d', value: 0.08, color: '#ef4444' },
        { name: 'contract = 月付', value: 0.06, color: '#ef4444' },
        { name: 'spending = 800', value: 0.04, color: '#ef4444' },
        { name: 'income = 85k', value: -0.03, color: '#10b981' },
        { name: 'age = 28', value: -0.06, color: '#10b981' },
      ]
    },
    {
      label: '客戶 #2048 (低風險)',
      baseValue: 0.28,
      prediction: 0.08,
      features: [
        { name: 'tenure_days = 1200', value: -0.08, color: '#10b981' },
        { name: 'complaints = 0', value: -0.06, color: '#10b981' },
        { name: 'satisfaction = 4.5', value: -0.05, color: '#10b981' },
        { name: 'contract = 兩年約', value: -0.04, color: '#10b981' },
        { name: 'spending = 5200', value: -0.03, color: '#10b981' },
        { name: 'last_login = 2d', value: -0.02, color: '#10b981' },
        { name: 'income = 120k', value: -0.01, color: '#10b981' },
        { name: 'age = 42', value: 0.01, color: '#ef4444' },
      ]
    },
    {
      label: '客戶 #3072 (中風險)',
      baseValue: 0.28,
      prediction: 0.45,
      features: [
        { name: 'complaints = 2', value: 0.08, color: '#ef4444' },
        { name: 'last_login = 30d', value: 0.05, color: '#ef4444' },
        { name: 'contract = 月付', value: 0.04, color: '#ef4444' },
        { name: 'satisfaction = 3.2', value: 0.03, color: '#ef4444' },
        { name: 'tenure_days = 365', value: -0.01, color: '#10b981' },
        { name: 'spending = 3000', value: -0.01, color: '#10b981' },
        { name: 'income = 55k', value: 0.01, color: '#ef4444' },
        { name: 'age = 35', value: -0.02, color: '#10b981' },
      ]
    }
  ],

  // Performance trend data
  performanceTrend: {
    dates: ['03/01','03/03','03/05','03/07','03/09','03/11','03/13','03/15','03/17','03/19','03/21'],
    xgboost: [0.912,0.918,0.922,0.925,0.928,0.930,0.932,0.935,0.936,0.937,0.938],
    lightgbm: [0.908,0.915,0.920,0.925,0.929,0.933,0.935,0.938,0.940,0.941,0.942],
    ensemble: [0.920,0.925,0.930,0.935,0.938,0.942,0.944,0.946,0.947,0.948,0.948],
  },

  // Optimization history
  optimizationHistory: (() => {
    const data = [];
    let bestF1 = 0.85;
    for (let i = 1; i <= 100; i++) {
      const trial = 0.85 + Math.random() * 0.1;
      bestF1 = Math.max(bestF1, trial);
      data.push({ trial: i, f1: parseFloat(trial.toFixed(4)), best: parseFloat(bestF1.toFixed(4)) });
    }
    return data;
  })(),

  // API usage data
  apiUsage: {
    hours: Array.from({length: 24}, (_, i) => `${i}:00`),
    calls: [120,85,60,42,38,55,180,450,820,1050,980,890,750,680,720,850,920,780,620,480,350,280,200,150],
  },

  // SHAP beeswarm data
  shapBeeswarm: (() => {
    const features = ['tenure_days', 'monthly_spending', 'complaints', 'satisfaction_score', 'last_login_days', 'contract_type', 'age', 'income'];
    const data = [];
    features.forEach((feat, fi) => {
      for (let i = 0; i < 60; i++) {
        const shapVal = (Math.random() - 0.5) * (0.3 - fi * 0.03);
        const featureVal = Math.random();
        data.push([parseFloat(shapVal.toFixed(4)), fi, parseFloat(featureVal.toFixed(3))]);
      }
    });
    return { features, data };
  })(),
};
