// ===== MAIN APPLICATION LOGIC =====

// ---- Global State ----
// 'demo' | 'real' — 預設實作模式。mode badge 暫時隱藏 (見 index.html),
// 沒有 UI 入口可切到 demo,但 mode 邏輯仍保留以便日後啟用。
let appMode = 'real';

document.addEventListener('DOMContentLoaded', () => {
  initToast();
  initSidebarDrawer();
  initModalA11y();
  initNavigation();
  initModeToggle();
  initLeaderboard();
  initWhatIfSimulator();
  initTrainingButton();
  initCsvUpload();
  initPreviewPagination();
  initCorrTopN();
  initShapSampleSelect();
  initSettings();
  initApiKeepAlive();
  initNotifications();
  initPipelinePage();
  loadTrainingHistory();
  // Auth — 讀 localStorage / 接 OAuth callback,後續所有 ApiClient fetch 自動帶 token
  if (typeof AuthClient !== 'undefined') AuthClient.init();

  // 從新介面跳回來的 ?need_login=1&next=... — 已登入直接跳回,否則開登入框
  initNewUiAuthHandshake();

  // 監聽 auth 狀態變化 — 登入 / 登出 / 切帳號都要全清 + 重抓新使用者的資料
  let _authInitial = true;
  window.addEventListener('auth:changed', async (ev) => {
    const wasInitial = _authInitial;
    _authInitial = false;
    // 登入 user 才需要抓 DB → 顯示載入遮罩 (guest 沒 DB 抓取,不用)
    const willFetchDb = ev.detail?.user && typeof ApiClient !== 'undefined' && ApiClient.enabled;
    if (willFetchDb) showGlobalLoading('載入你的資料集與訓練紀錄...');
    try {
      // 1. 清光上一位使用者的 in-memory 狀態
      clearAllUserState();
      // 2. 載入新使用者的訓練歷史 (per-user localStorage key)
      loadTrainingHistory();
      // 3. 抓 datasets
      await restoreUserDatasets();
      // 4. 登入 user → 從 DB 還原訓練歷史 + 模型 (跨瀏覽器/裝置可看到歷史)
      if (willFetchDb) {
        await hydrateUserHistoryFromDb();
      }
      // 5. 若有歷史,自動套用最新一筆 — dashboard 跟 leaderboard 才有東西可顯示
      if (_trainingHistory.length > 0) {
        _activeHistoryRunId = _trainingHistory[0].id;
        MLEngine.trainedModels = _trainingHistory[0].models || [];
      }
      // 6. 重新渲染當前頁面 (不然會卡在舊資料的 render)
      const visiblePage = document.querySelector('.page-section:not(.hidden)');
      if (visiblePage) {
        const pageId = visiblePage.id.replace('page-', '');
        if (!wasInitial && !ev.detail?.user && pageId !== 'dashboard') {
          navigateTo('dashboard');
        } else {
          renderPageCharts(pageId);
        }
      }
    } finally {
      hideGlobalLoading();
    }
  });
  // Show demo data on first load
  showDemoDataset();
  setTimeout(() => renderPageCharts('dashboard'), 100);
});

// ===== 全域載入遮罩 (登入後抓 DB 資料時顯示) =====
function showGlobalLoading(msg = '載入中...') {
  let el = document.getElementById('global-loading-overlay');
  if (!el) {
    el = document.createElement('div');
    el.id = 'global-loading-overlay';
    el.style.cssText = [
      'position:fixed', 'inset:0', 'z-index:9999',
      'display:flex', 'flex-direction:column', 'align-items:center', 'justify-content:center',
      'gap:16px', 'background:rgba(10,14,20,0.78)', 'backdrop-filter:blur(4px)',
    ].join(';');
    el.innerHTML = `
      <div style="width:42px;height:42px;border:3px solid rgba(99,102,241,0.25);border-top-color:#6366f1;border-radius:50%;animation:gl-spin 0.8s linear infinite"></div>
      <p id="global-loading-msg" style="color:#cbd5e1;font-size:14px;letter-spacing:0.02em"></p>
      <style>@keyframes gl-spin{to{transform:rotate(360deg)}}</style>
    `;
    document.body.appendChild(el);
  }
  const m = document.getElementById('global-loading-msg');
  if (m) m.textContent = msg;
  el.style.display = 'flex';
}

function hideGlobalLoading() {
  const el = document.getElementById('global-loading-overlay');
  if (el) el.style.display = 'none';
}

// ===== TOAST SYSTEM =====
// 全域 toast — 取代各區塊散落的 status span,所有非阻塞通知都走這裡。
// showToast('已儲存', { type: 'success', msg: '可選副標', duration: 3000 })
function initToast() {
  if (document.getElementById('toast-container')) return;
  const container = document.createElement('div');
  container.id = 'toast-container';
  container.setAttribute('role', 'status');
  container.setAttribute('aria-live', 'polite');
  document.body.appendChild(container);
}

function showToast(title, opts = {}) {
  const container = document.getElementById('toast-container');
  if (!container) return;
  const { type = 'info', msg = '', duration = 3500 } = opts;
  const iconHref = {
    success: '#i-check-circle',
    error:   '#i-warning',
    warning: '#i-warning',
    info:    '#i-info',
  }[type] || '#i-info';

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <svg class="toast-icon"><use href="${iconHref}"/></svg>
    <div class="toast-body">
      <div class="toast-title">${_escapeText(title)}</div>
      ${msg ? `<div class="toast-msg">${_escapeText(msg)}</div>` : ''}
    </div>
    <button class="toast-close" aria-label="關閉">
      <svg viewBox="0 0 24 24" width="16" height="16"><use href="#i-close"/></svg>
    </button>
  `;
  container.appendChild(toast);

  const dismiss = () => {
    if (toast.classList.contains('toast-leaving')) return;
    toast.classList.add('toast-leaving');
    toast.addEventListener('animationend', () => toast.remove(), { once: true });
  };
  toast.querySelector('.toast-close').addEventListener('click', dismiss);
  if (duration > 0) setTimeout(dismiss, duration);
  return { dismiss };
}

function _escapeText(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// ===== MOBILE SIDEBAR DRAWER =====
// 手機 / 平板 (<768px) 把 sidebar 收進抽屜,header 的漢堡按鈕負責開合,
// 點 backdrop 或選了 nav-item 都會關掉。
function initSidebarDrawer() {
  const sidebar = document.getElementById('sidebar');
  const toggle = document.getElementById('btn-sidebar-toggle');
  const backdrop = document.getElementById('drawer-backdrop');
  if (!sidebar || !toggle || !backdrop) return;

  const openDrawer = () => {
    sidebar.classList.add('open');
    backdrop.classList.add('open');
    toggle.setAttribute('aria-expanded', 'true');
    document.body.style.overflow = 'hidden';
  };
  const closeDrawer = () => {
    sidebar.classList.remove('open');
    backdrop.classList.remove('open');
    toggle.setAttribute('aria-expanded', 'false');
    document.body.style.overflow = '';
  };

  toggle.addEventListener('click', () => {
    sidebar.classList.contains('open') ? closeDrawer() : openDrawer();
  });
  backdrop.addEventListener('click', closeDrawer);
  // 點任一 nav-item 後自動收起 (手機才需要,desktop 沒影響因為 .open 在 ≥768px 不會被套用 transform)
  sidebar.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
      if (window.matchMedia('(max-width: 767px)').matches) closeDrawer();
    });
  });
  // 視窗放大到 desktop 時順手收掉 inline overflow lock
  window.addEventListener('resize', () => {
    if (window.matchMedia('(min-width: 768px)').matches) closeDrawer();
  });
}

// ===== MODAL A11Y HELPER =====
// 把每個 .hidden fixed inset-0 ... 的 modal 補上 dialog role / aria 屬性,
// 並掛 ESC 全域監聽 (關閉最上層那一個 modal)。
// 焦點陷阱:打開時把焦點丟進去,Tab 出去會繞回來。
function initModalA11y() {
  const modalIds = ['process-detail-modal', 'compare-modal', 'auth-modal'];
  modalIds.forEach(id => {
    const m = document.getElementById(id);
    if (!m) return;
    m.setAttribute('role', 'dialog');
    m.setAttribute('aria-modal', 'true');
    // 找第一個 h3 當 label
    const heading = m.querySelector('h3, h4');
    if (heading) {
      if (!heading.id) heading.id = `${id}-title`;
      m.setAttribute('aria-labelledby', heading.id);
    }
    // 觀察 hidden class 變化,開啟時把焦點丟進去
    const observer = new MutationObserver(() => {
      if (!m.classList.contains('hidden')) {
        _trapFocus(m);
      }
    });
    observer.observe(m, { attributes: true, attributeFilter: ['class'] });
  });

  // ESC 全域關閉最上層 modal
  document.addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;
    // 由上而下找第一個可見的 modal
    const visible = modalIds
      .map(id => document.getElementById(id))
      .filter(el => el && !el.classList.contains('hidden'))
      .pop();
    if (visible) visible.classList.add('hidden');
  });
}

function _trapFocus(modal) {
  const focusables = modal.querySelectorAll(
    'a[href], button:not([disabled]), textarea, input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'
  );
  if (!focusables.length) return;
  const first = focusables[0];
  const last = focusables[focusables.length - 1];
  // 焦點放到第一個可互動元素 (跳過視覺裝飾)
  setTimeout(() => first.focus(), 50);
  modal.addEventListener('keydown', e => {
    if (e.key !== 'Tab') return;
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault(); last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault(); first.focus();
    }
  });
}

// ===== API KEEP-ALIVE =====
// 每 30 秒 ping 一次後端 /api/health,避免 Render 免費方案 15 分鐘無請求就睡眠。
// 只在「使用 Python 後端 API」開啟時才 ping。
function initApiKeepAlive() {
  if (typeof ApiClient === 'undefined') return;
  setInterval(() => {
    if (ApiClient.enabled) {
      ApiClient.health().catch(() => {});  // 靜默 — keep-alive 失敗不打擾使用者
    }
  }, 30000);
}

// ===== NOTIFICATIONS (右上角鈴鐺) =====
// ===== GLOBAL STATUS INDICATOR (header dot + text) =====
// state: 'idle' | 'running' | 'success' | 'error' | 'warning'
const _STATUS_COLORS = {
  idle:    { ping: 'bg-dark-400',    dot: 'bg-dark-500',    animate: false },
  running: { ping: 'bg-primary-400', dot: 'bg-primary-500', animate: true  },
  success: { ping: 'bg-success-400', dot: 'bg-success-500', animate: true  },
  error:   { ping: 'bg-danger-400',  dot: 'bg-danger-500',  animate: false },
  warning: { ping: 'bg-warning-400', dot: 'bg-warning-500', animate: true  },
};

function setGlobalStatus(state, message) {
  const dotWrap = document.getElementById('global-status-dot');
  const text = document.getElementById('global-status-text');
  if (!dotWrap || !text) return;
  const c = _STATUS_COLORS[state] || _STATUS_COLORS.idle;
  // 重組兩層 dot:外層 ping (動畫光暈)、內層 solid
  dotWrap.innerHTML = `
    <span class="${c.animate ? 'animate-ping' : ''} absolute inline-flex h-full w-full rounded-full ${c.ping} opacity-75"></span>
    <span class="relative inline-flex rounded-full h-2 w-2 ${c.dot}"></span>
  `;
  if (message) text.textContent = message;
}

// ===== 切換使用者時清空所有 in-memory 狀態 =====
// auth:changed 觸發時 (登入 / 登出 / 切帳號) 都要先清,避免看到上一位的資料。
function clearAllUserState() {
  // ML 引擎 — 訓練好的模型
  if (typeof MLEngine !== 'undefined') {
    MLEngine.trainedModels = [];
    MLEngine.trainingHistory = [];
  }
  // 訓練歷史 (in-memory) + active 選擇 — 不動 localStorage,因為 key 已 per-user 隔離
  _trainingHistory = [];
  _activeHistoryRunId = null;
  // 通知 (session-only,不持久化)
  _notifications = [];
  // 預處理歷史
  ppHistory = [];
  ppLastPreprocessorId = null;
  ppLastFeatureColumns = [];
  // 資料引擎
  if (typeof DataEngine !== 'undefined') {
    DataEngine.datasets = [];
    DataEngine.currentDataset = null;
  }
  // 重設訓練按鈕狀態
  if (typeof setTrainBtnState === 'function') setTrainBtnState('idle');
  // 重設全域狀態指示器
  if (typeof setGlobalStatus === 'function') setGlobalStatus('idle', '系統就緒');
  // 隱藏通知 popover (若還開著)
  document.getElementById('notification-detail-popover')?.classList.add('hidden');
  // 重新渲染相關 UI
  renderNotifications();
  // 回到 dashboard (避免停在「實驗室 / 排行榜」看到空畫面卻不知所云)
  // 但不要在 init 階段 navigate (那時還沒切頁) — 只在 logout 後才跳
}

// ===== USER DATASET RESTORATION =====
// auth:changed 時呼叫 — 把後端記憶體裡屬於當前 user 的 datasets 抓回前端 DataEngine。
// 只先抓 metadata (stub),點擊時才 lazy fetch 完整 rows/analysis。
async function restoreUserDatasets() {
  if (typeof ApiClient === 'undefined' || !ApiClient.enabled) return;
  try {
    const list = await ApiClient.datasetList();
    // 清掉舊的 (避免切換使用者後看到上一位的)
    DataEngine.datasets = [];
    DataEngine.currentDataset = null;
    for (const meta of list) {
      DataEngine.datasets.push({
        id: meta.id,
        fileName: meta.fileName,
        rowCount: meta.rowCount,
        colCount: meta.colCount,
        headers: meta.headers || [],
        // stub — 點擊時才補齊
        data: [],
        columns: {},
        analysis: [],
        loadedAt: meta.loadedAt ? new Date(meta.loadedAt * 1000) : new Date(),
        _fromApi: true,
        _stub: true,
      });
    }
    // 自動把最新一筆當 currentDataset (但需要 fetch full data 才能 render)
    if (DataEngine.datasets.length > 0) {
      await hydrateDatasetIfStub(DataEngine.datasets[0]);
      DataEngine.currentDataset = DataEngine.datasets[0];
    }
    // 重新渲染 dataset 頁面跟相關 UI
    if (typeof renderDatasetPage === 'function') renderDatasetPage();
    // 也同步 file-info-bar / upload-card 顯示狀態
    const uploadCard = document.getElementById('upload-card');
    const infoBar = document.getElementById('file-info-bar');
    if (DataEngine.currentDataset) {
      if (uploadCard) uploadCard.classList.add('hidden');
      if (infoBar) {
        infoBar.classList.remove('hidden');
        const ds = DataEngine.currentDataset;
        const nameEl = document.getElementById('file-info-name');
        const metaEl = document.getElementById('file-info-meta');
        if (nameEl) nameEl.textContent = ds.fileName;
        if (metaEl) metaEl.textContent = `${ds.rowCount.toLocaleString()} 筆資料 | ${ds.colCount} 個欄位`;
      }
    } else {
      if (uploadCard) uploadCard.classList.remove('hidden');
      if (infoBar) infoBar.classList.add('hidden');
    }
  } catch (e) {
    console.warn('restoreUserDatasets 失敗:', e.message);
  }
}

// 登入後從 DB 還原訓練歷史 + 模型 — 這是「跨瀏覽器/裝置看到歷史」的關鍵 hook。
// 流程:
//   1. fetch /api/training-runs   → 取得 user 所有訓練紀錄 (含 results_summary)
//   2. fetch /api/models          → 取得所有 sklearn 模型的 bundle (含 metrics/featureImportance/...)
//   3. 把 models 依 trainingRunId 分組,套到對應的 training run 上
//   4. 重建 _trainingHistory (跟 localStorage 既有格式相容)
//   5. _trainingHistory 加進 DB 來源後,以 timestamp 排序
// Pipeline runs (engine='pipeline') 因為沒存 model bundles 到 DB,只還原 summary
// (沒法重做 SHAP/predict,但歷史紀錄看得到)
async function hydrateUserHistoryFromDb() {
  try {
    const baseUrl = ApiClient.baseUrl;
    const headers = {};
    if (typeof AuthClient !== 'undefined' && AuthClient.token) {
      headers['Authorization'] = `Bearer ${AuthClient.token}`;
    }
    const [runsResp, modelsResp] = await Promise.all([
      fetch(`${baseUrl}/api/training-runs?limit=100`, { headers }),
      fetch(`${baseUrl}/api/models?limit=500`, { headers }),
    ]);
    if (!runsResp.ok || !modelsResp.ok) {
      console.warn('[hydrate] training-runs / models fetch 失敗', runsResp.status, modelsResp.status);
      return;
    }
    const { runs } = await runsResp.json();
    const { models } = await modelsResp.json();

    // models 依 trainingRunId 分組
    const modelsByRun = {};
    for (const m of models) {
      const rid = m.trainingRunId;
      if (!rid) continue;
      (modelsByRun[rid] = modelsByRun[rid] || []).push({
        ...m.bundle,
        id: m.id,
        hyperparameters: m.hyperparameters || {},
        preprocessorId: m.preprocessorId,
        // 標記:這個 model 的重 blob (estimator/scaler/X_test) 在 DB,前端用 modelId 跟後端要
        _fromDb: true,
      });
    }

    // 把 DB runs 轉成 _trainingHistory 格式 (跟 localStorage push 出來的格式一致)
    const dbHistory = [];
    for (const r of runs) {
      const runModels = modelsByRun[r.id] || [];
      // sklearn: 若有對應 models 才能完整重建;pipeline: 沒 models,用 resultsSummary 重建顯示用 bundle
      let modelList = runModels;
      if (r.engine === 'pipeline' && modelList.length === 0 && r.resultsSummary?.perSource) {
        // pipeline run → 用 summary 假造 bundle (跟既有 danielResultToModel 邏輯一致)
        modelList = r.resultsSummary.perSource.map((p, i) => {
          // 優先用 perSource.taskType (新 schema),沒有就 fallback 到 run.taskType
          const _tt = p.taskType || r.taskType || 'classification';
          const _isReg = _tt === 'regression';
          // metric label:回歸用 R²,分類用 F1
          const _metricLabel = (p.metric || (_isReg ? 'R²' : 'F1')).toUpperCase();
          return {
            id: `daniel_db_${r.id}_${i}`,
            name: `[${p.label || p.source}] Pipeline (${(p.bestScore != null && p.scoreStack === p.bestScore) ? 'Stack' : 'Blend'})`,
            type: 'daniel_pipeline',
            taskType: _tt,
            targetName: r.target,
            dataSource: p.source,
            dataSourceLabel: p.label,
            metrics: _isReg ? {
              taskType: 'regression',
              testR2:   p.r2   ?? p.bestScore ?? 0,
              testRMSE: p.rmse ?? 0,
              testMAE:  p.mae  ?? 0,
              testScore: p.bestScore ?? p.r2 ?? 0,
              testScoreLabel: _metricLabel,
              scoreBlend: p.scoreBlend,
              scoreStack: p.scoreStack,
            } : {
              taskType: 'classification',
              testAccuracy: p.accuracy ?? 0,
              f1: p.f1 ?? 0,
              precision: p.f1 ?? 0,
              recall: p.f1 ?? 0,
              testScore: p.bestScore ?? 0,
              testScoreLabel: _metricLabel,
              scoreBlend: p.scoreBlend,
              scoreStack: p.scoreStack,
            },
            trainTime: (p.elapsedSec || 0) * 1000,
            _fromDb: true,
          };
        });
      }
      if (modelList.length === 0) continue;  // 沒模型沒法顯示,跳過

      // 找最佳 model
      modelList.sort((a, b) => (b.metrics?.testScore || 0) - (a.metrics?.testScore || 0));
      const best = modelList[0];
      const isReg = r.taskType === 'regression';
      const metric = isReg ? 'R²' : 'Accuracy';

      dbHistory.push({
        id: r.id,
        timestamp: r.startedAt ? r.startedAt * 1000 : Date.now(),
        datasetId: r.datasetId,
        // "(loading)" 是後端建立 run 時的佔位字串,不是真檔名 → 視為無效,往後 fallback
        datasetName: (r.datasetName && r.datasetName !== '(loading)')
          ? r.datasetName
          : (r.resultsSummary?.datasetName || '(未知)'),
        target: r.target,
        taskType: r.taskType,
        sources: r.sources || [],
        modelCount: modelList.length,
        metric,
        bestModel: { name: best.name, score: best.metrics?.testScore || 0 },
        options: r.options || {},
        models: modelList,
        engine: r.engine,
        _fromDb: true,
      });
    }

    // DB 為唯一真相 (authoritative):fetch 成功 → 直接用 DB 結果取代,不再 merge localStorage。
    // 這樣 `python tools/wipe_data.py --all` 清空 DB 後,使用者重新整理頁面就會自動看到乾淨畫面
    // (DB 0 筆 → dbHistory 0 筆 → 覆蓋掉殘留的 localStorage 舊歷史),不需要手動清 local。
    // ※ 只有 fetch「成功」才覆蓋;前面的 !runsResp.ok 已 return,所以網路暫時失敗不會誤刪歷史。
    const dbIds = new Set(dbHistory.map(h => h.id));
    _trainingHistory = dbHistory.sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));

    // 最近一筆的「完整版」(latestFull,含 testTrue/testPred 給圖表用) 若還在 DB,套回 index 0;
    // 若已被 wipe (id 不在 DB),清掉殘留的 latestFull 快取。
    const latestFull = _loadLatestFull();
    if (latestFull && dbIds.has(latestFull.id)) {
      const idx = _trainingHistory.findIndex(h => h.id === latestFull.id);
      if (idx >= 0) _trainingHistory[idx] = latestFull;
    } else if (latestFull) {
      try { localStorage.removeItem(_latestFullKey()); } catch (_) {}
    }

    // 把 DB 真相寫回 localStorage,讓下次離線載入也是乾淨的 (避免又冒出舊資料)
    _persistTrainingHistory();
    console.log(`[hydrate] DB 為準,還原 ${dbHistory.length} 個 training runs (已覆蓋本機殘留歷史)`);
  } catch (e) {
    console.warn('[hydrate] 從 DB 還原歷史失敗:', e.message);
  }
}


// stub dataset 點擊時 lazy fetch — 把 rows/analysis 補齊
async function hydrateDatasetIfStub(ds) {
  if (!ds || !ds._stub) return ds;
  try {
    const full = await ApiClient.datasetGet(ds.id);
    // 重建 columns 字典 (headers + data → {header: [values]})
    const columns = {};
    (full.headers || []).forEach((h, i) => {
      columns[h] = (full.data || []).map(row => row[i]);
    });
    Object.assign(ds, {
      data: full.data || [],
      columns,
      analysis: full.analysis || [],
      _correlation: full.correlation || null,
      _healthScore: full.healthScore || null,
      _processingLog: full.processingLog || null,
      _stub: false,
    });
  } catch (e) {
    console.warn('hydrateDataset 失敗:', e.message);
  }
  return ds;
}

// ===== TRAINING HISTORY (localStorage 持久化,最近 5 筆) =====
// 每筆: { id, timestamp, datasetId, datasetName, target, taskType,
//        sources, modelCount, bestModel: {name, score}, metric, options, models[] }
// Per-user 隔離:key 後綴帶 owner (u{user.id} 或 guest),登出不會看到別人的紀錄。
const _HISTORY_LIMIT = 5;
let _trainingHistory = [];

function _historyKey() {
  // 跟後端 store.owner_id() 同邏輯
  const u = (typeof AuthClient !== 'undefined' && AuthClient.user) ? `u${AuthClient.user.id}` : 'guest';
  return `automl_training_history_${u}`;
}

// ===== 實驗室表單設定的記憶 (切頁回來不要回復預設) =====
function _expFormKey() {
  const u = (typeof AuthClient !== 'undefined' && AuthClient.user) ? `u${AuthClient.user.id}` : 'guest';
  return `automl_exp_form_${u}`;
}

// 把目前實驗室表單狀態存進 localStorage (依 datasetId 分開存)
function saveExpFormState() {
  try {
    const ds = DataEngine.currentDataset;
    if (!ds) return;
    const get = (id) => document.getElementById(id);
    const checkedVals = (sel) => Array.from(document.querySelectorAll(sel)).filter(c => c.checked).map(c => c.value);
    const state = {
      engine: document.querySelector('input[name="exp-engine"]:checked')?.value,
      taskType: document.querySelector('input[name="exp-task-type"]:checked')?.value,
      timeSeries: get('exp-time-series')?.checked,
      srcRaw: get('exp-src-raw')?.checked,
      srcPp: get('exp-src-pp')?.checked,
      danielFast: get('exp-daniel-fast')?.checked,
      danielSkipDl: get('exp-daniel-skip-dl')?.checked,
      danielNoNas: get('exp-daniel-no-nas')?.checked,
      danielMetric: get('exp-daniel-metric')?.value,
      danielTimeLimit: get('exp-daniel-time-limit')?.value,
      target: get('exp-target-select')?.value,
      algos: checkedVals('#exp-algo-checkboxes .exp-algo-cb'),
      features: checkedVals('#exp-feature-checkboxes .exp-feat-cb'),
    };
    const all = JSON.parse(localStorage.getItem(_expFormKey()) || '{}');
    all[ds.id] = state;
    localStorage.setItem(_expFormKey(), JSON.stringify(all));
  } catch (e) { /* localStorage 滿/壞掉就放掉,不影響功能 */ }
}

// 從 localStorage 還原表單狀態。回傳 true 表示有套用過 (呼叫端可跳過預設)
function restoreExpFormState() {
  try {
    const ds = DataEngine.currentDataset;
    if (!ds) return false;
    const all = JSON.parse(localStorage.getItem(_expFormKey()) || '{}');
    const s = all[ds.id];
    if (!s) return false;
    const get = (id) => document.getElementById(id);
    const setRadio = (name, val) => { const r = document.querySelector(`input[name="${name}"][value="${val}"]`); if (r) r.checked = true; };
    const setCheck = (id, val) => { const el = get(id); if (el && typeof val === 'boolean') el.checked = val; };

    // 1. target 先設,並觸發 change 重建特徵 checkbox (全勾) → 之後再還原 feature 勾選
    const tsel = get('exp-target-select');
    if (s.target && tsel && Array.from(tsel.options).some(o => o.value === s.target)) {
      tsel.value = s.target;
      tsel.dispatchEvent(new Event('change'));
    }
    // 2. 還原 feature 勾選 (只動還存在的欄位)
    if (Array.isArray(s.features)) {
      document.querySelectorAll('#exp-feature-checkboxes .exp-feat-cb').forEach(cb => {
        cb.checked = s.features.includes(cb.value);
      });
    }
    // 3. 還原 algorithm 勾選
    if (Array.isArray(s.algos)) {
      document.querySelectorAll('#exp-algo-checkboxes .exp-algo-cb').forEach(cb => {
        cb.checked = s.algos.includes(cb.value);
      });
    }
    // 4. 其他控制項
    if (s.engine) setRadio('exp-engine', s.engine);
    if (s.taskType) setRadio('exp-task-type', s.taskType);
    setCheck('exp-time-series', s.timeSeries);
    setCheck('exp-src-raw', s.srcRaw);
    if (!get('exp-src-pp')?.disabled) setCheck('exp-src-pp', s.srcPp);  // pp 沒 preprocessor 時別硬開
    setCheck('exp-daniel-fast', s.danielFast);
    setCheck('exp-daniel-skip-dl', s.danielSkipDl);
    setCheck('exp-daniel-no-nas', s.danielNoNas);
    if (s.danielMetric && get('exp-daniel-metric')) get('exp-daniel-metric').value = s.danielMetric;
    if (s.danielTimeLimit != null && get('exp-daniel-time-limit')) get('exp-daniel-time-limit').value = s.danielTimeLimit;
    return true;
  } catch (e) { return false; }
}

// 最新一次訓練的完整資料 (含 testTrue/testPred/featureStats) 獨立存一份,
// 這樣即使主歷史被 slim 過,最新那筆的圖表/What-If 永遠能完整 render。
function _latestFullKey() {
  const u = (typeof AuthClient !== 'undefined' && AuthClient.user) ? `u${AuthClient.user.id}` : 'guest';
  return `automl_latest_full_${u}`;
}

function _saveLatestFull(entry) {
  try { localStorage.setItem(_latestFullKey(), JSON.stringify(entry)); }
  catch (e) {
    // 一筆都裝不下就放棄,不影響主歷史
    try { localStorage.removeItem(_latestFullKey()); } catch (_) {}
  }
}

function _loadLatestFull() {
  try {
    const raw = localStorage.getItem(_latestFullKey());
    return raw ? JSON.parse(raw) : null;
  } catch (e) { return null; }
}

function loadTrainingHistory() {
  try {
    const raw = localStorage.getItem(_historyKey());
    _trainingHistory = raw ? JSON.parse(raw) : [];
  } catch (e) {
    _trainingHistory = [];
  }
  // 把最新一次的完整版本還原回去 (主歷史可能被 slim 過)
  const latestFull = _loadLatestFull();
  if (latestFull && _trainingHistory.length > 0 && _trainingHistory[0].id === latestFull.id) {
    _trainingHistory[0] = latestFull;
  }
}

function _persistTrainingHistory() {
  const key = _historyKey();
  const trySave = (data) => {
    try { localStorage.setItem(key, JSON.stringify(data)); return true; }
    catch (e) { return false; }
  };

  // 重要:in-memory _trainingHistory 永遠保留全部筆數 (5 筆),不因為 localStorage
  // 寫不下就丟掉。寫 localStorage 時用 slim 副本,讓筆數能完整持久化。

  // Level 0: 完整版
  if (trySave(_trainingHistory)) return;
  console.warn('[訓練歷史] localStorage 容量不足,啟動瘦身模式...');

  // Level 1: 先砍最大的 What-If 用資料 (featureStats / means / stds)
  // 散點圖 / 殘差圖 / 特徵重要性都還能用
  const slim1 = _trainingHistory.map(h => ({
    ...h,
    models: (h.models || []).map(m => {
      const { featureStats, means, stds, ...rest } = m;
      return rest;
    }),
  }));
  if (trySave(slim1)) return;

  // Level 2: 再砍 testTrue / testPred (散點圖 / 殘差圖會 fallback 到佔位訊息)
  const slim2 = slim1.map(h => ({
    ...h,
    models: h.models.map(m => {
      const { testTrue, testPred, ...rest } = m;
      return rest;
    }),
  }));
  if (trySave(slim2)) return;

  // Level 3: 再砍 featureImportance / featureNames (特徵重要性 fallback)
  const slim3 = slim2.map(h => ({
    ...h,
    models: h.models.map(m => {
      const { featureImportance, featureNames, X_test_df, ...rest } = m;
      return rest;
    }),
  }));
  if (trySave(slim3)) return;

  // Level 3: 只留 metadata + bestModel (連 models 陣列都丟)
  const metaOnly = _trainingHistory.map(h => ({
    id: h.id, timestamp: h.timestamp,
    datasetId: h.datasetId, datasetName: h.datasetName,
    target: h.target, taskType: h.taskType,
    sources: h.sources, modelCount: h.modelCount,
    metric: h.metric, bestModel: h.bestModel,
    options: h.options,
    models: [],
  }));
  if (trySave(metaOnly)) return;

  // Level 4: 投降清空
  try { localStorage.removeItem(key); } catch (e) {}
}

function pushTrainingHistory(entry) {
  entry.id = `run_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
  _trainingHistory.unshift(entry);
  if (_trainingHistory.length > _HISTORY_LIMIT) _trainingHistory = _trainingHistory.slice(0, _HISTORY_LIMIT);
  _persistTrainingHistory();
  // 把這筆最新的完整版另存一份 — 之後 reload / 登出登入後還能保留 testTrue/featureStats
  _saveLatestFull(entry);
  _activeHistoryRunId = entry.id; // 新訓練即「現在」
}

// 排行榜 / 洞察頁顯示中的歷史 runId — 切換選單時更新
let _activeHistoryRunId = null;

// 三層級聯歷史選擇器:資料集 → target → 訓練紀錄。
// 用法:HTML 給一個 <div class="history-cascade"></div>,呼叫 renderHistoryCascade(container)。
function renderHistoryCascade(container) {
  if (!container) return;

  const variant = container.dataset.variant || 'full'; // 'compact' = 排行榜縮小版
  const selectCls = variant === 'compact'
    ? 'bg-dark-800 border border-dark-600 rounded px-2 py-1 text-xs focus:border-primary-500 outline-none'
    : 'bg-dark-800 border border-dark-600 rounded-lg px-3 py-2 text-sm focus:border-primary-500 outline-none';
  const labelCls = variant === 'compact' ? 'text-[10px] text-dark-500' : 'text-xs text-dark-400';

  if (_trainingHistory.length === 0) {
    container.innerHTML = `<p class="${variant === 'compact' ? 'text-[11px]' : 'text-xs'} text-dark-500">尚無歷史訓練紀錄 — 訓練後會顯示在這裡</p>`;
    return;
  }

  // 永遠重建 markup (簡單,不會有 cache 狀態問題)
  container.innerHTML = `
    <div class="flex items-center gap-2 flex-wrap">
      <div class="flex items-center gap-1">
        <span class="${labelCls}">資料集</span>
        <select class="cascade-dataset ${selectCls} max-w-[12rem]"></select>
      </div>
      <span class="text-dark-600">›</span>
      <div class="flex items-center gap-1">
        <span class="${labelCls}">target</span>
        <select class="cascade-target ${selectCls} max-w-[10rem]"></select>
      </div>
      <span class="text-dark-600">›</span>
      <div class="flex items-center gap-1">
        <span class="${labelCls}">訓練</span>
        <select class="cascade-run ${selectCls} max-w-[20rem]"></select>
      </div>
      <button class="history-info-btn w-6 h-6 rounded-full bg-dark-700 hover:bg-primary-500/20 text-dark-400 hover:text-primary-300 text-[12px] flex items-center justify-center transition-colors" title="查看訓練詳情">ⓘ</button>
    </div>
  `;

  const dsSelect  = container.querySelector('.cascade-dataset');
  const tgSelect  = container.querySelector('.cascade-target');
  const runSelect = container.querySelector('.cascade-run');
  const infoBtn   = container.querySelector('.history-info-btn');

  const fmtScore = (r) => {
    // bestModel.score 在 hydration / pushTrainingHistory 兩條路徑都可能是 undefined
    const s = r.bestModel?.score;
    if (typeof s !== 'number' || !isFinite(s)) return r.taskType === 'regression' ? 'R²=—' : 'Acc=—';
    return r.taskType === 'regression' ? `R²=${s.toFixed(4)}` : `Acc=${(s * 100).toFixed(1)}%`;
  };
  const fmtTime = (ts) => {
    const d = new Date(ts);
    return `${String(d.getMonth()+1).padStart(2,'0')}/${String(d.getDate()).padStart(2,'0')} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
  };

  // ---- 內部 populate helpers ----
  const refillRunSelect = (datasetName, target) => {
    const runs = _trainingHistory
      .filter(h => h.datasetName === datasetName && h.target === target)
      .sort((a, b) => {
        if (b.modelCount !== a.modelCount) return b.modelCount - a.modelCount;
        return (b.bestModel?.score || 0) - (a.bestModel?.score || 0);
      });
    runSelect.innerHTML = '';
    runs.forEach(r => {
      const opt = document.createElement('option');
      opt.value = r.id;
      opt.textContent = `${fmtTime(r.timestamp)} · ${r.modelCount} 模型 · ${fmtScore(r)}`;
      runSelect.appendChild(opt);
    });
    // 預設選 activeHistoryRunId (若還屬於這個資料集+target),否則選第一筆
    if (runs.find(r => r.id === _activeHistoryRunId)) {
      runSelect.value = _activeHistoryRunId;
    } else if (runs.length > 0) {
      runSelect.value = runs[0].id;
    }
    if (infoBtn) infoBtn.dataset.runId = runSelect.value || '';
  };

  const refillTargetSelect = (datasetName) => {
    const targets = [...new Set(
      _trainingHistory.filter(h => h.datasetName === datasetName).map(h => h.target)
    )];
    tgSelect.innerHTML = '';
    targets.forEach(t => {
      const opt = document.createElement('option');
      opt.value = t;
      opt.textContent = t;
      tgSelect.appendChild(opt);
    });
    // 預設選 activeRun 的 target (若還在),否則第一個
    const activeRun = _trainingHistory.find(h => h.id === _activeHistoryRunId);
    if (activeRun && activeRun.datasetName === datasetName && targets.includes(activeRun.target)) {
      tgSelect.value = activeRun.target;
    } else {
      tgSelect.value = targets[0];
    }
    refillRunSelect(datasetName, tgSelect.value);
  };

  // ---- 第一層:資料集 ----
  const datasets = [...new Set(_trainingHistory.map(h => h.datasetName))];
  dsSelect.innerHTML = '';
  datasets.forEach(ds => {
    const opt = document.createElement('option');
    opt.value = ds;
    opt.textContent = ds;
    dsSelect.appendChild(opt);
  });
  const activeRun = _trainingHistory.find(h => h.id === _activeHistoryRunId);
  dsSelect.value = activeRun ? activeRun.datasetName : datasets[0];
  refillTargetSelect(dsSelect.value);

  // ---- listeners (每次 render 都重綁,因為元素是新的) ----
  dsSelect.addEventListener('change', () => {
    refillTargetSelect(dsSelect.value);
    if (runSelect.value) applyHistoricalRun(runSelect.value);
  });
  tgSelect.addEventListener('change', () => {
    refillRunSelect(dsSelect.value, tgSelect.value);
    if (runSelect.value) applyHistoricalRun(runSelect.value);
  });
  runSelect.addEventListener('change', () => {
    if (infoBtn) infoBtn.dataset.runId = runSelect.value;
    applyHistoricalRun(runSelect.value);
  });
  if (infoBtn) _wireHistoryInfoBtn(infoBtn);
}

// 給 ⓘ 按鈕綁 hover 事件 (顯示訓練詳情 popover)
function _wireHistoryInfoBtn(btn) {
  btn.addEventListener('mouseenter', e => {
    const runId = e.currentTarget.dataset.runId;
    const run = _trainingHistory.find(h => h.id === runId);
    if (!run) return;
    const popover = document.getElementById('notification-detail-popover');
    if (!popover) return;
    if (_notifDetailHideTimer) { clearTimeout(_notifDetailHideTimer); _notifDetailHideTimer = null; }

    const isReg = run.taskType === 'regression';
    popover.innerHTML = renderNotificationDetailContent({
      type: 'training',
      dataset: run.datasetName,
      target: run.target,
      taskType: run.taskType,
      sources: run.sources || ['raw'],
      modelCount: run.modelCount,
      metric: run.metric || (isReg ? 'R²' : 'Accuracy'),
      topModels: (run.models || []).slice(0, 5).map(m => ({
        name: m.name.replace(/^\[(原始|預處理)\]\s*/, ''),
        source: m.dataSource === 'preprocessed' ? '預處理' : '原始',
        score: isReg ? m.metrics.testR2 : m.metrics.testAccuracy,
        trainTime: m.trainTime || 0,
      })),
      totalTime: (run.models || []).reduce((s, m) => s + (m.trainTime || 0), 0),
    });
    popover.classList.remove('hidden');

    // 定位 — 放按鈕左下,空間不夠時改右
    requestAnimationFrame(() => {
      const rect = btn.getBoundingClientRect();
      const popRect = popover.getBoundingClientRect();
      const margin = 8;
      let left = rect.left - popRect.width - margin;
      let top  = rect.bottom + margin;
      if (left < margin) left = rect.right + margin;
      if (top + popRect.height > window.innerHeight - margin) {
        top = Math.max(margin, window.innerHeight - popRect.height - margin);
      }
      popover.style.left = `${left}px`;
      popover.style.top = `${top}px`;
    });

    popover.onmouseenter = () => { if (_notifDetailHideTimer) { clearTimeout(_notifDetailHideTimer); _notifDetailHideTimer = null; } };
    popover.onmouseleave = () => scheduleHideNotificationDetail();
  });
  btn.addEventListener('mouseleave', () => scheduleHideNotificationDetail());
}

function clearTrainingHistory() {
  _trainingHistory = [];
  try { localStorage.removeItem(_historyKey()); } catch (e) {}
}

// 把指定的歷史紀錄套用到 MLEngine.trainedModels,並重新渲染目前頁面相關區塊
function applyHistoricalRun(runId) {
  const run = _trainingHistory.find(h => h.id === runId);
  if (!run) return;
  _activeHistoryRunId = runId;
  MLEngine.trainedModels = run.models || [];
  // 重新渲染當前頁面
  const visiblePage = document.querySelector('.page-section:not(.hidden)');
  if (visiblePage) {
    const pageId = visiblePage.id.replace('page-', '');
    if (pageId === 'leaderboard') renderRealLeaderboard();
    else if (pageId === 'insights') renderRealInsights();
    else if (pageId === 'dashboard') {
      renderPerformanceTrend(); renderTaskDistribution(); updateDashboardRealMetrics();
    }
  }
  notify('已載入歷史訓練', `${run.datasetName || 'Dataset'} — ${run.modelCount} 模型`, 'info');
}

let _notifications = [];   // { title, message, type, time, read, details? }

function notify(title, message, type = 'info', details = null) {
  _notifications.unshift({
    title, message, type, details,
    time: new Date(),
    read: false,
  });
  if (_notifications.length > 30) _notifications = _notifications.slice(0, 30);
  renderNotifications();
}

function renderNotifications() {
  const list = document.getElementById('notification-list');
  const badge = document.getElementById('notification-badge');
  if (!list || !badge) return;

  const unread = _notifications.filter(n => !n.read).length;
  if (unread > 0) {
    badge.textContent = unread > 9 ? '9+' : String(unread);
    badge.classList.remove('hidden');
    badge.classList.add('flex');
  } else {
    badge.classList.add('hidden');
    badge.classList.remove('flex');
  }

  if (_notifications.length === 0) {
    list.innerHTML = '<p class="px-4 py-6 text-center text-xs text-dark-500">目前沒有通知</p>';
    return;
  }

  const iconMap = {
    success: '<span class="text-success-400">✓</span>',
    error: '<span class="text-danger-400">✗</span>',
    warning: '<span class="text-warning-400">⚠</span>',
    info: '<span class="text-primary-400">ℹ</span>',
  };
  list.innerHTML = _notifications.map((n, idx) => {
    const t = n.time;
    const ts = `${String(t.getHours()).padStart(2,'0')}:${String(t.getMinutes()).padStart(2,'0')}`;
    const infoBtn = n.details
      ? `<button class="notif-info-btn shrink-0 ml-1 w-5 h-5 rounded-full bg-dark-700 hover:bg-primary-500/20 text-dark-400 hover:text-primary-300 text-[11px] flex items-center justify-center transition-colors" data-notif-idx="${idx}" title="查看詳情">ⓘ</button>`
      : '';
    return `
      <div class="px-4 py-3 border-b border-dark-800/50 ${n.read ? 'opacity-60' : 'bg-dark-800/30'}">
        <div class="flex items-start gap-2">
          <span class="text-sm mt-0.5">${iconMap[n.type] || iconMap.info}</span>
          <div class="flex-1 min-w-0">
            <div class="flex items-start justify-between gap-1">
              <p class="text-xs font-semibold text-dark-100">${escapeHtml(n.title)}</p>
              ${infoBtn}
            </div>
            <p class="text-[11px] text-dark-400 mt-0.5">${escapeHtml(n.message)}</p>
            <p class="text-[10px] text-dark-600 mt-1">${ts}</p>
          </div>
        </div>
      </div>`;
  }).join('');

  // 綁定 ⓘ 按鈕 hover handler
  list.querySelectorAll('.notif-info-btn').forEach(btn => {
    btn.addEventListener('mouseenter', e => showNotificationDetail(e.currentTarget));
    btn.addEventListener('mouseleave', () => scheduleHideNotificationDetail());
  });
}

// ===== 通知詳情 Popover =====
let _notifDetailHideTimer = null;

function showNotificationDetail(btnEl) {
  const popover = document.getElementById('notification-detail-popover');
  if (!popover) return;
  const idx = parseInt(btnEl.dataset.notifIdx, 10);
  const notif = _notifications[idx];
  if (!notif || !notif.details) return;

  // ★ 把 popover 移到 <body> 底下 — 不然它的祖先 header 有 backdrop-blur,
  //   會讓 position:fixed 變成相對 header 定位 (不是視窗),導致位置偏掉 / 被切。
  if (popover.parentElement !== document.body) {
    document.body.appendChild(popover);
  }

  // 取消任何待執行的隱藏
  if (_notifDetailHideTimer) { clearTimeout(_notifDetailHideTimer); _notifDetailHideTimer = null; }

  popover.innerHTML = renderNotificationDetailContent(notif.details);
  popover.classList.remove('hidden');

  // 定位:盡量放在按鈕左側,空間不夠時放右側、放下方
  const rect = btnEl.getBoundingClientRect();
  const popRect = popover.getBoundingClientRect();
  const margin = 8;
  // 預設往左 (因為通知 panel 在右上,左邊有空間)
  let left = rect.left - popRect.width - margin;
  let top  = rect.top;
  if (left < margin) {
    // 左側空間不夠 → 改放右側
    left = rect.right + margin;
  }
  // 防止超出底邊
  if (top + popRect.height > window.innerHeight - margin) {
    top = Math.max(margin, window.innerHeight - popRect.height - margin);
  }
  popover.style.left = `${left}px`;
  popover.style.top = `${top}px`;

  // popover 自己也要可 hover,游標移上去不會被隱藏
  popover.onmouseenter = () => { if (_notifDetailHideTimer) { clearTimeout(_notifDetailHideTimer); _notifDetailHideTimer = null; } };
  popover.onmouseleave = () => scheduleHideNotificationDetail();
}

function scheduleHideNotificationDetail() {
  // 給 150ms 緩衝,讓游標從按鈕移到 popover 的瞬間不會被立即關閉
  if (_notifDetailHideTimer) clearTimeout(_notifDetailHideTimer);
  _notifDetailHideTimer = setTimeout(() => {
    const popover = document.getElementById('notification-detail-popover');
    if (popover) popover.classList.add('hidden');
    _notifDetailHideTimer = null;
  }, 150);
}

function renderNotificationDetailContent(details) {
  if (!details || details.type !== 'training') return '';

  const taskLabel = details.taskType === 'regression' ? '迴歸' : '分類';
  const sourcesLabel = (details.sources || []).map(s =>
    s === 'raw' ? '原始' : s === 'preprocessed' ? '預處理' : s
  ).join(' + ') || '—';
  const totalSec = ((details.totalTime || 0) / 1000).toFixed(1);

  const topRows = (details.topModels || []).map((m, i) => {
    const scoreStr = details.metric === 'R²'
      ? m.score.toFixed(4)
      : (m.score * 100).toFixed(2) + '%';
    const srcBadge = m.source === '預處理'
      ? '<span class="text-[9px] px-1 py-0.5 rounded bg-accent-500/15 text-accent-400 border border-accent-500/20">預處理</span>'
      : '<span class="text-[9px] px-1 py-0.5 rounded bg-primary-500/15 text-primary-400 border border-primary-500/20">原始</span>';
    const rankIcon = i === 0
      ? '<span class="text-warning-400">🏆</span>'
      : `<span class="text-dark-500">${i + 1}</span>`;

    // 超參數行 — 有資料才插
    const hp = m.hyperparameters;
    let hpRow = '';
    if (hp && Object.keys(hp).length) {
      // 過濾掉值為 null / undefined / 空 list 的;依 key 排序;最多顯示 12 個
      const entries = Object.entries(hp)
        .filter(([_, v]) => v !== null && v !== undefined && !(Array.isArray(v) && v.length === 0))
        .sort((a, b) => a[0].localeCompare(b[0]))
        .slice(0, 12);
      if (entries.length) {
        const items = entries.map(([k, v]) => {
          const valStr = typeof v === 'number'
            ? (Number.isInteger(v) ? String(v) : v.toFixed(4))
            : (typeof v === 'object' ? JSON.stringify(v) : String(v));
          const valTrunc = valStr.length > 24 ? valStr.slice(0, 22) + '…' : valStr;
          return `<span class="inline-block text-[10px] px-1.5 py-0.5 rounded bg-dark-800 text-dark-300 mr-1 mb-1" title="${escapeHtml(k)} = ${escapeHtml(valStr)}"><span class="text-dark-500">${escapeHtml(k)}</span>=<span class="font-mono">${escapeHtml(valTrunc)}</span></span>`;
        }).join('');
        hpRow = `
          <tr class="bg-dark-900/30">
            <td></td>
            <td colspan="4" class="py-1 px-1">
              <details>
                <summary class="text-[10px] text-dark-400 cursor-pointer hover:text-dark-200 select-none">超參數 (${entries.length})</summary>
                <div class="mt-1 leading-relaxed">${items}</div>
              </details>
            </td>
          </tr>
        `;
      }
    }

    return `
      <tr class="border-b border-dark-800/40">
        <td class="py-1 pr-2 text-center w-6">${rankIcon}</td>
        <td class="py-1 pr-2">${srcBadge}</td>
        <td class="py-1 px-1 font-mono text-dark-200 text-[11px] truncate" style="max-width:140px" title="${escapeHtml(m.name)}">${escapeHtml(m.name)}</td>
        <td class="py-1 px-1 text-right font-mono text-success-400 text-[11px]">${scoreStr}</td>
        <td class="py-1 pl-2 text-right text-[10px] text-dark-500">${m.trainTime.toFixed(0)}ms</td>
      </tr>
      ${hpRow}
    `;
  }).join('');

  return `
    <div class="text-[11px] space-y-2">
      <div class="flex items-center gap-2 pb-2 border-b border-dark-800">
        <span class="text-success-400">✓</span>
        <span class="font-semibold text-dark-100">訓練詳情</span>
      </div>
      <div class="grid grid-cols-2 gap-x-3 gap-y-1.5">
        <div><span class="text-dark-500">資料集</span><br><span class="font-mono text-dark-100">${escapeHtml(details.dataset || '—')}</span></div>
        <div><span class="text-dark-500">目標欄位</span><br><span class="font-mono text-dark-100">${escapeHtml(details.target || '—')}</span></div>
        <div><span class="text-dark-500">任務型別</span><br><span class="text-dark-100">${taskLabel}</span></div>
        <div><span class="text-dark-500">總訓練時間</span><br><span class="font-mono text-dark-100">${totalSec} 秒</span></div>
        <div class="col-span-2"><span class="text-dark-500">資料來源</span><br><span class="text-dark-100">${sourcesLabel}</span></div>
      </div>
      <div class="border-t border-dark-800 pt-2">
        <p class="text-[10px] text-dark-500 mb-1">Top ${(details.topModels || []).length} / 共 ${details.modelCount} 個模型 (${details.metric})</p>
        <table class="w-full">${topRows}</table>
      </div>
    </div>
  `;
}

function initNotifications() {
  const btn = document.getElementById('btn-notification');
  const panel = document.getElementById('notification-panel');
  const clearBtn = document.getElementById('btn-clear-notifications');
  if (!btn || !panel) return;

  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    const opening = panel.classList.contains('hidden');
    panel.classList.toggle('hidden');
    if (opening) {
      // 打開時把所有通知標為已讀
      _notifications.forEach(n => n.read = true);
      renderNotifications();
    }
  });
  // 點面板外面關閉
  document.addEventListener('click', (e) => {
    if (!panel.classList.contains('hidden') && !panel.contains(e.target) && e.target !== btn) {
      panel.classList.add('hidden');
    }
  });
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      _notifications = [];
      renderNotifications();
    });
  }
  renderNotifications();
}

// ===== MODE TOGGLE =====
// Mode 徽章已移到 top bar #mode-badge,讓使用者一眼看到目前是 Demo 還是 Real。
function initModeToggle() {
  const badge = document.getElementById('mode-badge');
  if (!badge) {
    console.warn('[mode] #mode-badge 不在 DOM 中,切換按鈕綁不上');
    return;
  }
  // 統一用 type=button 避免任何 form-submit 行為
  badge.setAttribute('type', 'button');
  badge.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    setAppMode(appMode === 'demo' ? 'real' : 'demo', { showToast: true });
  });
  updateModeUI();  // 初始 sync 文字 / 顏色 / nav 顯示
}

// 切換模式的單一入口 — 改 appMode + 同步 UI + 自動導頁
function setAppMode(next, opts = {}) {
  if (next !== 'demo' && next !== 'real') return;
  const prev = appMode;
  appMode = next;
  console.log(`[mode] ${prev} → ${next}`);
  updateModeUI();

  // 切換模式時,若是同頁就重新渲染圖表 (demo/real 內容會切換);不強制跳頁
  const cur = document.querySelector('.page-section:not(.hidden)');
  if (cur) {
    renderPageCharts(cur.id.replace('page-', ''));
  }

  if (opts.showToast && typeof window.showToast === 'function') {
    window.showToast(
      next === 'demo' ? '已切換至測試模式' : '已切換至實作模式',
      {
        type: 'info',
        msg: next === 'demo'
          ? '以下數據為示範用途,可探索完整 UI 與圖表'
          : '需要先上傳資料 / 訓練模型,所有結果都是真實的',
      },
    );
  }
}

function updateModeUI() {
  const badge = document.getElementById('mode-badge');
  const label = document.getElementById('mode-badge-label');
  if (!badge || !label) return;
  badge.dataset.mode = appMode;
  label.textContent = appMode === 'demo' ? '測試模式' : '實作模式';
  // 測試模式專屬 nav (Daniel pipeline) 只在 demo 顯示
  // 用自訂 display 屬性而不是 .hidden,避開 .nav-item { display: flex } 跟 .hidden 的 specificity 平手問題
  document.querySelectorAll('.nav-mode-demo').forEach(el => {
    el.style.display = (appMode === 'demo') ? '' : 'none';
  });
}

// ===== NAVIGATION =====
const PAGE_NAMES = {
  dashboard: '總覽儀表板',
  datasets: '數據集管理',
  preprocessing: '預處理',
  pipeline: 'AutoML Pipeline',
  experiments: '實驗室',
  leaderboard: '模型排行榜',
  insights: '洞察與決策',
  settings: '系統設定',
};

function navigateTo(page) {
  // ⚠ 不要在換頁時 abort SHAP — 後端 sklearn PermutationExplainer 無法被取消,
  // abort 只會讓 client 丟掉 response,server 還在算;切回來又會 fire 新請求 →
  // 兩個並行。改用 reqKey 防重入(loadShapFigures 開頭擋),server 算完就回。
  document.querySelectorAll('.page-section').forEach(s => s.classList.add('hidden'));
  const target = document.getElementById(`page-${page}`);
  if (target) target.classList.remove('hidden');
  document.querySelectorAll('.nav-item').forEach(item => {
    item.classList.toggle('active', item.dataset.page === page);
  });
  document.getElementById('breadcrumb-current').textContent = PAGE_NAMES[page] || page;
  setTimeout(() => {
    renderPageCharts(page);
    // 多丟一次 resize 給保險:處理「容器剛從 hidden 切回 visible,ECharts 還沒重算尺寸」
    setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
  }, 50);
}

function initNavigation() {
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', e => {
      const page = item.dataset.page;
      if (page) {
        e.preventDefault();
        navigateTo(page);
      }
      // 沒 data-page (例如 「新介面」切換按鈕) → 讓瀏覽器照原生 href 跳轉
    });
  });
}

function renderPageCharts(page) {
  // Toggle demo/real content for pages with overlays
  updatePageModeVisibility(page);

  switch (page) {
    case 'dashboard':
      if (appMode === 'real' && (MLEngine.trainedModels.length > 0 || _trainingHistory.length > 0)) {
        // Real mode 且有訓練紀錄 → 用真實資料 (歷史 + 通知 + 最近實驗)
        renderRealPerformanceTrend();
        renderRealTaskDistribution();
        updateDashboardRealMetrics();          // metrics + 最近實驗
        renderSystemNotificationsCard();
      } else if (appMode === 'demo' || MLEngine.trainedModels.length > 0) {
        // 純 demo 模式 → mock 圖
        renderPerformanceTrend();
        renderTaskDistribution();
      }
      break;
    case 'datasets':
      renderDatasetPage();
      break;
    case 'preprocessing':
      renderPreprocessingPage();
      break;
    case 'experiments':
      if (appMode === 'demo') {
        renderOptimizationHistory();
      } else {
        renderRealExperimentsPage();
      }
      break;
    case 'leaderboard':
      if (appMode === 'real' && MLEngine.trainedModels.length > 0) {
        renderRealLeaderboard();
      }
      break;
    case 'insights':
      if (appMode === 'demo') {
        renderFeatureImportance();
        renderShapWaterfall(0);
        renderShapBeeswarm();
        renderWhatIfGauge(0.42);
      } else if (MLEngine.trainedModels.length > 0) {
        renderRealInsights();
      } else if (_shapRequestedModelId) {
        // Pipeline best model path — 沒有 sklearn 模型但有 pipeline bestModelId
        _renderPipelineShapOnly(_shapRequestedModelId);
      } else {
        // 完全沒有模型 — 顯示引導空白狀態
        _renderInsightsEmptyState();
      }
      break;
    case 'deployments':
      if (appMode === 'demo') renderApiUsage();
      else renderRealDeployments();
      break;
    case 'pipeline':
      renderPipelinePage();
      break;
  }
}

// Toggle visibility of demo content vs real empty state for each page
function updatePageModeVisibility(page) {
  const pages = ['dashboard', 'experiments', 'leaderboard', 'insights', 'deployments'];
  const targetPage = page || null;
  const pagesToUpdate = targetPage ? [targetPage] : pages;

  pagesToUpdate.forEach(p => {
    if (!pages.includes(p)) return;
    const demoEl = document.getElementById(`${p}-demo-content`);
    const realEmptyEl = document.getElementById(`${p}-real-empty`);
    if (!demoEl || !realEmptyEl) return;

    if (appMode === 'demo') {
      demoEl.classList.remove('hidden');
      realEmptyEl.classList.add('hidden');
      // Hide real content for insights
      const realContent = document.getElementById(`${p}-real-content`);
      if (realContent) realContent.classList.add('hidden');
    } else {
      demoEl.classList.add('hidden');
      const realContent = document.getElementById(`${p}-real-content`);
      const hasModels = MLEngine.trainedModels.length > 0 || _trainingHistory.length > 0;
      if ((p === 'leaderboard' || p === 'dashboard') && MLEngine.trainedModels.length > 0) {
        realEmptyEl.classList.add('hidden');
        demoEl.classList.remove('hidden');
      } else if (p === 'insights' && MLEngine.trainedModels.length > 0) {
        realEmptyEl.classList.add('hidden');
        if (realContent) realContent.classList.remove('hidden');
      } else if (p === 'deployments' && hasModels) {
        // deployments: use the real-content div (injected by renderRealDeployments)
        realEmptyEl.classList.add('hidden');
        demoEl.classList.add('hidden');
        if (realContent) realContent.classList.remove('hidden');
      } else {
        realEmptyEl.classList.remove('hidden');
        if (realContent) realContent.classList.add('hidden');
      }
    }
  });
}

// ===== CSV UPLOAD =====
function initCsvUpload() {
  const zone = document.getElementById('upload-zone');
  const fileInput = document.getElementById('csv-file-input');
  const btnChoose = document.getElementById('btn-choose-file');
  const btnReupload = document.getElementById('btn-reupload');
  const btnExport = document.getElementById('btn-export-csv');

  if (!zone || !fileInput) return;

  // Click to choose
  if (btnChoose) btnChoose.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
  zone.addEventListener('click', () => fileInput.click());

  // File input change
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
  });

  // Drag and drop
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  // Re-upload
  if (btnReupload) btnReupload.addEventListener('click', () => {
    fileInput.value = '';
    fileInput.click();
  });

  // Export
  if (btnExport) btnExport.addEventListener('click', () => {
    if (!DataEngine.currentDataset) return;
    const csv = DataEngine.exportCleanedCSV();
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'cleaned_' + DataEngine.currentDataset.fileName;
    a.click();
    URL.revokeObjectURL(url);
  });
}

function handleFile(file) {
  if (!file) return;

  // Switch to real mode automatically
  if (appMode === 'demo') {
    setAppMode('real');
  }

  // Show loading state
  const zone = document.getElementById('upload-zone');
  zone.innerHTML = `
    <div class="flex items-center justify-center gap-3 py-4">
      <div class="w-5 h-5 border-2 border-primary-400 border-t-transparent rounded-full animate-spin"></div>
      <span class="text-sm text-primary-400">正在解析並分析 <strong>${escapeHtml(file.name)}</strong>...</span>
    </div>
  `;

  const showInfoBar = (dataset) => {
    document.getElementById('upload-card').classList.add('hidden');
    const infoBar = document.getElementById('file-info-bar');
    infoBar.classList.remove('hidden');
    document.getElementById('file-info-name').textContent = file.name;
    const sizeMB = (file.size / 1024 / 1024).toFixed(2);
    const sizeKB = (file.size / 1024).toFixed(1);
    const sizeStr = file.size > 1048576 ? `${sizeMB} MB` : `${sizeKB} KB`;
    document.getElementById('file-info-meta').textContent = `${dataset.rowCount.toLocaleString()} 筆資料 | ${dataset.colCount} 個欄位 | ${sizeStr}`;
    renderDatasetPage();
  };

  // ---- Python API path ----
  if (typeof ApiClient !== 'undefined' && ApiClient.enabled) {
    ApiClient.preprocess(file)
      .then(apiDs => showInfoBar(adoptApiDataset(apiDs)))
      .catch(err => {
        zone.innerHTML = `
          <div class="flex items-center justify-center gap-3 py-4">
            <svg class="w-6 h-6 text-danger-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
            <span class="text-sm text-danger-400">後端 API 失敗: ${escapeHtml(err.message)}</span>
          </div>
          <button onclick="resetUploadZone()" class="mt-3 px-4 py-1.5 bg-dark-700 hover:bg-dark-600 rounded-lg text-xs">重試</button>
        `;
      });
    return;
  }

  // ---- Local JS path (current behavior) ----
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const text = e.target.result;
      const dataset = DataEngine.loadCSV(text, file.name);
      showInfoBar(dataset);

    } catch (err) {
      zone.innerHTML = `
        <div class="flex items-center justify-center gap-3 py-4">
          <svg class="w-6 h-6 text-danger-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
          <span class="text-sm text-danger-400">解析失敗: ${escapeHtml(err.message)}</span>
        </div>
        <button onclick="resetUploadZone()" class="mt-3 px-4 py-1.5 bg-dark-700 hover:bg-dark-600 rounded-lg text-xs">重試</button>
      `;
    }
  };
  reader.onerror = () => {
    zone.innerHTML = `
      <div class="flex items-center justify-center gap-3 py-4">
        <svg class="w-6 h-6 text-danger-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
        <span class="text-sm text-danger-400">無法讀取檔案</span>
      </div>
    `;
  };
  reader.readAsText(file);
}

function resetUploadZone() {
  const zone = document.getElementById('upload-zone');
  zone.innerHTML = `
    <div class="w-16 h-16 mx-auto mb-4 rounded-2xl bg-primary-500/10 flex items-center justify-center group-hover:scale-110 transition-transform">
      <svg class="w-8 h-8 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/></svg>
    </div>
    <p class="text-lg font-semibold mb-1">拖拽 CSV 檔案至此或點擊上傳</p>
    <p class="text-sm text-dark-400">支援 CSV / TSV 格式，系統將自動分析數據結構</p>
    <div class="flex items-center justify-center gap-4 mt-4">
      <button id="btn-choose-file" class="px-4 py-2 bg-primary-600 hover:bg-primary-500 rounded-lg text-sm font-medium transition-colors">選擇檔案</button>
    </div>
  `;
  // Re-bind choose button
  document.getElementById('btn-choose-file').addEventListener('click', e => {
    e.stopPropagation();
    document.getElementById('csv-file-input').click();
  });
  document.getElementById('upload-card').classList.remove('hidden');
  document.getElementById('file-info-bar').classList.add('hidden');
}
window.resetUploadZone = resetUploadZone;

// ===== DATASET LIST =====
function renderDatasetList() {
  const listBar = document.getElementById('dataset-list-bar');
  const listEl = document.getElementById('dataset-list');
  if (!listBar || !listEl) return;

  if (DataEngine.datasets.length === 0) {
    listBar.classList.add('hidden');
    return;
  }

  listBar.classList.remove('hidden');
  listEl.innerHTML = '';

  DataEngine.datasets.forEach(ds => {
    const isActive = DataEngine.currentDataset && DataEngine.currentDataset.id === ds.id;
    const div = document.createElement('div');
    div.className = `flex items-center gap-3 p-3 rounded-lg border transition-all cursor-pointer ${isActive ? 'bg-primary-500/10 border-primary-500' : 'bg-dark-800 border-dark-600 hover:border-dark-500'}`;
    div.innerHTML = `
      <div class="w-8 h-8 rounded-lg ${isActive ? 'bg-primary-500/20' : 'bg-dark-700'} flex items-center justify-center flex-shrink-0">
        <svg class="w-4 h-4 ${isActive ? 'text-primary-400' : 'text-dark-400'}" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
      </div>
      <div class="flex-1 min-w-0">
        <p class="text-sm font-medium truncate ${isActive ? 'text-primary-300' : ''}">${escapeHtml(ds.fileName)}</p>
        <p class="text-xs text-dark-400">${ds.rowCount.toLocaleString()} 筆 | ${ds.colCount} 欄位</p>
      </div>
      ${isActive ? '<span class="text-xs text-primary-400 font-medium flex-shrink-0">使用中</span>' : ''}
      <button class="ds-remove-btn p-1 rounded hover:bg-dark-600 transition-colors flex-shrink-0" data-id="${ds.id}" title="移除">
        <svg class="w-4 h-4 text-dark-500 hover:text-danger-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
      </button>
    `;
    // Click to switch dataset (stub 需 lazy fetch)
    div.addEventListener('click', async (e) => {
      if (e.target.closest('.ds-remove-btn')) return;
      await hydrateDatasetIfStub(ds);
      DataEngine.switchDataset(ds.id);
      renderDatasetPage();
    });
    // Remove button — 先呼叫後端 DELETE (級聯刪 preprocessor + model),再清前端 state
    div.querySelector('.ds-remove-btn').addEventListener('click', async (e) => {
      e.stopPropagation();
      const ok = confirm(`確定要移除「${ds.fileName}」?\n相關預處理結果與訓練模型也會一併刪除。`);
      if (!ok) return;
      try {
        if (typeof ApiClient !== 'undefined' && ApiClient.enabled && ds._fromApi) {
          await ApiClient.datasetDelete(ds.id);
        }
      } catch (err) {
        // 不擋本機刪除 — 後端刪不到 (可能 stub 已過期) 還是允許清前端
        console.warn('後端刪除失敗,仍會清前端:', err.message);
      }
      // 清前端:dataset + 關聯的 pp/training history
      DataEngine.removeDataset(ds.id);
      ppHistory = ppHistory.filter(p => p.datasetId !== ds.id);
      _trainingHistory = _trainingHistory.filter(h => h.datasetId !== ds.id);
      if (_activeHistoryRunId && !_trainingHistory.find(h => h.id === _activeHistoryRunId)) {
        _activeHistoryRunId = _trainingHistory[0]?.id || null;
        MLEngine.trainedModels = _activeHistoryRunId
          ? (_trainingHistory.find(h => h.id === _activeHistoryRunId).models || [])
          : [];
      }
      _persistTrainingHistory();
      renderDatasetPage();
      notify('已移除資料集', `${ds.fileName} 連同預處理 / 訓練結果已刪除`, 'info');
    });
    listEl.appendChild(div);
  });

  // "Upload new" button
  const btnUploadNew = document.getElementById('btn-upload-new');
  btnUploadNew.onclick = () => {
    document.getElementById('upload-card').classList.remove('hidden');
    document.getElementById('csv-file-input').click();
  };
}

// ===== RENDER DATASET PAGE (depending on mode) =====
function renderDatasetPage() {
  if (appMode === 'demo') {
    renderDemoDataset();
  } else {
    renderRealDataset();
  }
}

// ---- Demo dataset rendering (uses MOCK data) ----
function showDemoDataset() {
  // Pre-populate on first load so tables are not empty
  renderDemoDataset();
}

function renderDemoDataset() {
  const analysis = document.getElementById('dataset-analysis');
  const empty = document.getElementById('dataset-empty-state');
  const uploadCard = document.getElementById('upload-card');
  const infoBar = document.getElementById('file-info-bar');
  const listBar = document.getElementById('dataset-list-bar');

  // In demo mode: hide upload and dataset list, show analysis with mock data
  uploadCard.classList.add('hidden');
  if (listBar) listBar.classList.add('hidden');
  infoBar.classList.remove('hidden');
  analysis.classList.remove('hidden');
  empty.classList.add('hidden');

  document.getElementById('file-info-name').textContent = 'demo_customer_churn.csv (Demo)';
  document.getElementById('file-info-meta').textContent = '12,450 筆資料 | 12 個欄位 | 2.4 MB | Demo 展示數據';

  // Render mock columns table
  renderMockColumnsTable();
  // Render mock health radar
  renderHealthScore();
  document.getElementById('health-overall-score').textContent = '85';
  document.getElementById('health-badge').textContent = '良好';
  document.getElementById('health-badge').className = 'badge badge-success';
  document.getElementById('col-count-label').textContent = `共 ${MOCK.datasetColumns.length} 個欄位`;

  // Render mock data preview
  renderMockDataPreview();
  // Mock correlation (just show placeholder)
  renderMockCorrelation();
  // // Mock column detail (commented out)
  // renderMockColDetail();
  // Mock process log
  renderMockProcessLog();
}

function renderMockColumnsTable() {
  const tbody = document.getElementById('dataset-columns-table');
  if (!tbody) return;
  tbody.innerHTML = '';
  MOCK.datasetColumns.forEach(col => {
    const tr = document.createElement('tr');
    tr.className = 'border-b border-dark-700/30 hover:bg-dark-800/30 transition-colors';
    const maxDist = Math.max(...col.dist);
    const sparklineHtml = `<div class="mini-distribution">${col.dist.map(v => {
      const h = Math.max(2, (v / maxDist) * 22);
      return `<div class="sparkline-bar" style="height:${h}px"></div>`;
    }).join('')}</div>`;
    let statusBadge;
    switch (col.status) {
      case 'ok': statusBadge = '<span class="badge badge-success">正常</span>'; break;
      case 'filled': statusBadge = '<span class="badge badge-primary">已填補</span>'; break;
      case 'encoded': statusBadge = '<span class="badge badge-primary">已編碼</span>'; break;
      case 'decomposed': statusBadge = '<span class="badge badge-primary">已拆解</span>'; break;
      case 'warning': statusBadge = '<span class="badge badge-warning">需審核</span>'; break;
      case 'target': statusBadge = '<span class="badge" style="background:rgba(139,92,246,0.1);color:#a78bfa">目標變數</span>'; break;
      default: statusBadge = '<span class="badge badge-success">正常</span>';
    }
    tr.innerHTML = `
      <td class="py-2.5 px-3"><span class="font-mono text-accent-400 text-xs">${col.name}</span></td>
      <td class="py-2.5 px-3"><span class="text-dark-400 text-xs">${col.type} (${col.dtype})</span></td>
      <td class="py-2.5 px-3">${sparklineHtml}</td>
      <td class="py-2.5 px-3"><span class="text-xs ${col.missing > 0 ? 'text-warning-400' : 'text-dark-500'}">${col.missing > 0 ? col.missing + '%' : '-'}</span></td>
      <td class="py-2.5 px-3"><span class="text-xs ${col.outliers > 0 ? 'text-danger-400' : 'text-dark-500'}">${col.outliers > 0 ? col.outliers + ' 筆' : '-'}</span></td>
      <td class="py-2.5 px-3">${statusBadge}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ===== DATA PREVIEW WITH COLUMN PAGINATION =====
const COL_PAGE_SIZE = 10; // columns per page
let previewColPage = 0;
let previewColTotal = 0;
let previewAllHeaders = [];
let previewAllRows = [];
let previewIsReal = false;

function renderMockDataPreview() {
  const cols = MOCK.datasetColumns.map(c => c.name);
  const rows = [];
  for (let i = 0; i < 8; i++) {
    rows.push(cols.map(c => {
      if (c === 'customer_id') return 1001 + i;
      if (c === 'age') return 25 + Math.floor(Math.random() * 40);
      if (c === 'gender') return Math.random() > 0.5 ? 'M' : 'F';
      if (c === 'income') return (30000 + Math.floor(Math.random() * 80000)).toLocaleString();
      if (c === 'tenure_days') return Math.floor(Math.random() * 1500);
      if (c === 'monthly_spending') return (500 + Math.floor(Math.random() * 8000)).toLocaleString();
      if (c === 'purchase_date') return '2024-0' + (1 + Math.floor(Math.random() * 9)) + '-' + String(1 + Math.floor(Math.random() * 28)).padStart(2, '0');
      if (c === 'complaints') return Math.floor(Math.random() * 5);
      if (c === 'contract_type') return ['monthly', 'yearly', 'two_year'][Math.floor(Math.random() * 3)];
      if (c === 'satisfaction_score') return (1 + Math.random() * 4).toFixed(1);
      if (c === 'last_login_days') return Math.floor(Math.random() * 90);
      if (c === 'churn') return Math.random() > 0.7 ? 1 : 0;
      return '-';
    }));
  }
  previewAllHeaders = cols;
  previewAllRows = rows;
  previewIsReal = false;
  previewColPage = 0;
  previewColTotal = cols.length;
  renderPreviewPage();
  document.getElementById('preview-row-label').textContent = 'Demo 展示數據 (8 筆)';
}

function renderPreviewPage() {
  const thead = document.getElementById('preview-thead');
  const tbody = document.getElementById('preview-tbody');
  if (!thead || !tbody) return;

  const totalPages = Math.ceil(previewColTotal / COL_PAGE_SIZE);
  const start = previewColPage * COL_PAGE_SIZE;
  const end = Math.min(start + COL_PAGE_SIZE, previewColTotal);
  const pageCols = previewAllHeaders.slice(start, end);

  // Header
  const rowNumTh = previewIsReal ? '<th class="py-2 px-2 font-medium text-left whitespace-nowrap text-dark-500">#</th>' : '';
  thead.innerHTML = `<tr class="text-dark-400 text-[10px] border-b border-dark-700/50">${rowNumTh}${pageCols.map(c => `<th class="py-2 px-2 font-medium text-left whitespace-nowrap">${escapeHtml(c)}</th>`).join('')}</tr>`;

  // Body
  tbody.innerHTML = '';
  previewAllRows.forEach((row, i) => {
    const tr = document.createElement('tr');
    tr.className = 'border-b border-dark-700/20 hover:bg-dark-800/30';
    const rowNumTd = previewIsReal ? `<td class="py-1.5 px-2 whitespace-nowrap text-dark-500">${i + 1}</td>` : '';
    tr.innerHTML = rowNumTd + row.slice(start, end).map(v =>
      `<td class="py-1.5 px-2 whitespace-nowrap max-w-[200px] truncate">${escapeHtml(String(v))}</td>`
    ).join('');
    tbody.appendChild(tr);
  });

  // Pagination controls
  const label = document.getElementById('col-page-label');
  const prevBtn = document.getElementById('btn-col-prev');
  const nextBtn = document.getElementById('btn-col-next');
  const controls = document.getElementById('col-page-controls');

  if (totalPages <= 1) {
    controls.style.display = 'none';
  } else {
    controls.style.display = 'flex';
    label.textContent = `欄位 ${start + 1}-${end} / ${previewColTotal}`;
    prevBtn.disabled = previewColPage === 0;
    nextBtn.disabled = previewColPage >= totalPages - 1;
  }
}

function initPreviewPagination() {
  const prevBtn = document.getElementById('btn-col-prev');
  const nextBtn = document.getElementById('btn-col-next');
  if (prevBtn) prevBtn.addEventListener('click', () => { if (previewColPage > 0) { previewColPage--; renderPreviewPage(); } });
  if (nextBtn) nextBtn.addEventListener('click', () => {
    const totalPages = Math.ceil(previewColTotal / COL_PAGE_SIZE);
    if (previewColPage < totalPages - 1) { previewColPage++; renderPreviewPage(); }
  });
}

// ===== CORRELATION MATRIX TOP-N =====
let currentCorrData = null;

function getCorrTopN() {
  const sel = document.getElementById('corr-top-n');
  return sel ? parseInt(sel.value, 10) || 0 : 0;
}

function initCorrTopN() {
  const sel = document.getElementById('corr-top-n');
  if (!sel) return;
  sel.addEventListener('change', () => {
    if (currentCorrData) {
      renderCorrelationHeatmap(currentCorrData, getCorrTopN());
    }
  });
}

function renderMockCorrelation() {
  const names = ['age', 'income', 'tenure_days', 'spending', 'complaints', 'satisfaction', 'login_days'];
  const matrix = [];
  for (let i = 0; i < names.length; i++) {
    for (let j = 0; j < names.length; j++) {
      let r = i === j ? 1 : parseFloat((Math.random() * 1.6 - 0.8).toFixed(3));
      if (j < i) r = matrix.find(m => m[0] === j && m[1] === i)[2];
      matrix.push([i, j, r]);
    }
  }
  currentCorrData = { names, matrix };
  document.getElementById('corr-top-toggle-wrap').style.display = 'none';
  document.getElementById('corr-feature-count').textContent = `${names.length} 個數值欄位`;
  // Reset height
  const el = document.getElementById('chart-correlation');
  if (el) el.style.height = '320px';
  renderCorrelationHeatmap(currentCorrData, 0);
}

// function renderMockColDetail() {
//   const select = document.getElementById('col-detail-select');
//   if (!select) return;
//   select.innerHTML = '';
//   MOCK.datasetColumns.forEach(col => {
//     const opt = document.createElement('option');
//     opt.value = col.name;
//     opt.textContent = col.name;
//     select.appendChild(opt);
//   });
//   renderMockColDetailChart(MOCK.datasetColumns[0]);
//   select.addEventListener('change', () => {
//     const col = MOCK.datasetColumns.find(c => c.name === select.value);
//     if (col) renderMockColDetailChart(col);
//   });
// }

// function renderMockColDetailChart(col) {
//   const fakeColInfo = {
//     type: col.dtype === 'category' ? 'categorical' : 'numeric',
//     distribution: col.dist.map((v, i) => ({ label: `Bin ${i+1}`, count: v * 50 })),
//     topValues: col.dtype === 'category' ? [{ value: 'A', count: col.dist[0] * 50 }, { value: 'B', count: col.dist[1] * 50 }] : [],
//   };
//   renderColumnDetail(fakeColInfo);
//   const statsEl = document.getElementById('col-detail-stats');
//   if (statsEl) {
//     if (col.dtype !== 'category' && col.dtype !== 'datetime') {
//       statsEl.innerHTML = `<div class="grid grid-cols-4 gap-2 text-xs text-center">
//         <div><p class="text-dark-500">平均</p><p class="font-mono text-dark-200">${(Math.random() * 100).toFixed(1)}</p></div>
//         <div><p class="text-dark-500">中位數</p><p class="font-mono text-dark-200">${(Math.random() * 100).toFixed(1)}</p></div>
//         <div><p class="text-dark-500">標準差</p><p class="font-mono text-dark-200">${(Math.random() * 30).toFixed(1)}</p></div>
//         <div><p class="text-dark-500">範圍</p><p class="font-mono text-dark-200">0 ~ ${Math.floor(Math.random() * 200)}</p></div>
//       </div>`;
//     } else {
//       statsEl.innerHTML = `<div class="text-xs text-dark-500 text-center">類別型欄位，共 ${col.dist.filter(d => d > 0).length} 個類別</div>`;
//     }
//   }
// }

function renderMockProcessLog() {
  const container = document.getElementById('auto-process-log');
  if (!container) return;
  const logs = [
    { action: 'fill_missing', colName: 'age', selectedMethod: 'median',
      _mock: { missingPct: 5.2, missingCount: 26 } },
    { action: 'fill_missing', colName: 'salary', selectedMethod: 'mean',
      _mock: { missingPct: 2.8, missingCount: 14 } },
    { action: 'handle_outlier', colName: 'income', selectedMethod: 'keep',
      _mock: { outlierCount: 3 } },
    { action: 'handle_outlier', colName: 'transaction_amount', selectedMethod: 'clip',
      _mock: { outlierCount: 7 } },
    { action: 'drop_column', colName: 'notes', selectedMethod: 'drop_column',
      _mock: { missingPct: 82.1 } },
    { action: 'encode', colName: 'gender', selectedMethod: 'onehot',
      _mock: { uniqueCount: 3 } },
    { action: 'encode', colName: 'city', selectedMethod: 'label',
      _mock: { uniqueCount: 18 } },
    { action: 'decompose_date', colName: 'purchase_date', selectedMethod: 'decompose',
      _mock: { dateMin: '2021-01', dateMax: '2024-12' } },
    { action: 'interaction', colName: 'age × income', selectedMethod: 'multiply' },
  ];
  renderProcessLogItems(container, logs);
}

// ---- Real dataset rendering (uses DataEngine) ----
function renderRealDataset() {
  const analysis = document.getElementById('dataset-analysis');
  const empty = document.getElementById('dataset-empty-state');
  const uploadCard = document.getElementById('upload-card');
  const infoBar = document.getElementById('file-info-bar');
  // Render dataset list if we have any
  renderDatasetList();

  if (!DataEngine.currentDataset) {
    // No data uploaded yet: show upload zone
    uploadCard.classList.remove('hidden');
    infoBar.classList.add('hidden');
    analysis.classList.add('hidden');
    empty.classList.remove('hidden');
    return;
  }

  // Data available
  uploadCard.classList.add('hidden');
  infoBar.classList.remove('hidden');
  analysis.classList.remove('hidden');
  empty.classList.add('hidden');

  const ds = DataEngine.currentDataset;

  // ---- Columns Table ----
  renderRealColumnsTable(ds.analysis);
  document.getElementById('col-count-label').textContent = `共 ${ds.colCount} 個欄位`;

  // ---- Health Score ----
  const health = DataEngine.computeHealthScore();
  if (health) {
    renderRealHealthScore(health);
    document.getElementById('health-overall-score').textContent = health.overall;
    const badge = document.getElementById('health-badge');
    if (health.overall >= 80) { badge.textContent = '良好'; badge.className = 'badge badge-success'; }
    else if (health.overall >= 60) { badge.textContent = '普通'; badge.className = 'badge badge-warning'; }
    else { badge.textContent = '需改善'; badge.className = 'badge badge-danger'; }
  }

  // ---- Data Preview ----
  renderRealDataPreview(ds);

  // ---- Correlation Heatmap ----
  currentCorrData = DataEngine.computeCorrelationMatrix();
  if (currentCorrData) {
    const n = currentCorrData.names.length;
    const wrap = document.getElementById('corr-top-toggle-wrap');
    const countLabel = document.getElementById('corr-feature-count');
    if (n > 10) {
      wrap.style.display = 'flex';
      countLabel.textContent = `共 ${n} 個數值欄位`;
    } else {
      wrap.style.display = 'none';
      countLabel.textContent = `${n} 個數值欄位`;
    }
    const topN = n > 10 ? getCorrTopN() : 0;
    renderCorrelationHeatmap(currentCorrData, topN);
  } else {
    document.getElementById('corr-top-toggle-wrap').style.display = 'none';
    document.getElementById('corr-feature-count').textContent = '僅數值型欄位';
    const chart = initChart('chart-correlation');
    if (chart) { chart.clear(); chart.setOption({ title: { text: '數值型欄位不足，無法計算相關性', left: 'center', top: 'center', textStyle: { color: '#475569', fontSize: 13 } } }); }
  }

  // ---- Column Detail Selector ----
  // // initRealColDetailSelect(ds.analysis); // commented out

  // ---- Auto Process Log ----
  const logs = DataEngine.generateProcessingLog();
  const container = document.getElementById('auto-process-log');
  renderProcessLogItems(container, logs);
}

function renderRealColumnsTable(analysisArr) {
  const tbody = document.getElementById('dataset-columns-table');
  if (!tbody) return;
  tbody.innerHTML = '';

  analysisArr.forEach(col => {
    const tr = document.createElement('tr');
    tr.className = 'border-b border-dark-700/30 hover:bg-dark-800/30 transition-colors cursor-pointer';
    // tr.addEventListener('click', () => {
    //   const sel = document.getElementById('col-detail-select');
    //   if (sel) { sel.value = col.name; sel.dispatchEvent(new Event('change')); }
    // });

    // Mini sparkline
    let sparklineHtml = '';
    if (col.type === 'numeric' && col.distribution.length > 0) {
      const maxC = Math.max(...col.distribution.map(b => b.count));
      sparklineHtml = `<div class="mini-distribution">${col.distribution.slice(0, 15).map(b => {
        const h = Math.max(2, maxC > 0 ? (b.count / maxC) * 22 : 2);
        return `<div class="sparkline-bar" style="height:${h}px"></div>`;
      }).join('')}</div>`;
    } else if ((col.type === 'categorical' || col.type === 'boolean') && col.topValues.length > 0) {
      const maxC = Math.max(...col.topValues.map(v => v.count));
      sparklineHtml = `<div class="mini-distribution">${col.topValues.slice(0, 10).map(v => {
        const h = Math.max(2, maxC > 0 ? (v.count / maxC) * 22 : 2);
        return `<div class="sparkline-bar" style="height:${h}px"></div>`;
      }).join('')}</div>`;
    } else {
      sparklineHtml = '<span class="text-dark-600 text-xs">—</span>';
    }

    // Type label
    const typeLabels = { numeric: '數值', categorical: '類別', boolean: '布林', datetime: '日期', text: '文字', unknown: '未知' };

    tr.innerHTML = `
      <td class="py-2.5 px-3"><span class="font-mono text-accent-400 text-xs">${escapeHtml(col.name)}</span></td>
      <td class="py-2.5 px-3"><span class="text-dark-400 text-xs">${typeLabels[col.type] || col.type} (${col.dtype})</span></td>
      <td class="py-2.5 px-3">${sparklineHtml}</td>
      <td class="py-2.5 px-3"><span class="text-xs ${col.missingPct > 0 ? (col.missingPct > 20 ? 'text-danger-400' : 'text-warning-400') : 'text-dark-500'}">${col.missingPct > 0 ? col.missingPct + '%' : '-'}</span></td>
      <td class="py-2.5 px-3"><span class="text-xs ${col.outlierCount > 0 ? 'text-danger-400' : 'text-dark-500'}">${col.outlierCount > 0 ? col.outlierCount + ' 筆' : '-'}</span></td>
      <td class="py-2.5 px-3"><span class="text-xs text-dark-400">${col.uniqueCount.toLocaleString()}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function renderRealDataPreview(ds) {
  previewAllHeaders = ds.headers;
  previewAllRows = ds.data.slice(0, 100);
  previewIsReal = true;
  previewColPage = 0;
  previewColTotal = ds.headers.length;
  renderPreviewPage();
  document.getElementById('preview-row-label').textContent = `前 ${Math.min(100, ds.rowCount)} 筆 / 共 ${ds.rowCount.toLocaleString()} 筆`;
}

// function initRealColDetailSelect(analysisArr) {
//   const select = document.getElementById('col-detail-select');
//   if (!select) return;
//   select.innerHTML = '';
//   analysisArr.forEach(col => {
//     const opt = document.createElement('option');
//     opt.value = col.name;
//     opt.textContent = `${col.name} (${col.type})`;
//     select.appendChild(opt);
//   });
//   const renderDetail = () => {
//     const col = analysisArr.find(c => c.name === select.value);
//     if (!col) return;
//     renderColumnDetail(col);
//     renderColDetailStats(col);
//   };
//   const newSelect = select.cloneNode(true);
//   select.parentNode.replaceChild(newSelect, select);
//   newSelect.addEventListener('change', renderDetail);
//   if (analysisArr.length > 0) {
//     newSelect.value = analysisArr[0].name;
//     const firstCol = analysisArr[0];
//     renderColumnDetail(firstCol);
//     renderColDetailStats(firstCol);
//   }
// }

// function renderColDetailStats(col) {
//   const statsEl = document.getElementById('col-detail-stats');
//   if (!statsEl) return;
//   if (col.type === 'numeric' && col.stats) {
//     const s = col.stats;
//     statsEl.innerHTML = `<div class="grid grid-cols-5 gap-2 text-xs text-center">
//       <div><p class="text-dark-500">平均</p><p class="font-mono text-dark-200">${s.mean}</p></div>
//       <div><p class="text-dark-500">中位數</p><p class="font-mono text-dark-200">${s.median}</p></div>
//       <div><p class="text-dark-500">標準差</p><p class="font-mono text-dark-200">${s.std}</p></div>
//       <div><p class="text-dark-500">最小</p><p class="font-mono text-dark-200">${s.min}</p></div>
//       <div><p class="text-dark-500">最大</p><p class="font-mono text-dark-200">${s.max}</p></div>
//     </div>`;
//   } else if ((col.type === 'categorical' || col.type === 'boolean' || col.type === 'text') && col.topValues && col.topValues.length > 0) {
//     statsEl.innerHTML = `<div class="text-xs text-dark-400 px-1 space-y-1">
//       <p>唯一值: <strong class="text-dark-200">${col.uniqueCount}</strong></p>
//       <p>最多: <strong class="text-accent-400">${escapeHtml(col.topValues[0].value)}</strong> (${col.topValues[0].pct}%)</p>
//     </div>`;
//   } else if (col.type === 'datetime' && col.stats) {
//     statsEl.innerHTML = `<div class="grid grid-cols-3 gap-2 text-xs text-center">
//       <div><p class="text-dark-500">最早日期</p><p class="font-mono text-dark-200">${col.stats.min}</p></div>
//       <div><p class="text-dark-500">最晚日期</p><p class="font-mono text-dark-200">${col.stats.max}</p></div>
//       <div><p class="text-dark-500">有效筆數</p><p class="font-mono text-dark-200">${(col.stats.count || col.totalCount - col.missingCount).toLocaleString()}</p></div>
//     </div>`;
//   } else {
//     statsEl.innerHTML = `<div class="text-xs text-dark-500 text-center">無額外統計資訊</div>`;
//   }
// }

// ---- Shared helpers ----
let currentProcessLogs = [];

// Category definitions: order, icon, color, label
const PROC_CATEGORIES = [
  { action: 'dataset_info',   icon: 'M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z',                                                                                                       label: '資料集概覽',  color: 'primary' },
  { action: 'sentinel',       icon: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z',                            label: '哨兵值偵測',  color: 'warning' },
  { action: 'fill_missing',   icon: 'M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10',       label: '缺失值處理',  color: 'warning' },
  { action: 'handle_outlier', icon: 'M13 10V3L4 14h7v7l9-11h-7z',                                                                                                                                       label: '異常值處理',  color: 'danger' },
  { action: 'drop_column',    icon: 'M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16',                                   label: '高缺失欄位',  color: 'danger' },
  { action: 'encode',         icon: 'M7 20l4-16m2 16l4-16M6 9h14M4 15h14',                                                                                                                               label: '類別編碼',    color: 'primary' },
  { action: 'decompose_date', icon: 'M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z',                                                                           label: '日期特徵工程', color: 'accent' },
  { action: 'interaction',    icon: 'M4 6h16M4 12h16M4 18h7',                                                                                                                                             label: '交互特徵',    color: 'accent' },
  { action: 'api_warning',    icon: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z',                            label: '後端警告',    color: 'warning' },
];

function renderProcessLogItems(container, logs) {
  if (!container) return;
  container.innerHTML = '';
  currentProcessLogs = logs;

  if (logs.length === 0) {
    container.innerHTML = '<p class="text-sm text-dark-500 text-center py-4">暫無處理記錄</p>';
    return;
  }

  // Group logs by action
  const groups = {};
  logs.forEach((log, idx) => {
    const key = log.action || 'other';
    if (!groups[key]) groups[key] = [];
    groups[key].push({ ...log, _idx: idx });
  });

  // Render each category in defined order
  PROC_CATEGORIES.forEach(cat => {
    const items = groups[cat.action];
    if (!items || items.length === 0) return;

    const section = document.createElement('div');
    section.className = 'proc-category mb-4';

    // Category header (clickable to collapse)
    // ⚠️ 用「完整字面 class」對照表,不要用變數拼 (bg-${color}-500/8)。
    //    Tailwind build 只掃得到字面字串,拼出來的 class 會被 purge 掉 → 標頭掉色。
    const catStyle = {
      warning: { bg: 'bg-warning-500/8', text: 'text-warning-400' },
      danger:  { bg: 'bg-danger-500/8',  text: 'text-danger-400' },
      primary: { bg: 'bg-primary-500/8', text: 'text-primary-400' },
      accent:  { bg: 'bg-accent-500/8',  text: 'text-accent-400' },
      success: { bg: 'bg-success-500/8', text: 'text-success-400' },
    };
    const cs = catStyle[cat.color] || { bg: 'bg-dark-700/40', text: 'text-dark-300' };

    section.innerHTML = `
      <div class="proc-cat-header flex items-center gap-2.5 px-3 py-2 rounded-lg ${cs.bg} cursor-pointer select-none" data-cat="${cat.action}">
        <svg class="w-4 h-4 ${cs.text} flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="${cat.icon}"/></svg>
        <span class="text-sm font-semibold ${cs.text}">${cat.label}</span>
        <span class="text-[10px] bg-dark-700/60 text-dark-300 px-1.5 py-0.5 rounded-full">${items.length} 個欄位</span>
        <svg class="w-3.5 h-3.5 text-dark-500 ml-auto proc-cat-arrow transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
      </div>
      <div class="proc-cat-body mt-1 space-y-1"></div>
    `;

    const body = section.querySelector('.proc-cat-body');
    const arrow = section.querySelector('.proc-cat-arrow');
    const header = section.querySelector('.proc-cat-header');

    // Toggle collapse
    header.addEventListener('click', () => {
      body.classList.toggle('hidden');
      arrow.classList.toggle('rotate-180');
    });

    // Render each item in this category
    items.forEach(log => {
      const colInfo = DataEngine.currentDataset?.analysis?.find(c => c.name === log.colName);
      const colType = colInfo?.type || 'numeric';
      const methods = DataEngine.getMethodOptions(log.action, colType);
      const itemDiv = document.createElement('div');
      itemDiv.className = 'proc-item flex items-center gap-3 py-2 px-3 rounded-lg bg-dark-800/40 hover:bg-dark-700/50 transition-all border border-transparent hover:border-dark-600/50';

      // Summary text per action (use colInfo from real data, or _mock fallback for demo)
      const mk = log._mock || {};
      let summary = '';
      if (log.action === 'fill_missing') {
        const pct = colInfo?.missingPct ?? mk.missingPct ?? '?';
        const cnt = colInfo?.missingCount ?? mk.missingCount ?? '?';
        summary = `<span class="text-warning-400">${pct}%</span> 缺失 (${cnt} 筆)`;
      } else if (log.action === 'handle_outlier') {
        const cnt = colInfo?.outlierCount ?? mk.outlierCount ?? '?';
        summary = `<span class="text-danger-400">${cnt}</span> 筆異常值 (IQR)`;
      } else if (log.action === 'drop_column') {
        const pct = colInfo?.missingPct ?? mk.missingPct ?? '?';
        summary = `缺失率 <span class="text-danger-400">${pct}%</span>`;
      } else if (log.action === 'encode') {
        const cnt = colInfo?.uniqueCount ?? mk.uniqueCount ?? '?';
        summary = `${cnt} 個類別`;
      } else if (log.action === 'decompose_date') {
        const min = colInfo?.stats?.min ?? mk.dateMin ?? '?';
        const max = colInfo?.stats?.max ?? mk.dateMax ?? '?';
        summary = `${min} ~ ${max}`;
      } else if (log.action === 'interaction') {
        summary = '產生新特徵';
      }

      // Build method <select> options
      const optionsHtml = methods.map(m =>
        `<option value="${m.value}" ${m.value === log.selectedMethod ? 'selected' : ''}>${m.label}</option>`
      ).join('');

      itemDiv.innerHTML = `
        <span class="font-mono text-accent-400 text-xs min-w-[100px] truncate" title="${escapeHtml(log.colName)}">${escapeHtml(log.colName)}</span>
        <span class="text-xs text-dark-400 flex-1">${summary}</span>
        <select class="proc-method-inline bg-dark-700 border border-dark-600 rounded px-2 py-1 text-[11px] focus:border-primary-500 outline-none min-w-[120px]" data-log-idx="${log._idx}">
          ${optionsHtml}
        </select>
        <button class="proc-detail-btn p-1.5 hover:bg-dark-600 rounded-lg transition-colors flex-shrink-0" data-log-idx="${log._idx}" title="查看前後對比">
          <svg class="w-4 h-4 text-dark-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/></svg>
        </button>
      `;

      // Inline method change: update log and re-render summary tag
      const sel = itemDiv.querySelector('.proc-method-inline');
      sel.addEventListener('click', e => e.stopPropagation());
      sel.addEventListener('change', () => {
        currentProcessLogs[log._idx].selectedMethod = sel.value;
      });

      // Detail button: open modal
      const detailBtn = itemDiv.querySelector('.proc-detail-btn');
      detailBtn.addEventListener('click', e => {
        e.stopPropagation();
        openProcessDetailModal(log._idx);
      });

      body.appendChild(itemDiv);
    });

    container.appendChild(section);
  });
}

function getMethodShortLabel(action, method) {
  const map = {
    fill_missing: { median: '中位數', mean: '平均數', zero: '填 0', mode: '眾數', unknown: 'Unknown', ffill: '前值填補', drop_rows: '刪除列' },
    handle_outlier: { keep: '保留', clip: 'IQR 截斷', remove: '刪除', median: '替換中位數' },
    encode: { onehot: 'One-Hot', label: 'Label', none: '不編碼' },
    decompose_date: { decompose: '拆解', timestamp: 'Unix 時間戳', none: '不處理' },
    drop_column: { drop_column: '移除', keep_fill: '保留填補' },
    interaction: { multiply: '相乘', add: '相加', ratio: '比值', none: '不產生' },
  };
  return map[action]?.[method] || null;
}

// ===== PROCESS DETAIL MODAL =====
function openProcessDetailModal(logIdx) {
  const log = currentProcessLogs[logIdx];
  if (!log) return;

  const modal = document.getElementById('process-detail-modal');
  modal.classList.remove('hidden');

  // Title
  const actionLabels = {
    fill_missing: '缺失值處理', handle_outlier: '異常值處理', encode: '類別編碼',
    decompose_date: '日期拆解', drop_column: '欄位移除', interaction: '交互特徵',
  };
  document.getElementById('proc-modal-title').textContent = actionLabels[log.action] || '處理詳情';
  document.getElementById('proc-modal-subtitle').textContent = `欄位: ${log.colName}`;

  // Populate method dropdown
  const colInfo = DataEngine.currentDataset?.analysis?.find(c => c.name === log.colName);
  const colType = colInfo?.type || 'numeric';
  const methods = DataEngine.getMethodOptions(log.action, colType);
  const select = document.getElementById('proc-method-select');
  select.innerHTML = '';
  methods.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.value;
    opt.textContent = m.label;
    if (m.value === log.selectedMethod) opt.selected = true;
    select.appendChild(opt);
  });

  // Show description of current method
  updateMethodDesc(methods, log.selectedMethod);

  // Render before/after for current method
  renderProcessPreview(log, log.selectedMethod);

  // When method changes, re-render preview
  const newSelect = select.cloneNode(true);
  select.parentNode.replaceChild(newSelect, select);
  newSelect.addEventListener('change', () => {
    const m = newSelect.value;
    updateMethodDesc(methods, m);
    renderProcessPreview(log, m);
  });

  // Apply button
  const applyBtn = document.getElementById('btn-apply-method');
  const newApply = applyBtn.cloneNode(true);
  applyBtn.parentNode.replaceChild(newApply, applyBtn);
  newApply.addEventListener('click', () => {
    const chosenMethod = newSelect.value;
    log.selectedMethod = chosenMethod;
    // Re-render the log list to show updated method tag
    const container = document.getElementById('auto-process-log');
    renderProcessLogItems(container, currentProcessLogs);
    modal.classList.add('hidden');
  });

  // Close
  const closeBtn = document.getElementById('close-process-modal');
  const newClose = closeBtn.cloneNode(true);
  closeBtn.parentNode.replaceChild(newClose, closeBtn);
  newClose.addEventListener('click', () => modal.classList.add('hidden'));
  modal.addEventListener('click', e => { if (e.target === modal) modal.classList.add('hidden'); });
}

function updateMethodDesc(methods, selectedValue) {
  const desc = methods.find(m => m.value === selectedValue)?.desc || '';
  document.getElementById('proc-method-desc').textContent = desc;
}

function renderProcessPreview(log, method) {
  // If we have real data, compute real before/after
  if (DataEngine.currentDataset && log.action !== 'interaction') {
    const ba = DataEngine.computeBeforeAfter(log.colName, log.action, method);
    if (ba) {
      renderProcessPreviewFromData(log, method, ba);
      return;
    }
  }
  // Fallback: demo/mock preview
  renderProcessPreviewMock(log, method);
}

function renderProcessPreviewFromData(log, method, ba) {
  const colInfo = DataEngine.currentDataset?.analysis?.find(c => c.name === log.colName);
  const isNumeric = colInfo?.type === 'numeric';

  // Before chart
  if (isNumeric && ba.before.distribution.length > 0) {
    renderBeforeAfterHistogram('chart-proc-before', ba.before.distribution, '#64748b');
  } else if (colInfo && (colInfo.type === 'categorical' || colInfo.type === 'boolean') && colInfo.topValues) {
    renderBeforeAfterCategorical('chart-proc-before', colInfo.topValues, '#64748b');
  } else {
    const c = initChart('chart-proc-before');
    if (c) { c.clear(); c.setOption({ title: { text: '此類型不支援分佈圖', left: 'center', top: 'center', textStyle: { color: '#475569', fontSize: 12 } } }); }
  }

  // After chart
  if (isNumeric && ba.after.distribution.length > 0) {
    renderBeforeAfterHistogram('chart-proc-after', ba.after.distribution, CHART_THEME.success);
  } else if (colInfo && (colInfo.type === 'categorical' || colInfo.type === 'boolean')) {
    // Recompute top values from after
    const afterValid = ba.after.values.filter(v => v !== '' && v !== '__DROPPED__');
    const counts = {};
    afterValid.forEach(v => { counts[v] = (counts[v] || 0) + 1; });
    const afterTop = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 8)
      .map(([value, count]) => ({ value, count }));
    renderBeforeAfterCategorical('chart-proc-after', afterTop, CHART_THEME.success);
  } else {
    const c = initChart('chart-proc-after');
    if (c) { c.clear(); c.setOption({ title: { text: '此類型不支援分佈圖', left: 'center', top: 'center', textStyle: { color: '#475569', fontSize: 12 } } }); }
  }

  // Info labels
  const beforeMissing = ba.before.values.filter(v => v === '' || (typeof v === 'string' && ['na','nan','null','?'].includes(v.toLowerCase()))).length;
  const afterMissing = ba.after.values.filter(v => v === '' || (typeof v === 'string' && ['na','nan','null','?'].includes(v.toLowerCase()))).length;
  const afterDropped = ba.after.values.filter(v => v === '__DROPPED__').length;
  document.getElementById('proc-before-info').textContent = `共 ${ba.before.values.length} 筆，缺失 ${beforeMissing} 筆`;
  document.getElementById('proc-after-info').textContent = afterDropped > 0
    ? `移除 ${afterDropped} 筆，剩餘 ${ba.after.values.length - afterDropped} 筆`
    : `共 ${ba.after.values.length} 筆，缺失 ${afterMissing} 筆`;

  // Stats comparison table
  renderStatsComparisonTable(ba.before.stats, ba.after.stats, isNumeric);

  // Affected rows preview
  renderAffectedRows(log.colName, ba);
}

function renderProcessPreviewMock(log, method) {
  // Generate synthetic before/after for demo mode
  const n = 200;
  const beforeDist = [];
  const afterDist = [];

  if (log.action === 'fill_missing') {
    // Simulate: before has a gap, after fills it
    for (let i = 0; i < 12; i++) {
      const base = Math.max(5, Math.round(30 * Math.exp(-0.5 * ((i - 5) / 2.5) ** 2)));
      beforeDist.push({ label: `Bin ${i+1}`, count: i === 5 || i === 6 ? Math.round(base * 0.3) : base });
      if (method === 'median' || method === 'mean') {
        afterDist.push({ label: `Bin ${i+1}`, count: i === 5 || i === 6 ? base + 8 : base });
      } else if (method === 'zero') {
        afterDist.push({ label: `Bin ${i+1}`, count: i === 0 ? base + 20 : base });
      } else {
        afterDist.push({ label: `Bin ${i+1}`, count: base });
      }
    }
  } else if (log.action === 'handle_outlier') {
    for (let i = 0; i < 12; i++) {
      const base = Math.max(3, Math.round(25 * Math.exp(-0.5 * ((i - 5) / 2.5) ** 2)));
      const outlier = (i === 0 || i === 11) ? 8 : 0;
      beforeDist.push({ label: `Bin ${i+1}`, count: base + outlier });
      if (method === 'clip') {
        afterDist.push({ label: `Bin ${i+1}`, count: (i === 0 || i === 11) ? 0 : (i === 1 || i === 10) ? base + 4 : base });
      } else if (method === 'remove') {
        afterDist.push({ label: `Bin ${i+1}`, count: (i === 0 || i === 11) ? 0 : base });
      } else if (method === 'median') {
        afterDist.push({ label: `Bin ${i+1}`, count: i === 5 ? base + outlier * 2 : (i === 0 || i === 11 ? 0 : base) });
      } else {
        afterDist.push({ label: `Bin ${i+1}`, count: base + outlier }); // keep
      }
    }
  } else {
    for (let i = 0; i < 8; i++) {
      const base = Math.round(10 + Math.random() * 30);
      beforeDist.push({ label: `Bin ${i+1}`, count: base });
      afterDist.push({ label: `Bin ${i+1}`, count: base });
    }
  }

  renderBeforeAfterHistogram('chart-proc-before', beforeDist, '#64748b');
  renderBeforeAfterHistogram('chart-proc-after', afterDist, CHART_THEME.success);

  document.getElementById('proc-before-info').textContent = `${n} 筆原始數據`;
  document.getElementById('proc-after-info').textContent = `${n} 筆處理後數據`;

  // Mock stats table
  const mockBefore = { count: n, mean: 45.2, median: 42.0, std: 18.3, min: 2, max: 195, q1: 30, q3: 58 };
  const mockAfter = { ...mockBefore };
  if (log.action === 'fill_missing') { mockAfter.count = n + 12; mockAfter.mean = method === 'zero' ? 40.1 : 44.8; mockAfter.std = method === 'zero' ? 20.1 : 17.9; }
  if (log.action === 'handle_outlier' && method === 'clip') { mockAfter.max = 95; mockAfter.std = 14.2; mockAfter.mean = 43.1; }
  if (log.action === 'handle_outlier' && method === 'remove') { mockAfter.count = n - 8; mockAfter.max = 90; mockAfter.std = 13.8; }
  renderStatsComparisonTable(mockBefore, mockAfter, true);

  // Mock affected rows
  document.getElementById('proc-affected-count').textContent = `共 ${log.action === 'fill_missing' ? 12 : 8} 筆受影響`;
  const thead = document.getElementById('proc-affected-thead');
  const tbody = document.getElementById('proc-affected-tbody');
  thead.innerHTML = `<tr class="text-dark-400 text-[10px] border-b border-dark-700/50">
    <th class="py-1.5 px-2 text-left">#</th>
    <th class="py-1.5 px-2 text-left">${escapeHtml(log.colName)} (前)</th>
    <th class="py-1.5 px-2 text-left">${escapeHtml(log.colName)} (後)</th>
    <th class="py-1.5 px-2 text-left">狀態</th>
  </tr>`;
  tbody.innerHTML = '';
  const mockRows = log.action === 'fill_missing' ? 12 : 8;
  for (let i = 0; i < Math.min(mockRows, 20); i++) {
    const row = document.createElement('tr');
    row.className = 'border-b border-dark-700/20';
    const beforeVal = log.action === 'fill_missing' ? '<span class="text-warning-400">NaN</span>' : (180 + Math.floor(Math.random() * 30));
    const afterVal = log.action === 'fill_missing'
      ? `<span class="text-success-400">${method === 'zero' ? '0' : '42.0'}</span>`
      : (method === 'clip' ? '<span class="text-success-400">95.0</span>' : method === 'remove' ? '<span class="text-danger-400 line-through">已移除</span>' : beforeVal);
    row.innerHTML = `
      <td class="py-1.5 px-2 text-dark-500">${100 + i * 37}</td>
      <td class="py-1.5 px-2">${beforeVal}</td>
      <td class="py-1.5 px-2">${afterVal}</td>
      <td class="py-1.5 px-2"><span class="text-[10px] ${method === 'remove' ? 'text-danger-400' : 'text-success-400'}">${method === 'remove' ? '已移除' : method === 'keep' ? '保留' : '已修改'}</span></td>
    `;
    tbody.appendChild(row);
  }
}

function renderStatsComparisonTable(before, after, isNumeric) {
  const tbody = document.getElementById('proc-stats-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  if (!isNumeric || !before || !after) {
    tbody.innerHTML = '<tr><td colspan="4" class="text-center text-dark-500 text-xs py-4">非數值型欄位，無統計對比</td></tr>';
    return;
  }

  const metrics = [
    { label: '筆數', key: 'count', fmt: v => v?.toLocaleString() ?? '—' },
    { label: '平均數', key: 'mean', fmt: v => v?.toFixed(4) ?? '—' },
    { label: '中位數', key: 'median', fmt: v => v?.toFixed(4) ?? '—' },
    { label: '標準差', key: 'std', fmt: v => v?.toFixed(4) ?? '—' },
    { label: '最小值', key: 'min', fmt: v => v?.toFixed(4) ?? '—' },
    { label: 'Q1 (25%)', key: 'q1', fmt: v => v?.toFixed(4) ?? '—' },
    { label: 'Q3 (75%)', key: 'q3', fmt: v => v?.toFixed(4) ?? '—' },
    { label: '最大值', key: 'max', fmt: v => v?.toFixed(4) ?? '—' },
  ];

  metrics.forEach(m => {
    const bv = before[m.key];
    const av = after[m.key];
    let diffStr = '—';
    let diffClass = 'text-dark-500';
    if (bv != null && av != null) {
      const diff = av - bv;
      if (Math.abs(diff) > 0.0001) {
        diffStr = (diff > 0 ? '+' : '') + (Number.isInteger(diff) ? diff.toLocaleString() : diff.toFixed(4));
        diffClass = diff > 0 ? 'text-accent-400' : 'text-warning-400';
      } else {
        diffStr = '—';
      }
    }
    const tr = document.createElement('tr');
    tr.className = 'border-b border-dark-700/20';
    tr.innerHTML = `
      <td class="py-2 px-4 text-dark-300 text-xs">${m.label}</td>
      <td class="py-2 px-4 text-right font-mono text-xs">${m.fmt(bv)}</td>
      <td class="py-2 px-4 text-right font-mono text-xs">${m.fmt(av)}</td>
      <td class="py-2 px-4 text-right font-mono text-xs ${diffClass}">${diffStr}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderAffectedRows(colName, ba) {
  const ds = DataEngine.currentDataset;
  if (!ds) return;
  const colIdx = ds.headers.indexOf(colName);
  if (colIdx < 0) return;

  const affected = ba.affectedIndices.slice(0, 20);
  document.getElementById('proc-affected-count').textContent = `共 ${ba.affectedIndices.length} 筆受影響`;

  const thead = document.getElementById('proc-affected-thead');
  const tbody = document.getElementById('proc-affected-tbody');
  thead.innerHTML = `<tr class="text-dark-400 text-[10px] border-b border-dark-700/50">
    <th class="py-1.5 px-2 text-left"># 列</th>
    <th class="py-1.5 px-2 text-left">${escapeHtml(colName)} (處理前)</th>
    <th class="py-1.5 px-2 text-left">${escapeHtml(colName)} (處理後)</th>
    <th class="py-1.5 px-2 text-left">狀態</th>
  </tr>`;
  tbody.innerHTML = '';

  affected.forEach(i => {
    const bv = ba.before.values[i] || '';
    const av = ba.after.values[i] || '';
    const bvLower = String(bv).trim().toLowerCase();
    const isMissing = bv === '' || ['na','nan','null','n/a','none','missing','undefined','?','-','--','.'].includes(bvLower);
    const isDropped = av === '__DROPPED__';
    const isChanged = bv !== av;

    const tr = document.createElement('tr');
    tr.className = 'border-b border-dark-700/20';
    tr.innerHTML = `
      <td class="py-1.5 px-2 text-dark-500">${i + 1}</td>
      <td class="py-1.5 px-2 ${isMissing ? 'text-warning-400 italic' : ''}">${isMissing ? '(缺失)' : escapeHtml(bv)}</td>
      <td class="py-1.5 px-2 ${isDropped ? 'text-danger-400 line-through' : isChanged ? 'text-success-400' : ''}">${isDropped ? '已移除' : escapeHtml(av)}</td>
      <td class="py-1.5 px-2"><span class="text-[10px] ${isDropped ? 'text-danger-400' : isChanged ? 'text-success-400' : 'text-dark-500'}">${isDropped ? '已移除' : isChanged ? '已修改' : '未變動'}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function escapeHtml(str) {
  if (typeof str !== 'string') return str;
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// 預處理下拉的時間顯示 — 今天的給 HH:MM,昨天前給 MM/DD HH:MM
function _formatPreprocessTime(ms) {
  const d = new Date(ms);
  const now = new Date();
  const sameDay = d.getFullYear() === now.getFullYear()
                && d.getMonth() === now.getMonth()
                && d.getDate() === now.getDate();
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  if (sameDay) return `今天 ${hh}:${mm}`;
  const mo = String(d.getMonth() + 1).padStart(2, '0');
  const dy = String(d.getDate()).padStart(2, '0');
  return `${mo}/${dy} ${hh}:${mm}`;
}

// ===== LEADERBOARD =====
let selectedModels = new Set();

function initLeaderboard() {
  renderLeaderboardTable();
  initSorting();
  initModelSelection();
  initCompareModal();
}

function renderLeaderboardTable() {
  const tbody = document.getElementById('leaderboard-body');
  if (!tbody) return;
  tbody.innerHTML = '';
  const models = [...MOCK.leaderboardModels].sort((a, b) => b.f1 - a.f1);
  models.forEach((m, i) => {
    const rank = i + 1;
    const tr = document.createElement('tr');
    tr.dataset.modelName = m.name;
    if (selectedModels.has(m.name)) tr.classList.add('selected');
    let tagsHtml = '';
    if (m.tags.includes('best')) tagsHtml += '<span class="text-[10px] bg-warning-500/10 text-warning-400 px-1.5 py-0.5 rounded-full mr-1">🏆 最佳</span>';
    if (m.tags.includes('fastest')) tagsHtml += '<span class="text-[10px] bg-accent-500/10 text-accent-400 px-1.5 py-0.5 rounded-full mr-1">⚡ 最快</span>';
    if (m.tags.includes('deployed')) tagsHtml += '<span class="text-[10px] bg-success-500/10 text-success-400 px-1.5 py-0.5 rounded-full">已部署</span>';
    const f1Bar = `<div class="w-16 bg-dark-800 rounded-full h-1.5 inline-block align-middle ml-2"><div class="h-1.5 rounded-full bg-primary-500" style="width:${m.f1 * 100}%"></div></div>`;
    tr.innerHTML = `
      <td class="py-3 px-4"><input type="checkbox" class="model-checkbox" data-model="${m.name}" ${selectedModels.has(m.name) ? 'checked' : ''}></td>
      <td class="py-3 px-4 font-mono text-dark-400 text-xs">#${rank}</td>
      <td class="py-3 px-4"><span class="inline-block px-2 py-0.5 rounded-md text-[11px] font-semibold bg-dark-700 text-dark-300 border border-dark-600">Demo</span></td>
      <td class="py-3 px-4 font-medium text-sm">${m.name}</td>
      <td class="py-3 px-4"><span class="font-mono text-sm ${rank <= 2 ? 'text-success-400' : ''}">${m.f1.toFixed(3)}</span>${f1Bar}</td>
      <td class="py-3 px-4 font-mono text-sm">${m.auc.toFixed(3)}</td>
      <td class="py-3 px-4 font-mono text-sm">${(m.accuracy * 100).toFixed(1)}%</td>
      <td class="py-3 px-4 text-dark-400 text-xs font-mono">${m.time}</td>
      <td class="py-3 px-4 text-dark-400 text-xs font-mono">${m.latency}</td>
      <td class="py-3 px-4">${tagsHtml || '<span class="text-dark-600 text-xs">-</span>'}</td>
      <td class="py-3 px-4">
        <button class="text-xs text-primary-400 hover:text-primary-300 mr-2" onclick="navigateTo('insights')">分析</button>
        <button class="text-xs text-dark-400 hover:text-dark-200">部署</button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function initSorting() {
  document.querySelectorAll('.sortable').forEach(th => {
    th.addEventListener('click', () => {
      const key = th.dataset.sort;
      const isAsc = th.classList.contains('sort-asc');
      document.querySelectorAll('.sortable').forEach(t => t.classList.remove('sort-asc', 'sort-desc'));
      th.classList.add(isAsc ? 'sort-desc' : 'sort-asc');
      const ascending = !isAsc;  // 點擊後的新狀態

      // 實作模式 — 排序真實訓練模型,不要去動 demo 假資料
      if (appMode === 'real' && MLEngine.trainedModels.length > 0) {
        sortRealLeaderboard(key, ascending);
        return;
      }

      MOCK.leaderboardModels.sort((a, b) => {
        let va = a[key], vb = b[key];
        if (key === 'name') return isAsc ? vb.localeCompare(va) : va.localeCompare(vb);
        if (key === 'time') { va = a.timeVal; vb = b.timeVal; }
        if (key === 'latency') { va = a.latencyVal; vb = b.latencyVal; }
        return isAsc ? va - vb : vb - va;
      });
      renderLeaderboardTable();
      initModelSelection();
    });
  });
}

// 實作模式的排行榜排序 — 直接排 MLEngine.trainedModels 再重繪
function sortRealLeaderboard(key, ascending) {
  const models = MLEngine.trainedModels;
  if (models.length === 0) return;
  const isReg = models[0].taskType === 'regression';

  const valOf = (m) => {
    switch (key) {
      case 'name':    return m.name;
      case 'metric1': return isReg ? m.metrics.testR2   : m.metrics.f1;
      case 'metric2': return isReg ? m.metrics.testRMSE : (m.metrics.auc || 0);
      case 'metric3': return isReg ? m.metrics.testMAE  : m.metrics.testAccuracy;
      case 'time':    return m.trainTime;
      case 'latency': return m.inferLatency || 0;
      default:        return 0;
    }
  };

  models.sort((a, b) => {
    const va = valOf(a), vb = valOf(b);
    if (key === 'name') {
      return ascending ? String(va).localeCompare(String(vb))
                       : String(vb).localeCompare(String(va));
    }
    return ascending ? va - vb : vb - va;
  });
  renderRealLeaderboard();
}

function initModelSelection() {
  const compareBtn = document.getElementById('btn-compare');
  document.querySelectorAll('.model-checkbox').forEach(cb => {
    cb.addEventListener('change', () => {
      if (cb.checked) {
        if (selectedModels.size >= 3) { cb.checked = false; return; }
        selectedModels.add(cb.dataset.model);
      } else {
        selectedModels.delete(cb.dataset.model);
      }
      cb.closest('tr').classList.toggle('selected', cb.checked);
      if (compareBtn) compareBtn.disabled = selectedModels.size < 2;
    });
  });
  const selectAll = document.getElementById('select-all-models');
  if (selectAll) {
    selectAll.addEventListener('change', () => {
      if (selectAll.checked) {
        selectedModels.clear();
        document.querySelectorAll('.model-checkbox').forEach((cb, i) => {
          if (i < 3) { cb.checked = true; selectedModels.add(cb.dataset.model); cb.closest('tr').classList.add('selected'); }
        });
      } else {
        selectedModels.clear();
        document.querySelectorAll('.model-checkbox').forEach(cb => { cb.checked = false; cb.closest('tr').classList.remove('selected'); });
      }
      if (compareBtn) compareBtn.disabled = selectedModels.size < 2;
    });
  }
}

function initCompareModal() {
  const modal = document.getElementById('compare-modal');
  const btn = document.getElementById('btn-compare');
  const closeBtn = document.getElementById('close-compare');
  if (btn) {
    btn.addEventListener('click', () => {
      if (selectedModels.size < 2) return;
      const models = MOCK.leaderboardModels.filter(m => selectedModels.has(m.name));
      modal.classList.remove('hidden');
      setTimeout(() => { renderCompareRadar(models); renderCompareConfusion(models); renderCompareROC(models); }, 100);
    });
  }
  if (closeBtn) closeBtn.addEventListener('click', () => modal.classList.add('hidden'));
  if (modal) modal.addEventListener('click', e => { if (e.target === modal) modal.classList.add('hidden'); });
}

// ===== WHAT-IF SIMULATOR =====
function initWhatIfSimulator() {
  const sliders = [
    { id: 'slider-age', valId: 'val-age', format: v => v },
    { id: 'slider-income', valId: 'val-income', format: v => Number(v).toLocaleString() },
    { id: 'slider-tenure', valId: 'val-tenure', format: v => v },
    { id: 'slider-recency', valId: 'val-recency', format: v => v },
    { id: 'slider-spending', valId: 'val-spending', format: v => Number(v).toLocaleString() },
  ];
  sliders.forEach(s => {
    const slider = document.getElementById(s.id);
    const valEl = document.getElementById(s.valId);
    if (slider && valEl) {
      slider.addEventListener('input', () => { valEl.textContent = s.format(slider.value); updateWhatIfPrediction(); });
    }
  });
  ['select-complaints', 'select-contract'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', updateWhatIfPrediction);
  });
  const resetBtn = document.getElementById('btn-reset-whatif');
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      setSlider('slider-age', 35); setSlider('slider-income', 50000);
      setSlider('slider-tenure', 365); setSlider('slider-recency', 30);
      setSlider('slider-spending', 3000);
      const complaints = document.getElementById('select-complaints');
      if (complaints) complaints.value = '2';
      const contract = document.getElementById('select-contract');
      if (contract) contract.value = 'yearly';
      document.getElementById('val-age').textContent = '35';
      document.getElementById('val-income').textContent = '50,000';
      document.getElementById('val-tenure').textContent = '365';
      document.getElementById('val-recency').textContent = '30';
      document.getElementById('val-spending').textContent = '3,000';
      updateWhatIfPrediction();
    });
  }
}
function setSlider(id, val) { const s = document.getElementById(id); if (s) s.value = val; }

function updateWhatIfPrediction() {
  const age = parseInt(document.getElementById('slider-age')?.value || 35);
  const income = parseInt(document.getElementById('slider-income')?.value || 50000);
  const tenure = parseInt(document.getElementById('slider-tenure')?.value || 365);
  const recency = parseInt(document.getElementById('slider-recency')?.value || 30);
  const spending = parseInt(document.getElementById('slider-spending')?.value || 3000);
  const complaints = parseInt(document.getElementById('select-complaints')?.value || 2);
  const contract = document.getElementById('select-contract')?.value || 'yearly';

  let churnProb = 0.28;
  churnProb += Math.max(0, (500 - tenure) / 1500) * 0.25;
  churnProb += complaints * 0.08;
  churnProb += Math.max(0, (recency - 15) / 350) * 0.15;
  churnProb += Math.max(0, (3000 - spending) / 50000) * 0.12;
  if (contract === 'monthly') churnProb += 0.12;
  else if (contract === 'two_year') churnProb -= 0.10;
  churnProb += Math.max(0, (35 - age) / 80) * 0.05;
  churnProb += Math.max(0, (50000 - income) / 200000) * 0.05;
  churnProb = Math.max(0.02, Math.min(0.98, churnProb));

  const pct = (churnProb * 100).toFixed(1);
  const retainPct = ((1 - churnProb) * 100).toFixed(1);
  const clv = Math.round((1 - churnProb) * spending * 12 * 0.3);

  document.getElementById('whatif-churn-pct').textContent = pct + '%';
  document.getElementById('whatif-retain-pct').textContent = retainPct + '%';
  document.getElementById('whatif-clv').textContent = 'NT$' + clv.toLocaleString();

  const labelEl = document.getElementById('whatif-result-label');
  const detailEl = document.getElementById('whatif-result-detail');
  const recEl = document.getElementById('whatif-recommendation');

  if (churnProb < 0.3) {
    labelEl.textContent = '低風險'; labelEl.className = 'text-2xl font-bold text-success-400';
    detailEl.textContent = `此客戶有 ${pct}% 的流失機率`;
    recEl.textContent = '客戶狀態良好，可考慮交叉銷售或推薦升級方案以增加客戶價值。';
  } else if (churnProb < 0.6) {
    labelEl.textContent = '中風險'; labelEl.className = 'text-2xl font-bold text-warning-400';
    detailEl.textContent = `此客戶有 ${pct}% 的流失機率`;
    recEl.textContent = '建議提供個人化優惠或升級方案以降低流失風險，並加強客服互動。';
  } else {
    labelEl.textContent = '高風險'; labelEl.className = 'text-2xl font-bold text-danger-400';
    detailEl.textContent = `此客戶有 ${pct}% 的流失機率`;
    recEl.textContent = '需立即介入！建議啟動客戶挽留計劃，提供專屬折扣或客服經理一對一關懷。';
  }
  document.getElementById('whatif-churn-pct').className = `font-mono ${churnProb >= 0.5 ? 'text-danger-400' : churnProb >= 0.3 ? 'text-warning-400' : 'text-success-400'}`;
  renderWhatIfGauge(churnProb);
}

// ===== TRAINING BUTTON =====
function initTrainingButton() {
  const btn = document.getElementById('btn-start-training');
  if (!btn) return;
  btn.addEventListener('click', () => {
    btn.disabled = true;
    btn.innerHTML = '<div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin inline-block mr-2"></div>訓練中...';
    btn.classList.add('opacity-75');
    let progress = 62;
    const interval = setInterval(() => {
      progress += Math.random() * 3;
      if (progress >= 100) {
        progress = 100;
        clearInterval(interval);
        btn.innerHTML = '✓ 訓練完成';
        btn.classList.remove('opacity-75');
        btn.classList.add('bg-success-600');
        document.querySelectorAll('.pipeline-node').forEach(n => n.classList.add('completed'));
        document.querySelectorAll('.pipeline-connector').forEach(c => { c.classList.add('completed'); c.classList.remove('active'); });
        setGlobalStatus('success', '訓練完成 (Demo)');
        const log = document.getElementById('training-log');
        if (log) { log.innerHTML += '<p class="text-success-400">[10:35:00] ✓ 所有模型訓練完成！最佳模型: LightGBM (F1=0.942)</p>'; log.scrollTop = log.scrollHeight; }
      }
      const bar = document.getElementById('training-progress-bar');
      const pctEl = document.getElementById('training-progress-pct');
      if (bar) bar.style.width = Math.round(progress) + '%';
      if (pctEl) pctEl.textContent = Math.round(progress) + '%';
      const eta = document.getElementById('training-eta');
      if (eta) {
        const remaining = Math.max(0, Math.round((100 - progress) / 3 * 2));
        eta.textContent = `${String(Math.floor(remaining / 60)).padStart(2, '0')}:${String(remaining % 60).padStart(2, '0')}`;
      }
    }, 800);
  });
}

// ===== PREPROCESSING PAGE (隊友模組) =====
let ppLastPreprocessorId = null;
let ppLastFeatureColumns = [];   // 原始欄位 (不含 target)
let ppHistory = [];              // 所有跑過的預處理結果 (給實驗室「資料來源」下拉用)
let ppListenersBound = false;

function renderPreprocessingPage() {
  const ds = DataEngine.currentDataset;
  const noData = document.getElementById('pp-no-data');
  const config = document.getElementById('pp-config-wrap');
  if (!ds) {
    noData.classList.remove('hidden');
    config.classList.add('hidden');
    return;
  }
  noData.classList.add('hidden');
  config.classList.remove('hidden');

  document.getElementById('pp-dataset-badge').textContent = ds.fileName || 'Dataset';

  // Target select
  const targetSel = document.getElementById('pp-target-select');
  const prevTarget = targetSel.value;
  targetSel.innerHTML = '';
  (ds.analysis || []).forEach(col => {
    const opt = document.createElement('option');
    opt.value = col.name;
    opt.textContent = `${col.name} (${col.type})`;
    targetSel.appendChild(opt);
  });
  if (prevTarget && [...targetSel.options].some(o => o.value === prevTarget)) {
    targetSel.value = prevTarget;
  }

  // Test size slider
  const slider = document.getElementById('pp-test-size');
  const sliderLabel = document.getElementById('pp-test-size-label');
  slider.oninput = () => { sliderLabel.textContent = `${Math.round(slider.value * 100)}%`; };
  sliderLabel.textContent = `${Math.round(slider.value * 100)}%`;

  // Bind buttons once
  if (!ppListenersBound) {
    document.getElementById('btn-pp-audit').addEventListener('click', runPreprocessAudit);
    document.getElementById('btn-pp-run').addEventListener('click', runPreprocessTransform);
    document.getElementById('btn-pp-inference').addEventListener('click', runPreprocessInference);
    document.getElementById('btn-export-audit-html')?.addEventListener('click', exportAuditReportHtml);
    document.getElementById('btn-export-audit-json')?.addEventListener('click', exportAuditReportJson);
    // 對抗驗證 file input — change 顯示檔名,clear 鈕清掉
    const advInput = document.getElementById('pp-adv-file');
    const advLabel = document.getElementById('pp-adv-file-label');
    const advClear = document.getElementById('pp-adv-file-clear');
    advInput?.addEventListener('change', () => {
      const f = advInput.files?.[0];
      if (f && advLabel) {
        const short = f.name.length > 22 ? f.name.slice(0, 20) + '…' : f.name;
        advLabel.textContent = `已選 ${short}`;
        advLabel.classList.add('text-success-400');
        advClear?.classList.remove('hidden');
      } else if (advLabel) {
        advLabel.textContent = '選擇 test.csv';
        advLabel.classList.remove('text-success-400');
        advClear?.classList.add('hidden');
      }
    });
    advClear?.addEventListener('click', (e) => {
      e.preventDefault();
      if (advInput) advInput.value = '';
      if (advLabel) {
        advLabel.textContent = '選擇 test.csv';
        advLabel.classList.remove('text-success-400');
      }
      advClear.classList.add('hidden');
    });
    ppListenersBound = true;
  }
}

function ppSetStatus(visible, text) {
  const wrap = document.getElementById('pp-status');
  const txt = document.getElementById('pp-status-text');
  if (visible) {
    wrap.classList.remove('hidden');
    txt.textContent = text || '處理中...';
    setGlobalStatus('running', text || '處理中...');
  } else {
    wrap.classList.add('hidden');
    // 不主動清空 global,讓接下來的 success/error 設定接手
  }
}

function ppRequireApi() {
  const ds = DataEngine.currentDataset;
  if (!ds || !ds._fromApi || !ds.id || typeof ds.id !== 'string') {
    alert('預處理功能需要連接 Python 後端 API。\n請至「系統設定」啟用「使用 Python 後端 API」,並重新上傳 CSV。');
    return null;
  }
  return ds;
}

async function runPreprocessAudit() {
  const ds = ppRequireApi();
  if (!ds) return;
  const target = document.getElementById('pp-target-select').value;
  ppSetStatus(true, '快速健檢中...');
  try {
    const res = await ApiClient.preprocessAudit({ datasetId: ds.id, target });
    renderAuditReport(res.auditReport);
    renderFeatureGroups(res.featureGroups);
    document.getElementById('pp-audit-section').classList.remove('hidden');
    document.getElementById('pp-groups-section').classList.remove('hidden');
    setGlobalStatus('success', `健檢完成 — ${ds.fileName || 'Dataset'}`);
  } catch (e) {
    setGlobalStatus('error', '健檢失敗');
    alert(`健檢失敗: ${e.message}`);
  } finally {
    ppSetStatus(false);
  }
}

async function runPreprocessTransform() {
  const ds = ppRequireApi();
  if (!ds) return;
  const target = document.getElementById('pp-target-select').value;
  const testSize = parseFloat(document.getElementById('pp-test-size').value);
  const useMice = document.getElementById('pp-opt-mice')?.checked || false;
  const useMiSelection = document.getElementById('pp-opt-mi')?.checked || false;
  // 對抗驗證測試集 — 上傳 Kaggle 風 test.csv 觸發 daniel 的 adv val 防護
  const advFile = document.getElementById('pp-adv-file')?.files?.[0] || null;
  ppSetStatus(true, advFile
    ? `執行完整預處理管線 + 對抗驗證 (${advFile.name})...`
    : '執行完整預處理管線...');
  try {
    const res = await ApiClient.preprocessTransform({
      datasetId: ds.id, target, testSize, useMice, useMiSelection,
      adversarialTestFile: advFile,
    });
    renderAuditReport(res.auditReport);
    renderFeatureGroups(res.featureGroups);
    renderTransformResult(res);
    renderAppliedSteps(res, { useMice, useMiSelection });
    renderInferenceForm(ds, target);
    ppLastPreprocessorId = res.preprocessorId;
    ppLastFeatureColumns = (ds.headers || []).filter(h => h !== target);
    // 記錄到 history,實驗室「資料來源」下拉會用到
    ppHistory = ppHistory.filter(p => p.id !== res.preprocessorId);
    ppHistory.unshift({
      id: res.preprocessorId,
      datasetId: ds.id,
      fileName: ds.fileName || 'Dataset',
      target,
      trainSize: res.trainSize,
      testSize: res.testSize,
      featureCount: res.transformedFeatureCount,
    });
    document.getElementById('pp-audit-section').classList.remove('hidden');
    document.getElementById('pp-groups-section').classList.remove('hidden');
    document.getElementById('pp-result-section').classList.remove('hidden');
    document.getElementById('pp-inference-section').classList.remove('hidden');
    setGlobalStatus('success', `預處理完成 — ${ds.fileName || 'Dataset'}`);
  } catch (e) {
    setGlobalStatus('error', '預處理失敗');
    alert(`預處理失敗: ${e.message}`);
  } finally {
    ppSetStatus(false);
  }
}

function renderAuditReport(audit) {
  // Save for Process Log to consume
  DataEngine.lastAuditReport = audit || null;
  // Refresh the Process Log if it's currently visible (re-run generateProcessingLog)
  const logContainer = document.getElementById('auto-process-log');
  if (logContainer) {
    const freshLogs = DataEngine.generateProcessingLog();
    renderProcessLogItems(logContainer, freshLogs);
  }
  if (!audit || audit.error) {
    document.getElementById('pp-stat-rows').textContent = '—';
    return;
  }
  document.getElementById('pp-stat-rows').textContent = audit.total_rows ?? '—';
  document.getElementById('pp-stat-cols').textContent = audit.total_columns ?? '—';
  document.getElementById('pp-stat-perfect').textContent = audit.perfect_columns ?? '—';
  const missingCols = Object.keys(audit.missing_summary || {}).length;
  document.getElementById('pp-stat-missing').textContent = missingCols;

  // Warnings
  const warnList = document.getElementById('pp-warnings-list');
  warnList.innerHTML = '';
  const warns = audit.warnings || [];
  document.getElementById('pp-warning-count').textContent = warns.length;
  if (warns.length === 0) {
    warnList.innerHTML = '<p class="text-xs text-success-400">✓ 沒有偵測到問題</p>';
  } else {
    warns.forEach(w => {
      const div = document.createElement('div');
      div.className = 'text-xs px-3 py-2 bg-warning-500/10 border-l-2 border-warning-500 rounded-r text-warning-300';
      div.textContent = w;
      warnList.appendChild(div);
    });
  }

  // Missing summary table
  const tbody = document.getElementById('pp-missing-table');
  tbody.innerHTML = '';
  const summary = audit.missing_summary || {};
  if (Object.keys(summary).length === 0) {
    tbody.innerHTML = '<tr><td colspan="3" class="text-center py-3 text-xs text-dark-500">所有欄位均無缺失值</td></tr>';
  } else {
    Object.entries(summary).forEach(([col, info]) => {
      const tr = document.createElement('tr');
      tr.className = 'border-b border-dark-800/50';
      const pct = (info.ratio * 100).toFixed(1);
      const pctColor = info.ratio > 0.3 ? 'text-danger-400' : info.ratio > 0.1 ? 'text-warning-400' : 'text-dark-300';
      tr.innerHTML = `
        <td class="py-1.5 px-3">${escapeHtml(col)}</td>
        <td class="py-1.5 px-3 text-right text-dark-300">${info.count}</td>
        <td class="py-1.5 px-3 text-right font-mono ${pctColor}">${pct}%</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // v2 audit 擴充欄位 — info / outliers / target_info / leakage
  renderAuditInfoMessages(audit.info || []);
  renderAuditOutliers(audit.outlier_summary || []);
  renderAuditTargetInfo(audit.target_info || {});
  renderAuditLeakage(audit.leakage_candidates || []);
  // v3 新增診斷 — 標籤雜訊 / 集成離群
  renderAuditLabelNoise(audit.label_noise_candidates || []);
  renderAuditEnsembleOutliers(audit.ensemble_outlier_summary || []);

  // 切 extras-section 顯示狀態:任一張子卡可見就秀
  const extras = document.getElementById('pp-extras-section');
  if (extras) {
    const hasTarget = audit.target_info && Object.keys(audit.target_info).length > 0;
    const hasOutliers = (audit.outlier_summary || []).length > 0;
    extras.classList.toggle('hidden', !(hasTarget || hasOutliers));
  }
}

function renderAuditInfoMessages(infos) {
  const list = document.getElementById('pp-info-list');
  if (!list) return;
  list.innerHTML = '';
  if (!infos || infos.length === 0) {
    list.classList.add('hidden');
    return;
  }
  list.classList.remove('hidden');
  infos.forEach(msg => {
    const div = document.createElement('div');
    div.className = 'text-xs px-3 py-2 bg-success-500/10 border-l-2 border-success-500 rounded-r text-success-300';
    div.textContent = msg;
    list.appendChild(div);
  });
}

function renderAuditOutliers(outliers) {
  const card = document.getElementById('pp-outliers-card');
  const tbody = document.getElementById('pp-outliers-table');
  const countEl = document.getElementById('pp-outliers-count');
  if (!card || !tbody || !countEl) return;
  if (!outliers || outliers.length === 0) {
    card.classList.add('hidden');
    return;
  }
  card.classList.remove('hidden');
  countEl.textContent = `${outliers.length} 欄`;
  tbody.innerHTML = '';
  outliers.forEach(o => {
    const tr = document.createElement('tr');
    tr.className = 'border-b border-dark-800/50';
    const pct = (o.outlier_ratio * 100).toFixed(1);
    const pctColor = o.outlier_ratio > 0.1 ? 'text-danger-400' : 'text-warning-400';
    tr.innerHTML = `
      <td class="py-1.5 px-3 font-mono">${escapeHtml(o.column)}</td>
      <td class="py-1.5 px-3 text-right text-dark-300">${o.outlier_count.toLocaleString()}</td>
      <td class="py-1.5 px-3 text-right font-mono ${pctColor}">${pct}%</td>
      <td class="py-1.5 px-3 text-right font-mono text-dark-400">${o.lower_bound}</td>
      <td class="py-1.5 px-3 text-right font-mono text-dark-400">${o.upper_bound}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderAuditLeakage(candidates) {
  const card = document.getElementById('pp-leakage-card');
  const tbody = document.getElementById('pp-leakage-table');
  const highEl = document.getElementById('pp-leakage-high-count');
  const medEl = document.getElementById('pp-leakage-med-count');
  if (!card || !tbody) return;
  if (!candidates || candidates.length === 0) {
    card.classList.add('hidden');
    return;
  }
  card.classList.remove('hidden');

  // 風險分桶 + 排序:高風險在前,內部按相關性遞減
  const highs = candidates.filter(c => c.risk === 'high').sort((a, b) => b.correlation - a.correlation);
  const meds  = candidates.filter(c => c.risk !== 'high').sort((a, b) => b.correlation - a.correlation);

  if (highs.length > 0) {
    highEl.textContent = `${highs.length} 高風險`;
    highEl.classList.remove('hidden');
  } else {
    highEl.classList.add('hidden');
  }
  if (meds.length > 0) {
    medEl.textContent = `${meds.length} 中風險`;
    medEl.classList.remove('hidden');
  } else {
    medEl.classList.add('hidden');
  }

  tbody.innerHTML = '';
  [...highs, ...meds].forEach(c => {
    const isHigh = c.risk === 'high';
    const tr = document.createElement('tr');
    tr.className = 'border-b border-dark-800/50';
    const corrPct = (c.correlation * 100).toFixed(1);
    const corrColor = isHigh ? 'text-danger-400' : 'text-warning-400';
    const riskBadge = isHigh
      ? '<span class="text-[10px] px-2 py-0.5 rounded border border-danger-500/30 bg-danger-500/10 text-danger-400">🚨 高風險</span>'
      : '<span class="text-[10px] px-2 py-0.5 rounded border border-warning-500/30 bg-warning-500/10 text-warning-400">ℹ️ 中風險</span>';
    const methodLabel = c.method === 'Pearson'
      ? '<span class="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/15 text-blue-300 font-mono">Pearson</span>'
      : `<span class="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/15 text-purple-300 font-mono">${escapeHtml(c.method)}</span>`;
    tr.innerHTML = `
      <td class="py-2 px-3 font-mono">${escapeHtml(c.column)}</td>
      <td class="py-2 px-3 text-right">
        <div class="flex items-center justify-end gap-2">
          <div class="w-20 bg-dark-800 rounded-full h-1.5 overflow-hidden">
            <div class="h-full ${isHigh ? 'bg-danger-500' : 'bg-warning-500'}" style="width:${corrPct}%"></div>
          </div>
          <span class="font-mono ${corrColor} w-12 text-right">${c.correlation.toFixed(3)}</span>
        </div>
      </td>
      <td class="py-2 px-3 text-center">${methodLabel}</td>
      <td class="py-2 px-3 text-center">${riskBadge}</td>
    `;
    tbody.appendChild(tr);
  });
}

// v3:顯示「這次實際套用了哪些前處理」(進階特徵生成 + MICE + MI 選擇)
function renderAppliedSteps(res, opts) {
  const card = document.getElementById('pp-applied-card');
  const body = document.getElementById('pp-applied-body');
  if (!card || !body) return;
  opts = opts || {};
  const steps = res.appliedFeatureSteps || [];
  const mi = res.miSelection || null;
  const rows = [];

  // 1. 進階特徵生成 (Grouped/Poly/NonLinear) — 自動觸發才有
  if (steps.length > 0) {
    steps.forEach(s => {
      rows.push(`<div class="flex items-start gap-2 text-xs px-3 py-2 bg-accent-500/10 border-l-2 border-accent-500 rounded-r">
        <span class="text-accent-300 font-medium whitespace-nowrap">⚙ ${escapeHtml(s.label || s.type || '')}</span>
        <span class="text-dark-300">${escapeHtml(s.detail || '')}</span>
      </div>`);
    });
  } else {
    rows.push(`<div class="text-xs px-3 py-2 bg-dark-800/40 border-l-2 border-dark-600 rounded-r text-dark-400">
      未觸發進階特徵生成(分組聚合 / 多項式交互 / 非線性矯正 需偵測到特定欄位名稱才會自動套用)
    </div>`);
  }

  // 2. MICE 補值狀態
  rows.push(opts.useMice
    ? `<div class="text-xs px-3 py-2 rounded-r border-l-2 bg-primary-500/10 border-primary-500 text-primary-300">✓ 已啟用 MICE 補值(IterativeImputer:用其他欄位預測缺值)</div>`
    : `<div class="text-xs px-3 py-2 rounded-r border-l-2 bg-dark-800/40 border-dark-600 text-dark-400">○ MICE 未啟用(用中位數補值)</div>`);

  // 3. MI 特徵選擇結果
  if (mi) {
    const dropped = mi.droppedFeatures || [];
    const droppedStr = dropped.length
      ? dropped.slice(0, 15).map(escapeHtml).join('、') + (dropped.length > 15 ? ` …+${dropped.length - 15}` : '')
      : '無';
    rows.push(`<div class="text-xs px-3 py-2 rounded-r border-l-2 bg-primary-500/10 border-primary-500 text-primary-300">
      ✓ MI 特徵選擇:保留 <b>${mi.nKept}/${mi.nIn}</b> 個特徵 (threshold=${mi.threshold})${mi.nDropped > 0 ? `,丟棄:<span class="text-dark-300 font-mono">${droppedStr}</span>` : ''}
    </div>`);
  } else if (opts.useMiSelection) {
    rows.push(`<div class="text-xs px-3 py-2 rounded-r border-l-2 bg-warning-500/10 border-warning-500 text-warning-300">MI 特徵選擇已開,但沒有篩選結果</div>`);
  } else {
    rows.push(`<div class="text-xs px-3 py-2 rounded-r border-l-2 bg-dark-800/40 border-dark-600 text-dark-400">○ MI 特徵選擇未啟用</div>`);
  }

  // 4. 對抗驗證結果
  const adv = res.adversarialReport;
  if (!adv) {
    rows.push(`<div class="text-xs px-3 py-2 rounded-r border-l-2 bg-dark-800/40 border-dark-600 text-dark-400">○ 對抗驗證未執行（未上傳 test.csv）</div>`);
  } else if (adv.verdict === 'error') {
    rows.push(`<div class="text-xs px-3 py-2 rounded-r border-l-2 bg-warning-500/10 border-warning-500 text-warning-300">⚠ 對抗驗證執行失敗：${escapeHtml(adv.message || '')}</div>`);
  } else {
    const auc = typeof adv.auc_mean === 'number' ? adv.auc_mean.toFixed(4) : '—';
    const nDropped = adv.n_dropped ?? (adv.features_to_drop || []).length;
    const verdictColor = adv.verdict === 'ok'
      ? 'bg-success-500/10 border-success-500 text-success-300'
      : adv.verdict === 'warning'
        ? 'bg-warning-500/10 border-warning-500 text-warning-300'
        : 'bg-danger-500/10 border-danger-500 text-danger-300';
    const verdictIcon = adv.verdict === 'ok' ? '✓' : adv.verdict === 'warning' ? '⚠' : '✕';
    const droppedFeats = adv.features_to_drop || [];
    const droppedStr = droppedFeats.length
      ? droppedFeats.slice(0, 10).map(escapeHtml).join('、') + (droppedFeats.length > 10 ? ` …+${droppedFeats.length - 10}` : '')
      : '無';
    rows.push(`<div class="text-xs px-3 py-2 rounded-r border-l-2 ${verdictColor}">
      ${verdictIcon} 對抗驗證：AUC = <b>${auc}</b>　${escapeHtml(adv.message || '')}
      ${nDropped > 0
        ? `<div class="mt-1 text-dark-300">已移除 <b>${nDropped}</b> 個漂移特徵：<span class="font-mono">${droppedStr}</span></div>`
        : `<div class="mt-1 text-dark-400">無特徵被移除</div>`}
    </div>`);
  }

  body.innerHTML = rows.join('');
  card.classList.remove('hidden');
}

// v3:標籤雜訊偵測 (CV 預測信心度過低 → 疑似標錯)
function renderAuditLabelNoise(candidates) {
  const card = document.getElementById('pp-labelnoise-card');
  const tbody = document.getElementById('pp-labelnoise-table');
  const countEl = document.getElementById('pp-labelnoise-count');
  if (!card || !tbody) return;
  if (!candidates || candidates.length === 0) { card.classList.add('hidden'); return; }
  card.classList.remove('hidden');
  if (countEl) { countEl.textContent = `${candidates.length} 筆`; countEl.classList.remove('hidden'); }
  // 信心度最低的排前面;最多顯示 50 筆避免爆量
  const sorted = [...candidates].sort((a, b) => (a.predicted_proba ?? 0) - (b.predicted_proba ?? 0)).slice(0, 50);
  tbody.innerHTML = '';
  sorted.forEach(c => {
    const tr = document.createElement('tr');
    tr.className = 'border-b border-dark-800/50';
    const prob = ((c.predicted_proba ?? 0) * 100).toFixed(1);
    tr.innerHTML = `
      <td class="py-1.5 px-3 font-mono text-dark-300">#${c.row_index}</td>
      <td class="py-1.5 px-3 font-mono">${escapeHtml(String(c.true_label))}</td>
      <td class="py-1.5 px-3 text-right font-mono text-warning-400">${prob}%</td>
    `;
    tbody.appendChild(tr);
  });
  if (candidates.length > 50) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td colspan="3" class="py-1.5 px-3 text-center text-[11px] text-dark-500">… 還有 ${candidates.length - 50} 筆 (僅顯示信心度最低的 50 筆)</td>`;
    tbody.appendChild(tr);
  }
}

// v3:集成離群偵測 (IQR + IsolationForest + LOF 投票,≥2 票)
function renderAuditEnsembleOutliers(rows) {
  const card = document.getElementById('pp-ensemble-outliers-card');
  const tbody = document.getElementById('pp-ensemble-outliers-table');
  const countEl = document.getElementById('pp-ensemble-outliers-count');
  if (!card || !tbody) return;
  if (!rows || rows.length === 0) { card.classList.add('hidden'); return; }
  card.classList.remove('hidden');
  if (countEl) { countEl.textContent = `${rows.length} 筆`; countEl.classList.remove('hidden'); }
  // 票數最高的排前面;最多 50 筆
  const sorted = [...rows].sort((a, b) => (b.votes ?? 0) - (a.votes ?? 0)).slice(0, 50);
  tbody.innerHTML = '';
  sorted.forEach(r => {
    const tr = document.createElement('tr');
    tr.className = 'border-b border-dark-800/50';
    const methods = (r.methods || []).map(m =>
      `<span class="text-[10px] px-1.5 py-0.5 rounded bg-dark-700 text-dark-300 mr-1 font-mono">${escapeHtml(m)}</span>`).join('');
    tr.innerHTML = `
      <td class="py-1.5 px-3 font-mono text-dark-300">#${r.row_index}</td>
      <td class="py-1.5 px-3 text-center"><span class="font-mono ${(r.votes >= 3) ? 'text-danger-400' : 'text-warning-400'}">${r.votes}</span></td>
      <td class="py-1.5 px-3">${methods}</td>
    `;
    tbody.appendChild(tr);
  });
  if (rows.length > 50) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td colspan="3" class="py-1.5 px-3 text-center text-[11px] text-dark-500">… 還有 ${rows.length - 50} 筆 (僅顯示票數最高的 50 筆)</td>`;
    tbody.appendChild(tr);
  }
}

function renderAuditTargetInfo(target) {
  const card = document.getElementById('pp-target-card');
  if (!card) return;
  if (!target || !target.column) {
    card.classList.add('hidden');
    return;
  }
  card.classList.remove('hidden');
  document.getElementById('pp-target-col').textContent = target.column;
  document.getElementById('pp-target-dtype').textContent = target.dtype || '—';
  document.getElementById('pp-target-unique').textContent = target.unique_values ?? '—';
  document.getElementById('pp-target-missing').textContent = target.missing_count ?? 0;

  const badge = document.getElementById('pp-target-task-badge');
  const isClf = target.task_type === 'classification';
  badge.textContent = isClf ? '分類' : '迴歸';
  badge.className = 'text-[10px] px-2 py-0.5 rounded border ' + (isClf
    ? 'border-amber-500/30 text-amber-400 bg-amber-500/10'
    : 'border-blue-500/30 text-blue-400 bg-blue-500/10');

  // 詳細區:分類顯示類別分布,迴歸顯示數值統計
  const detail = document.getElementById('pp-target-detail');
  detail.innerHTML = '';
  if (isClf && target.class_distribution) {
    const entries = Object.entries(target.class_distribution);
    const wrap = document.createElement('div');
    wrap.className = 'space-y-1';
    entries.forEach(([cls, ratio]) => {
      const pct = (ratio * 100).toFixed(1);
      const row = document.createElement('div');
      row.className = 'flex items-center gap-2';
      row.innerHTML = `
        <span class="text-[11px] font-mono text-dark-300 w-20 truncate" title="${escapeHtml(cls)}">${escapeHtml(cls)}</span>
        <div class="flex-1 bg-dark-800 rounded-full h-1.5 overflow-hidden">
          <div class="h-full bg-primary-500" style="width:${pct}%"></div>
        </div>
        <span class="text-[11px] font-mono text-dark-400 w-12 text-right">${pct}%</span>
      `;
      wrap.appendChild(row);
    });
    detail.appendChild(wrap);
  } else if (!isClf && target.numeric_stats) {
    const s = target.numeric_stats;
    detail.innerHTML = `
      <div class="grid grid-cols-5 gap-2 text-[11px]">
        <div class="text-center bg-dark-800/50 rounded p-1.5">
          <p class="text-dark-500 mb-0.5">min</p>
          <p class="font-mono text-dark-200">${s.min}</p>
        </div>
        <div class="text-center bg-dark-800/50 rounded p-1.5">
          <p class="text-dark-500 mb-0.5">median</p>
          <p class="font-mono text-dark-200">${s.median}</p>
        </div>
        <div class="text-center bg-dark-800/50 rounded p-1.5">
          <p class="text-dark-500 mb-0.5">mean</p>
          <p class="font-mono text-dark-200">${s.mean}</p>
        </div>
        <div class="text-center bg-dark-800/50 rounded p-1.5">
          <p class="text-dark-500 mb-0.5">max</p>
          <p class="font-mono text-dark-200">${s.max}</p>
        </div>
        <div class="text-center bg-dark-800/50 rounded p-1.5">
          <p class="text-dark-500 mb-0.5">std</p>
          <p class="font-mono text-dark-200">${s.std}</p>
        </div>
      </div>
    `;
  }
}

function renderFeatureGroups(groups) {
  if (!groups || groups.error) return;
  const types = ['numeric', 'categorical', 'high_cardinality', 'text', 'datetime'];
  const chipColors = {
    numeric: 'bg-blue-500/15 text-blue-300 border-blue-500/30',
    categorical: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
    high_cardinality: 'bg-rose-500/15 text-rose-300 border-rose-500/30',
    text: 'bg-purple-500/15 text-purple-300 border-purple-500/30',
    datetime: 'bg-green-500/15 text-green-300 border-green-500/30',
  };
  types.forEach(t => {
    const cols = groups[t] || [];
    const countEl = document.getElementById(`pp-group-${t}-count`);
    const wrap = document.getElementById(`pp-group-${t}`);
    if (!countEl || !wrap) return;
    countEl.textContent = cols.length;
    wrap.innerHTML = '';
    if (cols.length === 0) {
      wrap.innerHTML = '<span class="text-[10px] text-dark-600">(無)</span>';
    } else {
      cols.forEach(c => {
        const chip = document.createElement('span');
        chip.className = `text-[10px] px-1.5 py-0.5 rounded border ${chipColors[t]} font-mono`;
        chip.textContent = c;
        chip.title = c;
        wrap.appendChild(chip);
      });
    }
  });

  // dropped 群組 — 顯示被 router 自動剔除的欄位 (常數欄、ID 欄、缺失過高、inf 等)
  const droppedSection = document.getElementById('pp-dropped-section');
  const droppedWrap = document.getElementById('pp-dropped-chips');
  if (droppedSection && droppedWrap) {
    const dropped = groups.dropped || [];
    if (dropped.length === 0) {
      droppedSection.classList.add('hidden');
    } else {
      droppedSection.classList.remove('hidden');
      droppedWrap.innerHTML = '';
      dropped.forEach(c => {
        const chip = document.createElement('span');
        chip.className = 'text-[10px] px-1.5 py-0.5 rounded border border-dark-600 bg-dark-800 text-dark-400 line-through font-mono';
        chip.textContent = c;
        chip.title = `${c} (已剔除)`;
        droppedWrap.appendChild(chip);
      });
    }
  }
}

function renderTransformResult(res) {
  document.getElementById('pp-result-train').textContent = res.trainSize;
  document.getElementById('pp-result-test').textContent = res.testSize;
  document.getElementById('pp-result-feat').textContent = `${res.originalFeatureCount} → ${res.transformedFeatureCount}`;
  document.getElementById('pp-result-id').textContent = res.preprocessorId;

  const thead = document.getElementById('pp-result-thead');
  const tbody = document.getElementById('pp-result-tbody');
  thead.innerHTML = '';
  tbody.innerHTML = '';
  const cols = res.preview?.columns || [];
  const rows = res.preview?.rows || [];

  // Header — abbreviate long names with title tooltip
  thead.innerHTML = cols.map(c => {
    const short = c.length > 22 ? c.slice(0, 20) + '…' : c;
    return `<th class="text-left py-1 px-2 font-medium text-[10px] whitespace-nowrap" title="${escapeHtml(c)}">${escapeHtml(short)}</th>`;
  }).join('');

  rows.forEach(row => {
    const tr = document.createElement('tr');
    tr.className = 'border-b border-dark-800/50 hover:bg-dark-800/30';
    tr.innerHTML = row.map(v => {
      const num = typeof v === 'number' ? v.toFixed(3) : String(v);
      return `<td class="py-0.5 px-2 font-mono text-dark-300 whitespace-nowrap">${escapeHtml(num)}</td>`;
    }).join('');
    tbody.appendChild(tr);
  });

  // 收合 / 展開 toggle (預設收合)
  const wrap = document.getElementById('pp-result-table-wrap');
  const btn = document.getElementById('btn-pp-result-toggle');
  wrap.classList.add('hidden');
  btn.textContent = '展開預覽';
  btn.onclick = (e) => {
    e.stopPropagation();
    const hidden = wrap.classList.toggle('hidden');
    btn.textContent = hidden ? '展開預覽' : '收合預覽';
  };

  // 下載按鈕 — 直接導到 backend 的下載 endpoint
  const dl = (split) => {
    if (!ppLastPreprocessorId) { alert('請先執行預處理'); return; }
    const url = `${ApiClient.baseUrl}/api/preprocess/download/${ppLastPreprocessorId}/${split}`;
    // 用隱藏 <a download> 觸發瀏覽器下載
    const a = document.createElement('a');
    a.href = url;
    a.download = `${split}_${ppLastPreprocessorId}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };
  document.getElementById('btn-pp-download-train').onclick = (e) => { e.stopPropagation(); dl('train'); };
  document.getElementById('btn-pp-download-test').onclick = (e) => { e.stopPropagation(); dl('test'); };
}

function renderInferenceForm(ds, target) {
  const wrap = document.getElementById('pp-inference-inputs');
  const wrapOuter = document.getElementById('pp-inference-inputs-wrap');
  const hint = document.getElementById('pp-inference-hint');
  const toggleBtn = document.getElementById('btn-pp-inference-toggle');
  const searchInput = document.getElementById('pp-inference-search');
  const outputsWrap = document.getElementById('pp-inference-outputs');
  const inputCountEl = document.getElementById('pp-input-count');
  const outputCountEl = document.getElementById('pp-output-count');

  wrap.innerHTML = '';
  outputsWrap.innerHTML = '<p class="text-[11px] text-dark-500 px-2 py-1">點擊下方「轉換這筆資料」後會顯示處理結果。</p>';
  outputCountEl.textContent = '尚未轉換';

  const inputCols = (ds.headers || []).filter(h => h !== target);
  inputCountEl.textContent = `${inputCols.length} 個欄位`;

  inputCols.forEach(col => {
    const colMeta = (ds.analysis || []).find(c => c.name === col);
    const sample = ds.data?.[0]?.[ds.headers.indexOf(col)] ?? '';
    const row = document.createElement('div');
    row.dataset.ppCol = col.toLowerCase();
    row.className = 'flex items-center gap-2';
    row.innerHTML = `
      <span class="text-[10px] text-dark-400 truncate w-28 flex-shrink-0" title="${escapeHtml(col)}">${escapeHtml(col)} <span class="text-dark-600">${colMeta?.type || ''}</span></span>
      <input type="text" data-pp-input="${escapeHtml(col)}" value="${escapeHtml(String(sample))}"
        class="flex-1 min-w-0 bg-dark-800 border border-dark-600 rounded px-1.5 py-1 text-[11px] font-mono focus:border-accent-500 outline-none">
    `;
    wrap.appendChild(row);
  });

  // 大量欄位預設收合
  const manyCols = inputCols.length > 12;
  wrapOuter.classList.toggle('hidden', manyCols);
  toggleBtn.textContent = manyCols ? `展開 (${inputCols.length})` : `收合`;
  hint.textContent = manyCols
    ? `共 ${inputCols.length} 個欄位,預設值已用第一筆資料帶入。可直接按「轉換這筆資料」,或展開後修改。`
    : `預設值已自動帶入第一筆資料,可直接修改後轉換。`;

  toggleBtn.onclick = () => {
    const hidden = wrapOuter.classList.toggle('hidden');
    toggleBtn.textContent = hidden ? `展開 (${inputCols.length})` : `收合`;
  };
  searchInput.value = '';
  searchInput.oninput = () => {
    const q = searchInput.value.trim().toLowerCase();
    wrap.querySelectorAll('[data-pp-col]').forEach(el => {
      el.style.display = !q || el.dataset.ppCol.includes(q) ? '' : 'none';
    });
    // 右側也跟著過濾 (用 prefix 比對)
    outputsWrap.querySelectorAll('[data-pp-out]').forEach(el => {
      el.style.display = !q || el.dataset.ppOut.includes(q) ? '' : 'none';
    });
  };
}

async function runPreprocessInference() {
  if (!ppLastPreprocessorId) { alert('請先執行預處理'); return; }
  const inputs = document.querySelectorAll('[data-pp-input]');
  const row = {};
  inputs.forEach(el => {
    const v = el.value;
    const num = Number(v);
    row[el.dataset.ppInput] = (v !== '' && !isNaN(num)) ? num : v;
  });
  ppSetStatus(true, '套用 preprocessor...');
  try {
    const res = await ApiClient.preprocessInference({
      preprocessorId: ppLastPreprocessorId,
      rows: [row],
    });
    renderInferenceOutput(res.columns || [], res.rows[0] || []);
    // 確保右欄展開可見 (若 input 在收合狀態)
    document.getElementById('pp-inference-inputs-wrap').classList.remove('hidden');
    document.getElementById('btn-pp-inference-toggle').textContent = '收合';
  } catch (e) {
    alert(`Inference 失敗: ${e.message}`);
  } finally {
    ppSetStatus(false);
  }
}

function renderInferenceOutput(columns, values) {
  const wrap = document.getElementById('pp-inference-outputs');
  const countEl = document.getElementById('pp-output-count');
  wrap.innerHTML = '';
  countEl.textContent = `${values.length} 維`;

  if (values.length === 0) {
    wrap.innerHTML = '<p class="text-[11px] text-dark-500 px-2 py-1">無輸出</p>';
    return;
  }

  values.forEach((v, i) => {
    const name = columns[i] || `[${i}]`;
    // 把 sklearn pipeline prefix 拆出來,讓 UI 更乾淨
    // e.g. "num_pipeline__age" → group="num", short="age"
    const m = name.match(/^(num|cat|text|time)_pipeline__(.+)$/);
    const group = m ? m[1] : '';
    const short = m ? m[2] : name;
    const groupColors = {
      num: 'text-blue-400',
      cat: 'text-amber-400',
      text: 'text-purple-400',
      time: 'text-green-400',
    };
    const groupColor = groupColors[group] || 'text-dark-500';
    const num = typeof v === 'number' ? v.toFixed(4) : String(v);

    const row = document.createElement('div');
    row.className = 'flex items-center gap-2';
    row.dataset.ppOut = name.toLowerCase();
    row.innerHTML = `
      <span class="text-[10px] text-dark-500 w-6 flex-shrink-0 font-mono">[${i}]</span>
      <span class="text-[10px] ${groupColor} truncate flex-1 min-w-0 font-mono" title="${escapeHtml(name)}">${group ? `<span class="text-dark-600">${group}·</span>` : ''}${escapeHtml(short)}</span>
      <span class="text-[11px] font-mono text-accent-400 w-20 text-right flex-shrink-0">${escapeHtml(num)}</span>
    `;
    wrap.appendChild(row);
  });
}

// ===== SHAP SAMPLE SELECT =====
function initShapSampleSelect() {
  const select = document.getElementById('shap-sample-select');
  if (select) {
    select.addEventListener('change', () => { renderShapWaterfall(parseInt(select.value)); });
  }
}

// ===== REAL EXPERIMENTS =====
function renderRealExperimentsPage() {
  const ds = DataEngine.currentDataset;
  const noData = document.getElementById('exp-no-data');
  const config = document.getElementById('exp-real-config');

  if (!ds) {
    noData.classList.remove('hidden');
    config.classList.add('hidden');
    return;
  }

  noData.classList.add('hidden');
  config.classList.remove('hidden');

  // Dataset badge
  document.getElementById('exp-dataset-badge').textContent = ds.fileName || 'Dataset';

  // Dataset picker — 列出已上傳的所有 CSV,選了就 switchDataset + 重新建立 form
  // 不用 cloneNode (它會掃掉 select 的 value);改用 flag 確保 listener 只綁一次
  const dsSelect = document.getElementById('exp-dataset-select');
  const dsInfo = document.getElementById('exp-dataset-info');
  if (dsSelect) {
    dsSelect.innerHTML = '';
    // 同名 dataset 用上傳時間後綴區分,避免下拉看不出誰是誰
    const nameCounts = {};
    DataEngine.datasets.forEach(d => { nameCounts[d.fileName] = (nameCounts[d.fileName] || 0) + 1; });
    DataEngine.datasets.forEach(d => {
      const opt = document.createElement('option');
      opt.value = d.id;
      if (nameCounts[d.fileName] > 1) {
        // 同名 → 加上傳時間 (loadedAt 可能是 Date 或 timestamp)
        let when = '';
        try {
          const t = d.loadedAt instanceof Date ? d.loadedAt : new Date(typeof d.loadedAt === 'number' ? d.loadedAt * (d.loadedAt < 1e12 ? 1000 : 1) : d.loadedAt);
          when = `${t.getMonth() + 1}/${t.getDate()} ${String(t.getHours()).padStart(2,'0')}:${String(t.getMinutes()).padStart(2,'0')}`;
        } catch (e) {}
        opt.textContent = when ? `${d.fileName} (${when})` : `${d.fileName} (${String(d.id).slice(-4)})`;
      } else {
        opt.textContent = d.fileName;
      }
      dsSelect.appendChild(opt);
    });
    // 用 select.value 同步當前 dataset (cloneNode 會掉,這個不會)
    dsSelect.value = ds.id;
    if (dsInfo) dsInfo.textContent = `${ds.rowCount.toLocaleString()} 筆 × ${ds.colCount} 欄`;
    if (!dsSelect.__expDsWired) {
      dsSelect.__expDsWired = true;
      dsSelect.addEventListener('change', async e => {
        const newId = e.target.value;
        const cur = DataEngine.currentDataset;
        if (!cur || newId !== cur.id) {
          const target = DataEngine.datasets.find(d => d.id === newId);
          await hydrateDatasetIfStub(target);
          DataEngine.switchDataset(newId);
          renderRealExperimentsPage();
        }
      });
    }
  }

  // Populate target select with all columns (numeric + categorical + boolean)
  const targetSel = document.getElementById('exp-target-select');
  targetSel.innerHTML = '';
  ds.analysis.forEach(col => {
    const opt = document.createElement('option');
    opt.value = col.name;
    opt.textContent = `${col.name} (${col.type})`;
    targetSel.appendChild(opt);
  });
  // 自動選常見目標欄名；找不到則保持第一欄
  const _TARGET_PRIO = ['target', 'label', 'class', 'y', 'c'];
  const _autoTarget = _TARGET_PRIO.find(n => ds.analysis.some(c => c.name === n));
  if (_autoTarget) targetSel.value = _autoTarget;

  // --- Feature checkboxes ---
  const featBox = document.getElementById('exp-feature-checkboxes');
  const featCountEl = document.getElementById('exp-feature-count');

  // 算每個數值特徵跟 target 的 |Pearson r| → 洩漏風險。
  //   |r| >= 0.95 高風險 (幾乎確定洩漏);0.85~0.95 中風險。
  const computeFeatureRisks = (targetCol) => {
    const risks = {};
    const tIdx = ds.headers ? ds.headers.indexOf(targetCol) : -1;
    if (tIdx < 0 || !ds.data || ds.data.length < 10) return risks;
    const N = Math.min(ds.data.length, 5000);   // 取樣上限,大資料才不會卡
    const yv = new Array(N);
    for (let i = 0; i < N; i++) yv[i] = parseFloat(ds.data[i][tIdx]);
    ds.analysis.filter(c => c.type === 'numeric' && c.name !== targetCol).forEach(col => {
      const cIdx = ds.headers.indexOf(col.name);
      if (cIdx < 0) return;
      let n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
      for (let i = 0; i < N; i++) {
        const x = parseFloat(ds.data[i][cIdx]), y = yv[i];
        if (!isFinite(x) || !isFinite(y)) continue;
        n++; sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
      }
      if (n < 10) return;
      const cov = sxy - sx * sy / n, vx = sxx - sx * sx / n, vy = syy - sy * sy / n;
      const r = (vx > 0 && vy > 0) ? cov / Math.sqrt(vx * vy) : 0;
      const ar = Math.abs(r);
      if (ar >= 0.95) risks[col.name] = { level: 'high', corr: r };
      else if (ar >= 0.85) risks[col.name] = { level: 'medium', corr: r };
    });
    return risks;
  };

  const buildFeatureCheckboxes = () => {
    const targetCol = targetSel.value;
    const numericCols = ds.analysis.filter(c => c.type === 'numeric' && c.name !== targetCol);
    const dateCols = ds.analysis.filter(c => c.type === 'datetime' && c.name !== targetCol);
    const risks = computeFeatureRisks(targetCol);
    featBox.innerHTML = '';
    // Numeric features (帶風險標示)
    numericCols.forEach(col => {
      const risk = risks[col.name];
      const lbl = document.createElement('label');
      lbl.className = 'flex items-center gap-1.5 text-xs text-dark-300 cursor-pointer hover:text-dark-100 transition-colors';
      lbl.dataset.risk = risk ? risk.level : '';
      let badge = '';
      if (risk && risk.level === 'high') {
        badge = `<span class="text-[9px] px-1 py-0.5 rounded bg-danger-500/20 text-danger-300 font-semibold ml-1" title="與 target 相關係數 ${risk.corr.toFixed(3)},極可能洩漏">高風險</span>`;
      } else if (risk && risk.level === 'medium') {
        badge = `<span class="text-[9px] px-1 py-0.5 rounded bg-warning-500/20 text-warning-300 font-semibold ml-1" title="與 target 相關係數 ${risk.corr.toFixed(3)},可能洩漏">中風險</span>`;
      }
      lbl.innerHTML = `<input type="checkbox" class="exp-feat-cb accent-primary-500" value="${escapeHtml(col.name)}" checked> ${escapeHtml(col.name)}${badge}`;
      featBox.appendChild(lbl);
    });
    // Date-derived features
    dateCols.forEach(col => {
      const dateFeats = MLEngine.getDateFeatureNames(col.name);
      const dateLabels = {
        year: '年份', month: '月份', day_of_year: '年中日',
        month_sin: '月份(sin)', month_cos: '月份(cos)',
        day_sin: '日(sin)', day_cos: '日(cos)',
      };
      dateFeats.forEach(fn => {
        const suffix = fn.replace(col.name + '_', '');
        const label = dateLabels[suffix] || suffix;
        const lbl = document.createElement('label');
        lbl.className = 'flex items-center gap-1.5 text-xs text-dark-300 cursor-pointer hover:text-dark-100 transition-colors';
        lbl.dataset.risk = '';   // 日期衍生特徵不算洩漏風險
        lbl.innerHTML = `<input type="checkbox" class="exp-feat-cb accent-primary-500" value="${escapeHtml(fn)}" checked> <span class="text-cyan-400">${escapeHtml(col.name)}</span>_${escapeHtml(label)}`;
        featBox.appendChild(lbl);
      });
    });
    updateFeatureCount();
    // 刷新風險按鈕顯示 (有偵測到該等級才顯示) — inline 避免 const TDZ
    const _hasHigh = featBox.querySelector('label[data-risk="high"]');
    const _hasMed = featBox.querySelector('label[data-risk="medium"]');
    document.getElementById('exp-feat-drop-high')?.classList.toggle('hidden', !_hasHigh);
    document.getElementById('exp-feat-drop-med')?.classList.toggle('hidden', !_hasMed);
  };

  const updateFeatureCount = () => {
    const total = featBox.querySelectorAll('.exp-feat-cb').length;
    const checked = featBox.querySelectorAll('.exp-feat-cb:checked').length;
    featCountEl.textContent = `(${checked}/${total})`;
  };

  featBox.addEventListener('change', updateFeatureCount);
  document.getElementById('exp-feat-all').onclick = () => { featBox.querySelectorAll('.exp-feat-cb').forEach(cb => cb.checked = true); updateFeatureCount(); };
  document.getElementById('exp-feat-none').onclick = () => { featBox.querySelectorAll('.exp-feat-cb').forEach(cb => cb.checked = false); updateFeatureCount(); };

  // 取消高/中風險特徵 — 依 label 的 data-risk 取消勾選。只有偵測到該等級時才顯示按鈕。
  const dropByRisk = (level) => {
    featBox.querySelectorAll('label[data-risk="' + level + '"] .exp-feat-cb').forEach(cb => { cb.checked = false; });
    updateFeatureCount();
    saveExpFormState();
  };
  document.getElementById('exp-feat-drop-high').onclick = () => dropByRisk('high');
  document.getElementById('exp-feat-drop-med').onclick = () => dropByRisk('medium');

  // 搜尋欄位 — 即時過濾下方 checkbox
  const featSearch = document.getElementById('exp-feature-search');
  if (featSearch) {
    featSearch.value = '';
    featSearch.oninput = () => {
      const q = featSearch.value.trim().toLowerCase();
      featBox.querySelectorAll('label').forEach(lbl => {
        const cb = lbl.querySelector('.exp-feat-cb');
        const name = (cb?.value || lbl.textContent || '').toLowerCase();
        lbl.style.display = !q || name.includes(q) ? '' : 'none';
      });
    };
  }

  // --- Algorithm checkboxes ---
  const algoBox = document.getElementById('exp-algo-checkboxes');
  const algoCountEl = document.getElementById('exp-algo-count');

  const buildAlgoCheckboxes = () => {
    algoBox.innerHTML = '';
    const activeAlgos = getSettings().activeAlgos;  // 系統設定的預設啟用演算法
    Object.entries(MLEngine.ALGORITHMS).forEach(([key, algo]) => {
      const lbl = document.createElement('label');
      lbl.className = 'flex items-center gap-1.5 text-xs text-dark-300 cursor-pointer hover:text-dark-100 transition-colors';
      const typeTag = algo.type === 'regression' ? '回歸' : algo.type === 'classification' ? '分類' : '通用';
      const tagColor = algo.type === 'regression' ? 'text-blue-400' : algo.type === 'classification' ? 'text-amber-400' : 'text-green-400';
      const checked = activeAlgos.includes(key) ? 'checked' : '';
      lbl.innerHTML = `<input type="checkbox" class="exp-algo-cb accent-primary-500" value="${key}" data-algo-type="${algo.type}" ${checked}> ${escapeHtml(algo.label)} <span class="${tagColor} text-[10px]">(${typeTag})</span>`;
      algoBox.appendChild(lbl);
    });
    updateAlgoCount();
  };

  const updateAlgoCount = () => {
    const total = algoBox.querySelectorAll('.exp-algo-cb').length;
    const checked = algoBox.querySelectorAll('.exp-algo-cb:checked').length;
    algoCountEl.textContent = `(${checked}/${total})`;
  };

  algoBox.addEventListener('change', updateAlgoCount);
  document.getElementById('exp-algo-all').onclick = () => { algoBox.querySelectorAll('.exp-algo-cb').forEach(cb => cb.checked = true); updateAlgoCount(); };
  document.getElementById('exp-algo-none').onclick = () => { algoBox.querySelectorAll('.exp-algo-cb').forEach(cb => cb.checked = false); updateAlgoCount(); };
  document.getElementById('exp-algo-compatible').onclick = () => {
    const taskType = getSelectedTaskType();
    algoBox.querySelectorAll('.exp-algo-cb').forEach(cb => {
      const aType = cb.dataset.algoType;
      cb.checked = (aType === 'both' || aType === taskType);
    });
    updateAlgoCount();
  };

  // --- Task type detection ---
  const updateTargetInfo = () => {
    const colName = targetSel.value;
    const col = ds.analysis.find(c => c.name === colName);
    if (!col) return;
    const colIdx = ds.headers.indexOf(colName);
    const vals = ds.data.map(r => r[colIdx]);
    const uniqueVals = new Set(vals).size;
    const detectedType = uniqueVals <= 10 ? 'classification' : 'regression';
    const label = detectedType === 'classification' ? `分類 (${uniqueVals} 類)` : '回歸 (連續值)';
    document.getElementById('exp-task-label').textContent = label;
    document.getElementById('exp-target-info').textContent = col.stats
      ? `範圍: ${col.stats.min} ~ ${col.stats.max}, 平均: ${col.stats.mean}`
      : '';

    // 不平衡偵測：分類任務才檢查
    const cwRow = document.getElementById('exp-class-weight-row');
    const imbalanceHint = document.getElementById('exp-imbalance-hint');
    if (detectedType === 'classification' && uniqueVals >= 2 && uniqueVals <= 10) {
      // 計算各類別頻率
      const counts = {};
      vals.forEach(v => { counts[v] = (counts[v] || 0) + 1; });
      const freqs = Object.values(counts);
      const maxF = Math.max(...freqs), minF = Math.min(...freqs);
      const ratio = maxF / Math.max(minF, 1);
      if (cwRow) cwRow.classList.remove('hidden');
      if (ratio >= 5) {
        // 嚴重不平衡：自動建議
        if (imbalanceHint) {
          imbalanceHint.textContent = `⚠ 偵測到類別不平衡（比例約 ${ratio.toFixed(0)}:1），建議啟用「類別平衡」選項`;
          imbalanceHint.classList.remove('hidden');
        }
      } else {
        if (imbalanceHint) imbalanceHint.classList.add('hidden');
      }
    } else {
      // 回歸任務：隱藏 class_weight 選項
      if (cwRow) cwRow.classList.add('hidden');
      if (imbalanceHint) imbalanceHint.classList.add('hidden');
    }

    // Rebuild feature checkboxes when target changes
    buildFeatureCheckboxes();
  };

  const getSelectedTaskType = () => {
    const radio = document.querySelector('input[name="exp-task-type"]:checked');
    if (!radio || radio.value === 'auto') {
      // Auto-detect
      const colName = targetSel.value;
      const uniqueVals = new Set(ds.data.map(r => r[ds.headers.indexOf(colName)])).size;
      return uniqueVals <= 10 ? 'classification' : 'regression';
    }
    return radio.value;
  };

  targetSel.addEventListener('change', updateTargetInfo);
  updateTargetInfo();
  buildAlgoCheckboxes();

  // --- 數據模式 radio (靜態/時序) 跟 #exp-time-series checkbox 雙向同步 ---
  // radio 是視覺主控,checkbox 是 JS 讀的單一真實來源
  const tsCheckbox = document.getElementById('exp-time-series');
  const dataModeRadios = document.querySelectorAll('input[name="data-mode"]');
  const dataModeLabels = document.querySelectorAll('.exp-data-mode-opt');
  const refreshDataModeVisual = (isTS) => {
    dataModeLabels.forEach(lbl => {
      const active = (lbl.dataset.mode === 'timeseries') === isTS;
      lbl.classList.toggle('bg-primary-500/10', active);
      lbl.classList.toggle('border-primary-500/30', active);
      lbl.classList.toggle('bg-dark-800', !active);
      lbl.classList.toggle('border-dark-600', !active);
    });
    dataModeRadios.forEach(r => {
      r.checked = (r.value === 'timeseries') === isTS;
    });
  };
  // radio change → 寫進 checkbox + 視覺
  dataModeRadios.forEach(r => r.addEventListener('change', () => {
    const isTS = r.value === 'timeseries' && r.checked;
    if (tsCheckbox) tsCheckbox.checked = isTS;
    refreshDataModeVisual(isTS);
  }));
  // checkbox change → 同步 radio + 視覺
  if (tsCheckbox) {
    tsCheckbox.addEventListener('change', () => refreshDataModeVisual(tsCheckbox.checked));
    // init: radio 跟 checkbox 對齊
    refreshDataModeVisual(tsCheckbox.checked);
  }

  // --- Data source (multi-select) ---
  const srcRaw = document.getElementById('exp-src-raw');
  const srcPp = document.getElementById('exp-src-pp');
  const ppSelect = document.getElementById('exp-pp-select');
  const ppHint = document.getElementById('exp-pp-hint');
  let ppMatches = [];   // 當前資料集可用的 preprocessor 清單

  // 選了 preprocessor → 目標變數自動同步成它的 target (兩邊一致才能比較)
  const syncTargetToPreprocessor = () => {
    if (!srcPp.checked || !ppSelect.value) {
      if (ppHint) ppHint.textContent = '';
      return;
    }
    const sel = ppMatches.find(p => p.id === ppSelect.value);
    if (!sel) return;
    const inDropdown = [...targetSel.options].some(o => o.value === sel.target);
    if (inDropdown) {
      if (targetSel.value !== sel.target) {
        targetSel.value = sel.target;
        updateTargetInfo();
      }
      if (ppHint) ppHint.textContent = `目標變數已同步為「${sel.target}」(預處理時設定的)`;
    } else {
      // preprocessor 的目標不在實驗室目標下拉裡 (例如分類目標) — 警告
      if (ppHint) ppHint.textContent = `⚠ 此預處理的目標「${sel.target}」不在目標下拉中,原始資料集那條路會用不同目標`;
    }
  };

  const ppInfoBtn = document.getElementById('exp-pp-info-btn');
  const ppInfoPop = document.getElementById('exp-pp-info-popover');

  const renderPpInfoPopover = () => {
    if (!ppInfoPop) return;
    const sel = ppMatches.find(p => p.id === ppSelect.value);
    if (!sel) { ppInfoPop.innerHTML = ''; return; }
    const stamp = sel.createdAt ? _formatPreprocessTime(sel.createdAt * 1000) : sel.id;
    const yes = '<span class="text-emerald-400">✓ 啟用</span>';
    const no  = '<span class="text-dark-400">✗ 未啟用</span>';
    ppInfoPop.innerHTML = `
      <div class="font-medium text-dark-100 mb-1.5">${escapeHtml(stamp)}</div>
      <div class="flex justify-between gap-2"><span class="text-dark-400">ID</span><span class="font-mono text-[10px] text-dark-300 truncate">${escapeHtml(sel.id)}</span></div>
      <div class="flex justify-between gap-2"><span class="text-dark-400">目標</span><span class="text-dark-200">${escapeHtml(sel.target || '-')}</span></div>
      <div class="flex justify-between gap-2"><span class="text-dark-400">特徵數</span><span class="text-dark-200">${sel.featureCount}</span></div>
      <div class="flex justify-between gap-2"><span class="text-dark-400">Train / Test</span><span class="text-dark-200">${sel.trainSize} / ${sel.testSize}</span></div>
      <div class="border-t border-dark-600 my-1.5"></div>
      <div class="flex justify-between gap-2"><span class="text-dark-400">MICE 補值</span>${sel.useMice ? yes : no}</div>
      <div class="flex justify-between gap-2"><span class="text-dark-400">MI 特徵選擇</span>${sel.useMiSelection ? yes : no}</div>
      ${sel.useMiSelection ? `<div class="flex justify-between gap-2"><span class="text-dark-400">MI 門檻</span><span class="text-dark-200">${sel.miThreshold}</span></div>` : ''}
    `;
  };

  const refreshPpSelect = async () => {
    // 直接問後端有哪些 preprocessor (不依賴前端記憶體,重整也不會掉)
    let all = [];
    let fetchErr = null;
    try {
      const res = await ApiClient.preprocessList();
      all = res.preprocessors || [];
    } catch (e) {
      fetchErr = e.message;
      all = ppHistory;  // fallback to in-memory
    }
    // 只列當前資料集的 preprocessor — 其他資料集 schema 不一樣,套錯只會炸,不顯示
    ppMatches = all.filter(p => p.datasetId === ds.id);

    ppSelect.innerHTML = '';
    if (ppMatches.length === 0) {
      const msg = fetchErr
        ? `無法取得預處理清單: ${fetchErr}`
        : '尚無預處理結果 — 請先到「預處理」頁面執行';
      ppSelect.innerHTML = `<option value="">${escapeHtml(msg)}</option>`;
      srcPp.checked = false;
      srcPp.disabled = true;
      ppSelect.disabled = true;
      if (ppHint) ppHint.textContent = '';
      if (ppInfoBtn) ppInfoBtn.disabled = true;
      if (ppInfoPop) ppInfoPop.classList.add('hidden');
    } else {
      srcPp.disabled = false;
      ppMatches.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p.id;
        // 用建立時間取代 ID 顯示 (人讀友善);沒 createdAt 的舊資料退回到 ID 顯示
        const stamp = p.createdAt ? _formatPreprocessTime(p.createdAt * 1000) : p.id;
        opt.textContent = `${stamp} — target=${p.target}, ${p.featureCount} 特徵 (train ${p.trainSize}/test ${p.testSize})`;
        opt.title = `preprocessorId: ${p.id}`;  // hover 還是看得到 ID
        ppSelect.appendChild(opt);
      });
      ppSelect.disabled = !srcPp.checked;
      if (ppInfoBtn) ppInfoBtn.disabled = false;
      syncTargetToPreprocessor();
      renderPpInfoPopover();
    }
  };
  refreshPpSelect();
  srcPp.addEventListener('change', () => {
    ppSelect.disabled = !srcPp.checked;
    syncTargetToPreprocessor();
  });
  ppSelect.addEventListener('change', () => {
    syncTargetToPreprocessor();
    renderPpInfoPopover();
    if (ppInfoPop) ppInfoPop.classList.add('hidden');
  });

  // i icon toggle popover + 點外面關閉
  if (ppInfoBtn && ppInfoPop) {
    ppInfoBtn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (ppInfoBtn.disabled) return;
      renderPpInfoPopover();
      ppInfoPop.classList.toggle('hidden');
    });
    document.addEventListener('click', (e) => {
      if (ppInfoPop.classList.contains('hidden')) return;
      if (!ppInfoPop.contains(e.target) && e.target !== ppInfoBtn) {
        ppInfoPop.classList.add('hidden');
      }
    });
  }

  // --- 套用系統設定的預設值 (任務類型 / 資料來源勾選) ---
  const _settings = getSettings();
  const taskRadio = document.querySelector(`input[name="exp-task-type"][value="${_settings.taskType}"]`);
  if (taskRadio) taskRadio.checked = true;
  srcRaw.checked = _settings.srcRaw;
  // 已預處理資料:只有後端真的有 preprocessor 時才依設定勾 (srcPp.disabled 由 refreshPpSelect 決定)
  if (!srcPp.disabled) srcPp.checked = _settings.srcPp;
  ppSelect.disabled = !srcPp.checked;

  // --- Pipeline 的 predict CSV picker — 顯示檔名 + 清除按鈕 ---
  const predictInputEl = document.getElementById('exp-pipeline-predict-csv');
  const predictInfoEl = document.getElementById('exp-pipeline-predict-info');
  const predictClearBtn = document.getElementById('exp-pipeline-predict-clear');
  if (predictInputEl && !predictInputEl.__wired) {
    predictInputEl.__wired = true;
    predictInputEl.addEventListener('change', () => {
      const f = predictInputEl.files?.[0];
      if (f && predictInfoEl) {
        predictInfoEl.textContent = `✓ ${f.name} (${(f.size / 1024).toFixed(1)} KB) — 訓練完成後將自動產生 submission.csv`;
        predictInfoEl.classList.remove('hidden');
      } else if (predictInfoEl) {
        predictInfoEl.classList.add('hidden');
      }
    });
    if (predictClearBtn) {
      predictClearBtn.addEventListener('click', () => {
        predictInputEl.value = '';
        predictInfoEl?.classList.add('hidden');
      });
    }
  }

  // --- 訓練引擎切換 (sklearn / daniel) ---
  // daniel 引擎:隱藏「演算法選擇」(Daniel 自己挑),顯示 daniel options;特徵欄位仍顯示但會
  // 註記「不適用」— Daniel 直接吃 raw 資料,內部做 select_dtypes(numeric)。
  const applyEngineUI = () => {
    const engineRadio = document.querySelector('input[name="exp-engine"]:checked');
    const engine = engineRadio ? engineRadio.value : 'sklearn';
    const danielOpts = document.getElementById('exp-daniel-options');
    const autogluonOpts = document.getElementById('exp-autogluon-options');
    const algoSection = algoBox.closest('.md\\:col-span-2');
    const featSection = featBox.closest('.md\\:col-span-2');
    // 預設都隱藏
    danielOpts?.classList.add('hidden');
    autogluonOpts?.classList.add('hidden');
    if (engine === 'daniel') {
      danielOpts?.classList.remove('hidden');
      algoSection?.classList.add('hidden');
      featSection?.classList.add('opacity-50');
      featSection?.setAttribute('title', 'Pipeline 引擎不使用此選項 — 自己挑特徵');
    } else if (engine === 'autogluon') {
      autogluonOpts?.classList.remove('hidden');
      algoSection?.classList.add('hidden');
      featSection?.classList.add('opacity-50');
      featSection?.setAttribute('title', 'Autogluon 內建自家 preprocessing + 自家挑模型,此選項不適用');
    } else {
      algoSection?.classList.remove('hidden');
      featSection?.classList.remove('opacity-50');
      featSection?.removeAttribute('title');
    }
  };
  document.querySelectorAll('input[name="exp-engine"]').forEach(r => {
    r.addEventListener('change', applyEngineUI);
  });
  applyEngineUI();

  // Train button — 用 cloneNode 重置 click handler。複製完後同步當下訓練狀態,
  // 避免切頁回來看到的是舊狀態 (cloneNode 雖會複製 innerHTML/disabled,但訓練中途
  // 完成時舊參考已被孤立、寫不到新 btn,所以這裡顯式 set 一次最安全)
  const btn = document.getElementById('btn-real-train');
  const newBtn = btn.cloneNode(true);
  btn.parentNode.replaceChild(newBtn, btn);
  setTrainBtnState(_trainingState);
  newBtn.addEventListener('click', () => {
    // 訓練中 / 取消中 → 按鈕已不是「開始訓練」,是「取消」,直接 dispatch 到 cancel 然後 return
    // (避免又跑一次 start training)
    if (_trainingState === 'training') {
      cancelCurrentTraining();
      return;
    }
    if (_trainingState === 'cancelling') return;  // 防連點

    const engineRadio = document.querySelector('input[name="exp-engine"]:checked');
    const engine = engineRadio ? engineRadio.value : 'sklearn';
    const taskTypeRadio = document.querySelector('input[name="exp-task-type"]:checked');
    const taskType = taskTypeRadio ? taskTypeRadio.value : 'auto';
    const timeSeries = document.getElementById('exp-time-series').checked;

    // 資料來源 (多選) — 兩個引擎共用
    const sources = [];
    if (srcRaw.checked) sources.push('raw');
    if (srcPp.checked) sources.push('preprocessed');
    if (sources.length === 0) { alert('請至少選擇一個資料來源'); return; }

    const preprocessorId = srcPp.checked ? ppSelect.value : null;
    if (srcPp.checked && !preprocessorId) { alert('已勾選「已預處理資料」,請選擇一個預處理結果'); return; }

    if (engine === 'daniel') {
      // Daniel 引擎:走 /api/train/pipeline/stream
      if (typeof ApiClient === 'undefined' || !ApiClient.enabled) {
        alert('Pipeline 引擎需要 Python 後端 API,請先在系統設定開啟「使用 Python 後端 API」'); return;
      }
      const predictInput = document.getElementById('exp-pipeline-predict-csv');
      const predictFile = predictInput?.files?.[0] || null;
      const danielOptions = {
        fast: document.getElementById('exp-daniel-fast')?.checked ?? true,
        skipDl: document.getElementById('exp-daniel-skip-dl')?.checked ?? false,
        noNas: document.getElementById('exp-daniel-no-nas')?.checked ?? false,
        metric: document.getElementById('exp-daniel-metric')?.value || 'f1',
        timeLimit: parseFloat(document.getElementById('exp-daniel-time-limit')?.value) || 0,
        timeSeries: timeSeries,
        target: targetSel.value,
        sources: sources,
        preprocessorId: preprocessorId,
        predictFile: predictFile,  // Option B:上傳的 Kaggle test.csv,可選
      };
      startDanielExperimentTraining(ds, targetSel.value, danielOptions);
      return;
    }

    if (engine === 'autogluon') {
      // Autogluon 引擎:走 /api/train/autogluon/stream
      if (typeof ApiClient === 'undefined' || !ApiClient.enabled) {
        alert('Autogluon 引擎需要 Python 後端 API,請先在系統設定開啟「使用 Python 後端 API」'); return;
      }
      if (sources.includes('preprocessed')) {
        alert('Autogluon 內建自家 preprocessing,請只勾「原始資料」。'); return;
      }
      const agOptions = {
        preset: document.getElementById('exp-autogluon-preset')?.value || 'medium_quality',
        timeLimit: parseFloat(document.getElementById('exp-autogluon-time-limit')?.value) || 0,
        target: targetSel.value,
        sources: sources,
      };
      startAutogluonExperimentTraining(ds, targetSel.value, agOptions);
      return;
    }

    // === sklearn 引擎(原本流程) ===
    const selectedFeatures = [...featBox.querySelectorAll('.exp-feat-cb:checked')].map(cb => cb.value);
    const selectedAlgos = [...algoBox.querySelectorAll('.exp-algo-cb:checked')].map(cb => cb.value);
    if (sources.includes('raw') && selectedFeatures.length === 0) {
      alert('「原始資料集」需至少選擇一個特徵欄位'); return;
    }
    if (selectedAlgos.length === 0) { alert('請至少選擇一個演算法'); return; }

    const s = getSettings();
    const options = {
      features: selectedFeatures,
      algorithms: selectedAlgos,
      taskType: taskType,
      timeSeries: timeSeries,
      sources: sources,
      preprocessorId: preprocessorId,
      testSize: s.testSize,
      randomState: s.seed,
      classWeightBalanced: document.getElementById('exp-class-weight-balanced')?.checked ?? false,
    };
    startRealTraining(ds, targetSel.value, options);
  });

  // If already trained, show results
  if (MLEngine.trainedModels.length > 0) {
    document.getElementById('exp-results').classList.remove('hidden');
  }

  // --- 還原上次的表單設定 (切頁回來不要回復預設) ---
  // 在所有預設 + listener 都 wire 完之後才套用,確保不被覆寫
  restoreExpFormState();
  if (typeof applyEngineUI === 'function') applyEngineUI();  // 還原 engine 後同步顯示/隱藏區塊

  // --- 任一控制項改動就存檔 (delegated,只綁一次) ---
  const configEl = document.getElementById('exp-real-config');
  if (configEl && !configEl.__expFormPersist) {
    configEl.__expFormPersist = true;
    configEl.addEventListener('change', saveExpFormState);
  }
}

// 訓練按鈕狀態機 — 用單一函式集中管理,避免切頁時 cloneNode 把舊參考孤立掉
let _trainingState = 'idle'; // 'idle' | 'training' | 'completed' | 'failed' | 'cancelling'
// 訓練中持有 AbortController,點「取消」呼叫 abort() 後 fetch 拋 AbortError,
// 後端 event_stream 偵測到 GeneratorExit 會 set cancel_token 並終止 subprocess
let _trainingAbort = null;

function cancelCurrentTraining() {
  if (!_trainingAbort) return;
  try { _trainingAbort.abort(); } catch (e) {}
  setTrainBtnState('cancelling');
}
window.cancelCurrentTraining = cancelCurrentTraining;

function setTrainBtnState(state) {
  _trainingState = state;
  // 任何非 training/cancelling 的最終態 → 停 elapsed timer(訓練結束/失敗/取消)
  if (state !== 'training' && state !== 'cancelling') {
    if (typeof _trainingTimerStop === 'function') _trainingTimerStop();
  }
  const btn = document.getElementById('btn-real-train');
  if (!btn) return;
  const spinner = '<div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin inline-block mr-2"></div>';
  // 點擊行為由 addEventListener 統一處理 (依 _trainingState 分派 start / cancel),
  // 這裡只負責「外觀」和 disabled flag。不再用 onclick = ...,避免跟 addEventListener 雙重觸發。
  switch (state) {
    case 'training':
      btn.innerHTML = '<svg class="w-4 h-4 inline-block mr-1.5 align-text-bottom"><use href="#i-close"/></svg>取消訓練';
      btn.disabled = false;
      btn.classList.remove('opacity-75');
      btn.classList.add('btn-danger-cancel');
      break;
    case 'cancelling':
      btn.innerHTML = spinner + '取消中...';
      btn.disabled = true;
      btn.classList.add('opacity-75');
      break;
    case 'completed':
      btn.innerHTML = '重新訓練';
      btn.disabled = false;
      btn.classList.remove('opacity-75', 'btn-danger-cancel');
      break;
    case 'failed':
      btn.innerHTML = '重試訓練';
      btn.disabled = false;
      btn.classList.remove('opacity-75', 'btn-danger-cancel');
      break;
    default: // 'idle'
      btn.innerHTML = '開始訓練';
      btn.disabled = false;
      btn.classList.remove('opacity-75', 'btn-danger-cancel');
  }
}

async function startRealTraining(ds, targetCol, options = {}) {
  setTrainBtnState('training');
  setGlobalStatus('running', `訓練中 — ${ds.fileName || 'Dataset'}`);

  const progressCard = document.getElementById('exp-training-progress');
  const resultsCard = document.getElementById('exp-results');
  progressCard.classList.remove('hidden');
  resultsCard.classList.add('hidden');
  // sklearn 沒 time_limit 概念,只顯示 elapsed
  _trainingTimerStart_();

  const logEl = document.getElementById('exp-training-log');
  logEl.innerHTML = '';

  const addLog = (msg, type) => {
    const colorMap = { info: 'text-dark-500', success: 'text-success-400', warning: 'text-warning-400', error: 'text-danger-400', best: 'text-accent-400' };
    const now = new Date();
    const ts = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}:${String(now.getSeconds()).padStart(2,'0')}`;
    const p = document.createElement('p');
    p.className = colorMap[type] || 'text-dark-400';
    p.textContent = `[${ts}] ${msg}`;
    logEl.appendChild(p);
    logEl.scrollTop = logEl.scrollHeight;
  };

  const onProgress = (ev) => {
    if (ev.type === 'log') {
      addLog(ev.msg, ev.logType);
    } else if (ev.type === 'progress') {
      document.getElementById('exp-progress-bar').style.width = ev.pct + '%';
      document.getElementById('exp-progress-step').textContent = ev.step;
    }
  };

  try {
    addLog('開始準備訓練資料...', 'info');
    if (options.timeSeries) addLog('時間序列模式：依時間順序切分訓練/測試集', 'info');
    if (options.classWeightBalanced) addLog('類別平衡模式已啟用：class_weight="balanced"', 'info');

    let models, data;
    if (typeof ApiClient !== 'undefined' && ApiClient.enabled) {
      addLog('使用 Python 後端 API 進行訓練 (SSE 即時推送)...', 'info');
      _trainingAbort = new AbortController();
      models = await ApiClient.trainStream({
        datasetId: ds.id,
        target: targetCol,
        features: options.features,
        algorithms: options.algorithms,
        options: {
          taskType: options.taskType,
          timeSeries: options.timeSeries,
          testSize: options.testSize,
          randomState: options.randomState,
          classWeightBalanced: options.classWeightBalanced ?? false,
        },
        sources: options.sources || ['raw'],
        preprocessorId: options.preprocessorId || null,
      }, (ev) => {
        if (ev.type === 'log') {
          addLog(ev.msg, ev.level || 'info');
        } else if (ev.type === 'progress') {
          onProgress({ type: 'progress', pct: ev.pct, step: ev.step });
        }
      }, _trainingAbort.signal);
      MLEngine.trainedModels = models;
      data = { taskType: models[0]?.taskType || 'regression', target: targetCol };
    } else {
      const result = await MLEngine.runExperiment(ds, targetCol, options, onProgress);
      models = result.models;
      data = result.data;
    }

    if (models.length === 0) {
      addLog('所有演算法均訓練失敗，請檢查資料或調整設定', 'error');
      setTrainBtnState('failed');
      setGlobalStatus('error', '訓練失敗');
      notify('訓練失敗', '所有演算法均訓練失敗,請檢查資料或設定', 'error');
      return;
    }

    // Show results — must un-hide BEFORE rendering charts so ECharts can compute size
    resultsCard.classList.remove('hidden');
    renderExperimentResults(models, data);

    setTrainBtnState('completed');
    setGlobalStatus('success', `訓練完成 — ${ds.fileName || 'Dataset'} (${models.length} 模型)`);

    // 推進歷史 (localStorage 持久化最近 5 筆)
    const _isRegForHist = (data.taskType === 'regression');
    const _bestM = models[0];
    pushTrainingHistory({
      timestamp: Date.now(),
      datasetId: ds.id,
      datasetName: ds.fileName || 'Dataset',
      target: targetCol,
      taskType: data.taskType,
      sources: options.sources || ['raw'],
      modelCount: models.length,
      metric: _isRegForHist ? 'R²' : 'Accuracy',
      bestModel: {
        name: _bestM.name,
        score: _isRegForHist ? _bestM.metrics.testR2 : _bestM.metrics.testAccuracy,
      },
      options: { features: options.features, algorithms: options.algorithms },
      models, // 完整模型陣列;localStorage 爆 quota 時 _persistTrainingHistory 會自動降級
    });

    // 訓練完成後同步重新渲染當前頁面 — 使用者若已在 dashboard / leaderboard / insights
    // 可以立刻看到新訓練,不用先切走再切回
    const _curPage = document.querySelector('.page-section:not(.hidden)');
    if (_curPage) {
      const _pid = _curPage.id.replace('page-', '');
      if (['dashboard', 'leaderboard', 'insights'].includes(_pid)) {
        renderPageCharts(_pid);
      }
    }

    // 訓練完成通知 — 右上角鈴鐺
    const best = models[0];
    const isReg = (data.taskType === 'regression');
    const metricKey = isReg ? 'R²' : 'Accuracy';
    // 防 undefined:hydration 後 metrics 可能缺欄,fallback '—'
    const _bestR2 = typeof best.metrics?.testR2 === 'number' ? best.metrics.testR2.toFixed(4) : '—';
    const _bestAc = typeof best.metrics?.testAccuracy === 'number' ? (best.metrics.testAccuracy * 100).toFixed(2) + '%' : '—';
    const scoreStr = isReg ? `${metricKey}=${_bestR2}` : `Acc=${_bestAc}`;
    const datasetName = ds.fileName || 'Dataset';
    notify(
      '訓練完成 ✓',
      `${datasetName} → ${models.length} 個模型,最佳: ${best.name} (${scoreStr})`,
      'success',
      {
        type: 'training',
        dataset: datasetName,
        target: targetCol,
        taskType: data.taskType,
        sources: options.sources || ['raw'],
        modelCount: models.length,
        metric: metricKey,
        topModels: models.slice(0, 5).map(m => ({
          name: m.name.replace(/^\[(原始|預處理)\]\s*/, ''),
          source: m.dataSource === 'preprocessed' ? '預處理' : '原始',
          score: isReg ? m.metrics.testR2 : m.metrics.testAccuracy,
          trainTime: m.trainTime || 0,
          hyperparameters: m.hyperparameters || null,
        })),
        totalTime: models.reduce((sum, m) => sum + (m.trainTime || 0), 0),
      }
    );

  } catch (err) {
    if (err.name === 'AbortError' || /aborted|abort/i.test(err.message || '')) {
      addLog('已取消訓練', 'warning');
      setTrainBtnState('idle');
      setGlobalStatus('idle', '已取消訓練');
      notify('訓練已取消', '已停止當前訓練', 'warning');
    } else {
      addLog(`錯誤: ${err.message}`, 'error');
      setTrainBtnState('failed');
      setGlobalStatus('error', '訓練發生錯誤');
      notify('訓練發生錯誤', err.message, 'error');
    }
  } finally {
    _trainingAbort = null;
  }
}

// ============================================================
// DANIEL PIPELINE training (in 實驗室) — 走 /api/train/pipeline/stream
// 支援多 source (raw / preprocessed),每個 source 跑一次,結果合併進 leaderboard
// ============================================================
// Autogluon engine training — 跟 daniel 同個 SSE 解析框架,只是 endpoint 跟 options 不同。
// 訓練完用 hydration 拉回 DB 上的 autogluon_model bundle 放進 leaderboard 顯示。
async function startAutogluonExperimentTraining(ds, targetCol, options) {
  setTrainBtnState('training');
  setGlobalStatus('running', `Autogluon 訓練中 — ${ds.fileName || 'Dataset'}`);

  const progressCard = document.getElementById('exp-training-progress');
  const resultsCard = document.getElementById('exp-results');
  progressCard.classList.remove('hidden');
  resultsCard.classList.add('hidden');
  _trainingTimerStart_(options.timeLimit || 0);

  const logEl = document.getElementById('exp-training-log');
  logEl.innerHTML = '';

  const addLog = (msg, level = 'info') => {
    const colorMap = { info: 'text-dark-500', muted: 'text-dark-600', success: 'text-success-400', warning: 'text-warning-400', error: 'text-danger-400' };
    const now = new Date();
    const ts = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}:${String(now.getSeconds()).padStart(2,'0')}`;
    const p = document.createElement('p');
    p.className = colorMap[level] || 'text-dark-400';
    p.textContent = `[${ts}] ${msg}`;
    logEl.appendChild(p);
    logEl.scrollTop = logEl.scrollHeight;
    if (logEl.children.length > 800) {
      while (logEl.children.length > 600) logEl.removeChild(logEl.firstChild);
    }
  };

  try {
    addLog(`啟動 Autogluon (preset=${options.preset}, time_limit=${options.timeLimit > 0 ? options.timeLimit + 's' : 'autogluon default'})...`, 'info');

    const form = new FormData();
    form.append('datasetId', ds.id);
    form.append('sources', JSON.stringify(options.sources));
    form.append('target', options.target);
    form.append('preset', options.preset);
    form.append('timeLimit', String(options.timeLimit || 0));

    const headers = {};
    if (typeof AuthClient !== 'undefined' && AuthClient.token) {
      headers['Authorization'] = `Bearer ${AuthClient.token}`;
    }

    _trainingAbort = new AbortController();
    const resp = await fetch(`${ApiClient.baseUrl}/api/train/autogluon/stream`, {
      method: 'POST', body: form, headers, signal: _trainingAbort.signal,
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status} ${resp.statusText}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buf = '';
    let finalResults = null;
    let runId = null;
    let errorMsg = null;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) !== -1) {
        const chunk = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        for (const line of chunk.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          let ev;
          try { ev = JSON.parse(line.slice(6)); } catch { continue; }
          if (ev.type === 'log') addLog(ev.msg, ev.level || 'info');
          else if (ev.type === 'progress') {
            document.getElementById('exp-progress-bar').style.width = ev.pct + '%';
            document.getElementById('exp-progress-step').textContent = ev.step;
          } else if (ev.type === 'done') {
            finalResults = ev.results || [];
            runId = ev.runId || null;
          } else if (ev.type === 'error') {
            errorMsg = ev.message;
          }
        }
      }
    }

    if (errorMsg) throw new Error(errorMsg);
    if (!finalResults || finalResults.length === 0) throw new Error('未收到 autogluon 結果');

    const okResults = finalResults.filter(r => r.ok !== false);
    if (okResults.length === 0) {
      const reasons = finalResults.map(r => `${r.dataSourceLabel || r.dataSource}: ${r.error || 'unknown'}`).join('\n');
      throw new Error(`autogluon 失敗:\n${reasons}`);
    }

    // Autogluon result → fake model bundle (跟 danielResultToModel 同型)
    const agTaskType = okResults[0]?.taskType === 'regression' ? 'regression' : 'classification';
    const models = okResults.map((r, i) => {
      const isReg = r.taskType === 'regression';
      const label = r.dataSourceLabel || (r.dataSource === 'preprocessed' ? '預處理' : '原始');
      return {
        id: `autogluon_${i}_${Date.now()}`,
        name: `[${label}] Autogluon (${r.bestModel || '?'})`,
        type: 'autogluon_model',
        taskType: r.taskType || 'classification',
        targetName: r.target,
        featureNames: [],
        metrics: isReg ? {
          taskType: 'regression',
          testR2: r.r2 ?? 0, testRMSE: r.rmse ?? 0, testMAE: r.mae ?? 0,
          testScore: r.bestScore ?? r.r2 ?? 0,
          testScoreLabel: 'R²',
        } : {
          taskType: 'classification',
          testAccuracy: r.accuracy ?? 0, f1: r.f1 ?? 0,
          testScore: r.bestScore ?? 0,
          testScoreLabel: (r.metric || 'F1').toUpperCase(),
        },
        featureImportance: [],
        testTrue: r.testTrueDecoded || [],
        testPred: r.testPredDecoded || [],
        trainTime: (r.elapsedSec || 0) * 1000,
        inferLatency: 0,
        trainSize: r.nTrain || 0,
        testSize: r.nTest || 0,
        means: [], stds: [], featureStats: [],
        dataSource: r.dataSource,
        dataSourceLabel: label,
        preprocessorId: null,
        estimatorMb: r.estimatorMb,
        // 把 autogluon 內部 leaderboard 當成 perModel 給排行榜可摺疊區塊用
        danielPerModel: r.perModel || [],
        presetUsed: r.presetUsed,
      };
    });
    const sortedModels = models.sort((a, b) => (b.metrics.testScore || 0) - (a.metrics.testScore || 0));
    MLEngine.trainedModels = sortedModels;

    resultsCard.classList.remove('hidden');
    renderExperimentResults(sortedModels, { taskType: agTaskType, target: targetCol });

    setTrainBtnState('completed');
    setGlobalStatus('success', `Autogluon 完成 — ${finalResults.length} 個 source`);

    const best = sortedModels[0];
    pushTrainingHistory({
      timestamp: Date.now(),
      datasetId: ds.id,
      datasetName: ds.fileName || 'Dataset',
      target: targetCol,
      taskType: agTaskType,
      sources: options.sources,
      modelCount: sortedModels.length,
      metric: best.metrics.testScoreLabel || (agTaskType === 'regression' ? 'R²' : 'F1'),
      bestModel: { name: best.name, score: best.metrics.testScore },
      options: { engine: 'autogluon', preset: options.preset, timeLimit: options.timeLimit },
      models: sortedModels,
      engine: 'autogluon',
    });

    // 從 DB re-hydrate 拿真正的 autogluon_model bundle (含 estimatorMb 跟 perModel)
    try {
      await hydrateUserHistoryFromDb();
      const dbRun = _trainingHistory.find(h => h.id === runId) || _trainingHistory[0];
      if (dbRun && dbRun.models && dbRun.models.length > 0) {
        _activeHistoryRunId = dbRun.id;
        MLEngine.trainedModels = dbRun.models.sort(
          (a, b) => (b.metrics?.testScore || 0) - (a.metrics?.testScore || 0)
        );
        renderExperimentResults(MLEngine.trainedModels, { taskType: agTaskType, target: targetCol });
      }
    } catch (e) {
      console.warn('[autogluon] re-hydrate 失敗:', e.message);
    }

    notify('Autogluon 完成 ✓',
      `${ds.fileName || 'Dataset'} — 最佳: ${best.name} (${best.metrics.testScoreLabel}=${best.metrics.testScore.toFixed(4)})`,
      'success');
  } catch (err) {
    if (err.name === 'AbortError' || /aborted|abort/i.test(err.message || '')) {
      addLog('已取消訓練 — 後端 subprocess 已終止', 'warning');
      setTrainBtnState('idle');
      setGlobalStatus('idle', '已取消訓練');
      notify('Autogluon 已取消', '訓練被使用者中止', 'warning');
      return;
    }
    addLog(`✗ Autogluon 失敗: ${err.message}`, 'error');
    setTrainBtnState('idle');
    setGlobalStatus('error', `Autogluon 失敗: ${err.message}`);
    notify('Autogluon 失敗', err.message || String(err), 'error');
  } finally {
    _trainingAbort = null;
  }
}


async function startDanielExperimentTraining(ds, targetCol, options) {
  setTrainBtnState('training');
  setGlobalStatus('running', `Pipeline 訓練中 — ${ds.fileName || 'Dataset'}`);

  const progressCard = document.getElementById('exp-training-progress');
  const resultsCard = document.getElementById('exp-results');
  progressCard.classList.remove('hidden');
  resultsCard.classList.add('hidden');
  _trainingTimerStart_(options.timeLimit || 0);

  const logEl = document.getElementById('exp-training-log');
  logEl.innerHTML = '';

  const addLog = (msg, level = 'info') => {
    const colorMap = { info: 'text-dark-500', muted: 'text-dark-600', success: 'text-success-400', warning: 'text-warning-400', error: 'text-danger-400' };
    const now = new Date();
    const ts = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}:${String(now.getSeconds()).padStart(2,'0')}`;
    const p = document.createElement('p');
    p.className = colorMap[level] || 'text-dark-400';
    p.textContent = `[${ts}] ${msg}`;
    logEl.appendChild(p);
    logEl.scrollTop = logEl.scrollHeight;
    // 防爆:超過 800 行就 trim
    if (logEl.children.length > 800) {
      while (logEl.children.length > 600) logEl.removeChild(logEl.firstChild);
    }
  };

  try {
    addLog(`啟動 AutoML Pipeline (來源: ${options.sources.join(' + ')})...`, 'info');
    if (options.timeSeries) addLog('時間序列模式:會跑 TCN / PatchTST', 'info');
    if (options.fast) addLog('快速模式:HPO trials 縮減', 'muted');

    // 組 form data
    const form = new FormData();
    form.append('datasetId', ds.id);
    form.append('sources', JSON.stringify(options.sources));
    if (options.preprocessorId) form.append('preprocessorId', options.preprocessorId);
    form.append('target', options.target);
    form.append('timeSeries', options.timeSeries ? 'true' : 'false');
    form.append('metric', options.metric);
    form.append('fast', options.fast ? 'true' : 'false');
    form.append('skipDl', options.skipDl ? 'true' : 'false');
    form.append('noNas', options.noNas ? 'true' : 'false');
    form.append('timeLimit', String(options.timeLimit || 0));
    // Option B:訓練時上傳的 predict.csv (可選)
    if (options.predictFile) {
      form.append('predictFile', options.predictFile);
      addLog(`已附加預測 CSV: ${options.predictFile.name} (${(options.predictFile.size / 1024).toFixed(1)} KB)`, 'info');
    }

    const headers = {};
    if (typeof AuthClient !== 'undefined' && AuthClient.token) {
      headers['Authorization'] = `Bearer ${AuthClient.token}`;
    }

    // 建立 AbortController,讓「取消訓練」按鈕能 abort fetch (後端會收到 GeneratorExit)
    _trainingAbort = new AbortController();
    const resp = await fetch(`${ApiClient.baseUrl}/api/train/pipeline/stream`, {
      method: 'POST', body: form, headers, signal: _trainingAbort.signal,
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status} ${resp.statusText}`);

    // SSE 解析
    const reader = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buf = '';
    let finalResults = null;
    let runId = null;
    let errorMsg = null;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) !== -1) {
        const chunk = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        for (const line of chunk.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          let ev;
          try { ev = JSON.parse(line.slice(6)); } catch { continue; }
          if (ev.type === 'log') {
            addLog(ev.msg, ev.level || 'info');
          } else if (ev.type === 'progress') {
            document.getElementById('exp-progress-bar').style.width = ev.pct + '%';
            document.getElementById('exp-progress-step').textContent = ev.step;
          } else if (ev.type === 'done') {
            finalResults = ev.results || [];
            runId = ev.runId || null;
          } else if (ev.type === 'error') {
            errorMsg = ev.message;
          }
        }
      }
    }

    if (errorMsg) throw new Error(errorMsg);
    if (!finalResults || finalResults.length === 0) throw new Error('未收到 pipeline 結果');

    // 過濾掉 subprocess 失敗 (ok=false) 的 result;全失敗才整批拋
    const okResults = finalResults.filter(r => r.ok !== false);
    if (okResults.length === 0) {
      const reasons = finalResults.map(r => `${r.dataSourceLabel || r.dataSource}: ${r.error || 'unknown'}`).join('\n');
      throw new Error(`所有 pipeline 任務都失敗:\n${reasons}`);
    }
    if (okResults.length < finalResults.length) {
      finalResults.filter(r => r.ok === false).forEach(r => {
        const msg = `✗ ${r.dataSourceLabel || r.dataSource} pipeline 失敗: ${r.error || '未知'}`;
        if (typeof addLog === 'function') addLog(msg, 'error');
        else console.error('[pipeline]', msg);
      });
    }

    // 把 pipeline result 包成「假 model bundle」塞進現有 leaderboard,共用 renderExperimentResults
    const models = okResults.map((r, i) => danielResultToModel(r, i));
    const sortedModels = models.sort((a, b) => (b.metrics.testScore || 0) - (a.metrics.testScore || 0));
    MLEngine.trainedModels = sortedModels;

    // 任務類型:從第一個 result 判斷 (回歸 / 分類),不再寫死分類
    const pipelineTaskType = okResults[0]?.taskType === 'regression' ? 'regression' : 'classification';
    resultsCard.classList.remove('hidden');
    renderExperimentResults(sortedModels, { taskType: pipelineTaskType, target: targetCol });

    // 在訓練結果區插一張 Daniel 專屬詳細卡片 (perModel / Blend vs Stack 切換指標)
    insertDanielDetailCard(finalResults, targetCol);

    // Option B:有上傳 predict.csv + 後端有產出 submission → 加下載按鈕
    insertPipelineSubmissionCard(finalResults, runId);

    setTrainBtnState('completed');
    setGlobalStatus('success', `Pipeline 完成 — ${finalResults.length} 個 source`);

    // 推進歷史
    const best = sortedModels[0];
    pushTrainingHistory({
      timestamp: Date.now(),
      datasetId: ds.id,
      datasetName: ds.fileName || 'Dataset',
      target: targetCol,
      taskType: pipelineTaskType,
      sources: options.sources,
      modelCount: sortedModels.length,
      metric: best.metrics.testScoreLabel || (pipelineTaskType === 'regression' ? 'R²' : 'F1'),
      bestModel: { name: best.name, score: best.metrics.testScore },
      options: { engine: 'daniel', fast: options.fast, skipDl: options.skipDl, noNas: options.noNas },
      models: sortedModels,
      engine: 'daniel',  // 標記是 Daniel 跑的,洞察頁可以判斷
    });

    // 訓練後從 DB re-hydrate,把上面塞的 placeholder bundle 換成 DB 真實的 ensemble Model
    // (backend 在訓練成功時用 storage.save_model 寫入 ensemble bundle,training_run_id 對得起來;
    //  hydrate 後 MLEngine.trainedModels 會變成可以 batch predict / 跑 SHAP 的真模型)
    try {
      await hydrateUserHistoryFromDb();
      console.log(`[pipeline] re-hydrate done. runId=${runId}, _trainingHistory ids:`,
                  _trainingHistory.map(h => h.id).slice(0, 5));
      // 用 DB 的版本覆蓋掉 placeholder,但只取對應的 run
      let dbRun = _trainingHistory.find(h => h.id === runId);
      // SSE 的 runId 是 DB UUID;pushTrainingHistory 塞的是 client 假 id (run_xxx)。
      // 若找不到,fallback 用最新一筆 (剛訓練的應該排在第一)
      if (!dbRun && _trainingHistory.length > 0) {
        console.warn(`[pipeline] 找不到 runId=${runId} 在 _trainingHistory,fallback 用第 0 筆`);
        dbRun = _trainingHistory[0];
      }
      if (dbRun && dbRun.models && dbRun.models.length > 0) {
        // 同步 _activeHistoryRunId 到 DB id,否則 cascade dropdown 切到別筆
        _activeHistoryRunId = dbRun.id;
        MLEngine.trainedModels = dbRun.models.sort(
          (a, b) => (b.metrics?.testScore || 0) - (a.metrics?.testScore || 0)
        );
        console.log(`[pipeline] DB re-hydrate: ${MLEngine.trainedModels.length} 個 model 取代 placeholder ` +
                    `(activeHistoryRunId=${_activeHistoryRunId})`);
        renderExperimentResults(MLEngine.trainedModels, { taskType: pipelineTaskType, target: targetCol });
      } else {
        console.warn(`[pipeline] dbRun 沒 models,保留 placeholder. dbRun=`, dbRun);
      }
    } catch (e) {
      console.warn('[pipeline] 訓練後 re-hydrate 失敗:', e.message);
    }

    notify(
      'Pipeline 完成 ✓',
      `${ds.fileName || 'Dataset'} — 最佳: ${best.name} (${best.metrics.testScoreLabel}=${best.metrics.testScore.toFixed(4)})`,
      'success',
      {
        type: 'training',
        dataset: ds.fileName || 'Dataset',
        target: targetCol,
        taskType: 'classification',
        sources: options.sources,
        modelCount: sortedModels.length,
        metric: best.metrics.testScoreLabel,
        topModels: sortedModels.map(m => ({
          name: m.name.replace(/^\[(原始|預處理)\]\s*/, ''),
          source: m.dataSource === 'preprocessed' ? '預處理' : '原始',
          score: m.metrics.testScore,
          trainTime: m.trainTime || 0,
        })),
        totalTime: sortedModels.reduce((sum, m) => sum + (m.trainTime || 0), 0),
      },
    );

  } catch (err) {
    // 使用者按取消 → 不算錯誤,只是取消
    if (err.name === 'AbortError' || /aborted|abort/i.test(err.message || '')) {
      addLog('已取消訓練 — 後端 subprocess 已終止', 'warning');
      setTrainBtnState('idle');
      setGlobalStatus('idle', '已取消訓練');
      notify('Pipeline 已取消', '訓練被使用者中止', 'warning');
    } else {
      addLog(`錯誤: ${err.message}`, 'error');
      setTrainBtnState('failed');
      setGlobalStatus('error', 'Pipeline 失敗');
      notify('Pipeline 失敗', err.message, 'error');
    }
  } finally {
    _trainingAbort = null;
  }
}

// 把 Pipeline 的 result 物件轉成 sklearn-bundle 的形狀,
// 讓現有 renderExperimentResults / leaderboard 直接吃。
function danielResultToModel(r, rank) {
  const sourceLabel = r.dataSourceLabel || (r.dataSource === 'preprocessed' ? '預處理' : '原始');
  const ensembleName = r.bestEnsemble === 'stack' ? 'Stack' : 'Blend';
  const isOof = r.scoreSource === 'oof';
  const oofSuffix = isOof ? ' · OOF' : '';
  const isReg = r.taskType === 'regression';
  const fakeName = `[${sourceLabel}] Pipeline (${ensembleName})${oofSuffix}`;

  if (isReg) {
    // 時序回歸:後端回 rmse / r2,沒有 accuracy/f1。testScore 用 R² (越大越好)
    return {
      id: r.bestModelId || `daniel_${rank}_${Date.now()}`,
      name: fakeName,
      type: 'daniel_pipeline',
      taskType: 'regression',
      targetName: r.target,
      featureNames: [],
      metrics: {
        taskType: 'regression',
        // 欄位名要跟 renderExperimentResults 的 isReg 分支一致 (testR2 / testRMSE / testMAE)
        testR2: r.r2 ?? 0,
        testRMSE: r.rmse ?? 0,
        testMAE: r.mae ?? 0,
        testScore: r.bestScore ?? r.r2 ?? 0,   // R²
        testScoreLabel: 'R²' + (isOof ? ' (OOF)' : ''),
        bestEnsemble: r.bestEnsemble,
        scoreSource: r.scoreSource || 'test',
      },
      featureImportance: [],
      testTrue: [],
      testPred: [],
      trainTime: (r.elapsedSec || 0) * 1000,
      inferLatency: 0,
      trainSize: r.nTrain || 0,
      testSize: r.nTest || 0,
      means: [], stds: [], featureStats: [],
      dataSource: r.dataSource,
      dataSourceLabel: sourceLabel,
      preprocessorId: r.preprocessorId,
      danielPerModel: r.perModel || [],
      danielSplitMode: r.splitMode,
      danielIsTs: r.isTimeSeries,
      danielDevice: r.device,
      danielNFeatures: r.nFeatures,
    };
  }

  // testScore 用 bestScore (pipeline 自己挑 Blend vs Stack 較佳者;沒 label 時 fallback OOF max)
  const scoreLabel = (r.metric || 'F1').toUpperCase() + (isOof ? ' (OOF)' : '');
  return {
    id: r.bestModelId || `daniel_${rank}_${Date.now()}`,
    name: fakeName,
    type: 'daniel_pipeline',
    taskType: 'classification',
    targetName: r.target,
    featureNames: [],  // Pipeline 不揭露最終特徵集 (內部做了 PCA/FFT/KMeans...)
    metrics: {
      taskType: 'classification',
      testAccuracy: r.accuracy ?? 0,
      f1: r.f1 ?? 0,
      precision: r.f1 ?? 0,    // Pipeline 不單獨回報 precision/recall,用 f1 當 placeholder
      recall: r.f1 ?? 0,
      testScore: r.bestScore ?? 0,
      testScoreLabel: scoreLabel,
      scoreBlend: r.scoreBlend,
      scoreStack: r.scoreStack,
      bestEnsemble: r.bestEnsemble,
      scoreSource: r.scoreSource || 'test',
      oofBestScore: r.oofBestScore,
      classes: r.classes || [],
    },
    // 新版後端會回 featureImportance / testTrueDecoded / testPredDecoded,沒有就 fallback 空陣列
    // 讓 Insights 散布圖 / 殘差圖 / 特徵重要性 同訓練完馬上看得到
    featureImportance: r.featureImportance || [],
    testTrue: r.testTrueDecoded || [],
    testPred: r.testPredDecoded || [],
    trainTime: (r.elapsedSec || 0) * 1000,
    inferLatency: 0,
    trainSize: r.nTrain || 0,
    testSize: r.nTest || 0,
    means: [],
    stds: [],
    featureStats: [],
    dataSource: r.dataSource,
    dataSourceLabel: sourceLabel,
    preprocessorId: r.preprocessorId,
    // 模型大小 (MB) — > 50MB 會落到 model_blobs/ 檔案,排行榜列會標 File
    estimatorMb: r.estimatorMb,
    // Daniel-specific
    danielPerModel: r.perModel || [],
    danielSplitMode: r.splitMode,
    danielIsTs: r.isTimeSeries,
    danielDevice: r.device,
    danielNFeatures: r.nFeatures,
    danielNClasses: r.nClasses,
  };
}

// 在「訓練結果」卡片下方插一張 Daniel 專屬詳細卡 (per-source perModel breakdown + Blend/Stack)
function insertDanielDetailCard(results, targetCol) {
  const old = document.getElementById('exp-daniel-detail');
  if (old) old.remove();

  const wrap = document.createElement('div');
  wrap.id = 'exp-daniel-detail';
  wrap.className = 'card mb-6';
  wrap.innerHTML = `
    <div class="card-header">
      <h3 class="card-title">Pipeline 詳細</h3>
      <span class="text-xs text-dark-400">每個 source 列出 Blend / Stack / 各模型 OOF 分數</span>
    </div>
    <div class="grid grid-cols-1 ${results.length > 1 ? 'lg:grid-cols-2' : ''} gap-4 p-4">
      ${results.map(r => {
        const label = r.dataSourceLabel || (r.dataSource === 'preprocessed' ? '預處理' : '原始');
        const accent = r.dataSource === 'preprocessed' ? 'accent' : 'primary';
        const bestEns = r.bestEnsemble === 'stack' ? 'Stack' : 'Blend';
        const perModel = (r.perModel || []).filter(m => m.oofScore != null)
          .sort((a, b) => (b.oofScore || 0) - (a.oofScore || 0));
        return `
          <div class="bg-dark-900/40 rounded-lg border border-${accent}-500/20 p-4">
            <div class="flex items-center justify-between mb-3">
              <p class="text-sm font-semibold text-${accent}-300">${escapeHtml(label)} 資料來源</p>
              <span class="text-[10px] px-1.5 py-0.5 rounded bg-${accent}-500/15 text-${accent}-400">最佳: ${bestEns}</span>
            </div>
            <div class="grid grid-cols-3 gap-2 text-center mb-3">
              <div>
                <p class="text-[10px] text-dark-500">Blend</p>
                <p class="text-sm font-mono ${r.bestEnsemble === 'blend' ? 'text-warning-300' : 'text-dark-300'}">${(r.scoreBlend ?? 0).toFixed(4)}</p>
              </div>
              <div>
                <p class="text-[10px] text-dark-500">Stack</p>
                <p class="text-sm font-mono ${r.bestEnsemble === 'stack' ? 'text-warning-300' : 'text-dark-300'}">${(r.scoreStack ?? 0).toFixed(4)}</p>
              </div>
              <div>
                <p class="text-[10px] text-dark-500">耗時</p>
                <p class="text-sm font-mono text-dark-200">${(r.elapsedSec ?? 0).toFixed(0)}s</p>
              </div>
            </div>
            <p class="text-[10px] text-dark-500 mb-2 pb-2 border-b border-dark-700/50">
              ${r.nTrain}/${r.nTest} · ${r.nClasses} class · ${r.nFeatures} features · ${escapeHtml(r.splitMode || '')} · ${escapeHtml(r.device || '')}
            </p>
            ${perModel.length ? `
              <details>
                <summary class="text-[11px] text-dark-400 cursor-pointer hover:text-dark-200">各模型 OOF (${perModel.length})</summary>
                <div class="mt-2 space-y-1 max-h-48 overflow-y-auto">
                  ${perModel.map(m => `
                    <div class="flex items-center justify-between text-[11px]">
                      <span class="font-mono text-dark-300 truncate" title="${escapeHtml(m.tag)}">${escapeHtml(m.tag)}</span>
                      <span class="font-mono text-${accent}-300 ml-2 shrink-0">${(m.oofScore ?? 0).toFixed(4)}</span>
                    </div>
                  `).join('')}
                </div>
              </details>
            ` : ''}
          </div>
        `;
      }).join('')}
    </div>
  `;
  // 插在訓練結果卡片下方
  const resultsCard = document.getElementById('exp-results');
  resultsCard.appendChild(wrap);
}


// Option B 專用 — 訓練時有上傳 predict.csv,訓練完顯示「下載 submission.csv」按鈕
function insertPipelineSubmissionCard(results, runId) {
  // 移除舊的 (重訓會重新插入)
  document.getElementById('exp-pipeline-submission')?.remove();

  const withSubmission = (results || []).filter(r => r.submissionAvailable && r.submissionKind);
  if (withSubmission.length === 0 || !runId) return;
  // 不確定 ApiClient 是否啟用
  const apiBase = (typeof ApiClient !== 'undefined' && ApiClient.baseUrl) ? ApiClient.baseUrl : '';
  if (!apiBase) return;

  const wrap = document.createElement('div');
  wrap.id = 'exp-pipeline-submission';
  wrap.className = 'card mb-6 border border-warning-500/30';
  wrap.innerHTML = `
    <div class="card-header">
      <h3 class="card-title flex items-center gap-2">
        <svg class="w-4 h-4 text-warning-400"><use href="#i-download"/></svg>
        預測結果下載 (submission.csv)
      </h3>
      <span class="text-xs text-dark-400">${withSubmission.length} 個 source 已產出</span>
    </div>
    <div class="p-4 space-y-2">
      <p class="text-[11px] text-dark-400 leading-relaxed">
        每筆預測 CSV 第一欄當 ID,接上預測標籤 → 可直接交 Kaggle / 其他評分系統。
      </p>
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
        ${withSubmission.map(r => {
          const label = r.dataSourceLabel || (r.dataSource === 'preprocessed' ? '預處理' : (r.dataSource === 'raw' ? '原始' : '上傳'));
          const accent = r.dataSource === 'preprocessed' ? 'accent' : 'primary';
          return `
            <a href="${apiBase}/api/train/pipeline/runs/${encodeURIComponent(runId)}/submission?kind=${encodeURIComponent(r.submissionKind)}"
               class="flex items-center justify-between p-3 rounded-lg bg-dark-800/60 hover:bg-dark-800 border border-${accent}-500/20 hover:border-${accent}-500/50 transition-colors group"
               download="submission_${encodeURIComponent(r.dataSource)}.csv">
              <div class="flex-1 min-w-0">
                <p class="text-sm font-medium text-${accent}-300">${escapeHtml(label)} 來源</p>
                <p class="text-[10px] text-dark-500 mt-0.5">submission_${escapeHtml(r.dataSource)}.csv</p>
              </div>
              <svg class="w-4 h-4 text-dark-400 group-hover:text-${accent}-300 ml-2 shrink-0"><use href="#i-download"/></svg>
            </a>
          `;
        }).join('')}
      </div>
    </div>
  `;
  const resultsCard = document.getElementById('exp-results');
  resultsCard.appendChild(wrap);
}


function renderExperimentResults(models, data) {
  const isReg = data.taskType === 'regression';

  // Daniel pipeline 跑出來的 model 沒有 per-feature importance / per-sample 預測 / 後端 estimator,
  // 顯示這些區塊只會看到「不支援」訊息或空下拉,直接隱藏更乾淨。
  const isPipelineRun = models.length > 0 && models.every(m => m.type === 'daniel_pipeline');
  // daniel ensemble (hydrated 後 type='daniel_pipeline_ensemble') 或 autogluon_model
  // 同樣的:訓練結果表的 3 個指標欄+訓練時間 跟「Pipeline 詳細」「排行榜」重複,隱藏整張表
  const isEnsembleish = models.length > 0 && models.every(m =>
    m.type === 'daniel_pipeline_ensemble' || m.type === 'autogluon_model'
  );
  const togglePipelineSections = (hidden) => {
    ['exp-chart-model-selector-wrap', 'exp-charts-row', 'exp-batch-predict-card'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.classList.toggle('hidden', hidden);
    });
  };
  togglePipelineSections(isPipelineRun);
  // 訓練結果表本身:ensemble / autogluon 隱藏(資訊在 Pipeline 詳細 + 排行榜)
  const resultsTableCard = document.getElementById('exp-results-table-card');
  if (resultsTableCard) resultsTableCard.classList.toggle('hidden', isPipelineRun || isEnsembleish);

  // Table headers
  if (isReg) {
    document.getElementById('exp-metric-col1').textContent = 'R² (測試)';
    document.getElementById('exp-metric-col2').textContent = 'RMSE';
    document.getElementById('exp-metric-col3').textContent = 'MAE';
  } else {
    document.getElementById('exp-metric-col1').textContent = 'Accuracy';
    document.getElementById('exp-metric-col2').textContent = 'F1-Score';
    document.getElementById('exp-metric-col3').textContent = 'AUC-ROC';
  }

  // Table body
  const tbody = document.getElementById('exp-results-body');
  tbody.innerHTML = '';
  models.forEach((m, i) => {
    const tr = document.createElement('tr');
    tr.className = 'border-b border-dark-700/30 hover:bg-dark-800/30 transition-colors';
    const rankBadge = i === 0
      ? '<span class="inline-flex items-center justify-center w-6 h-6 rounded-full bg-gradient-to-br from-warning-400 to-warning-600 text-dark-950 text-xs font-bold">1</span>'
      : `<span class="text-dark-400">${i + 1}</span>`;

    let col1, col2, col3;
    const _f = (v, d = 4) => (typeof v === 'number' ? v.toFixed(d) : '—');
    if (isReg) {
      col1 = _f(m.metrics.testR2);
      col2 = _f(m.metrics.testRMSE);
      col3 = _f(m.metrics.testMAE);
    } else {
      col1 = typeof m.metrics.testAccuracy === 'number' ? (m.metrics.testAccuracy * 100).toFixed(2) + '%' : '—';
      col2 = _f(m.metrics.f1);
      col3 = _f(m.metrics.auc);
    }

    const score = m.metrics.testScore;
    const barWidth = Math.max(5, Math.round(score * 100));
    const barColor = score > 0.8 ? 'bg-success-500' : score > 0.5 ? 'bg-warning-500' : 'bg-danger-500';

    // 資料來源 badge + 去掉名稱裡的 [原始]/[預處理] 文字前綴 (改用 badge 呈現)
    const cleanName = m.name.replace(/^\[(原始|預處理)\]\s*/, '');
    let srcBadge;
    if (m.dataSource === 'preprocessed') {
      srcBadge = '<span class="inline-block px-2 py-0.5 rounded-md text-[11px] font-semibold bg-accent-500/15 text-accent-400 border border-accent-500/30">預處理</span>';
    } else if (m.dataSource === 'raw') {
      srcBadge = '<span class="inline-block px-2 py-0.5 rounded-md text-[11px] font-semibold bg-primary-500/15 text-primary-400 border border-primary-500/30">原始</span>';
    } else {
      srcBadge = '<span class="text-dark-500 text-xs">—</span>';
    }
    // 預處理來源整列加一條左邊框,視覺上更好掃
    if (m.dataSource === 'preprocessed') tr.classList.add('border-l-2', 'border-l-accent-500/40');
    else if (m.dataSource === 'raw') tr.classList.add('border-l-2', 'border-l-primary-500/40');

    tr.innerHTML = `
      <td class="py-3 px-4">${rankBadge}</td>
      <td class="py-3 px-4">${srcBadge}</td>
      <td class="py-3 px-4">
        <p class="font-medium text-sm">${escapeHtml(cleanName)}</p>
        <div class="w-24 bg-dark-800 rounded-full h-1.5 mt-1"><div class="${barColor} h-1.5 rounded-full" style="width:${barWidth}%"></div></div>
      </td>
      <td class="py-3 px-4 font-mono text-sm ${i === 0 ? 'text-accent-400 font-semibold' : ''}">${col1}</td>
      <td class="py-3 px-4 font-mono text-sm">${col2}</td>
      <td class="py-3 px-4 font-mono text-sm">${col3}</td>
      <td class="py-3 px-4 text-xs text-dark-400">${m.trainTime.toFixed(0)}ms</td>
    `;
    tbody.appendChild(tr);
  });

  // 模型選擇器 — 可切換要看哪個模型的特徵重要性 / 預測散佈圖
  const chartSel = document.getElementById('exp-chart-model-select');
  if (chartSel && models.length > 0) {
    chartSel.innerHTML = '';
    models.forEach((m, idx) => {
      const opt = document.createElement('option');
      opt.value = String(idx);
      const srcTag = m.dataSource === 'preprocessed' ? '🟦 預處理 · '
                   : m.dataSource === 'raw' ? '🔵 原始 · ' : '';
      const cleanName = m.name.replace(/^\[(原始|預處理)\]\s*/, '');
      // 防 undefined:hydration 後的 daniel ensemble bundle 萬一沒 testR2/testAccuracy 不要炸
      const _r2 = typeof m.metrics?.testR2 === 'number' ? m.metrics.testR2.toFixed(3) : '—';
      const _ac = typeof m.metrics?.testAccuracy === 'number' ? (m.metrics.testAccuracy * 100).toFixed(1) + '%' : '—';
      const sc = isReg ? `R²=${_r2}` : `Acc=${_ac}`;
      opt.textContent = `${idx === 0 ? '⭐ ' : ''}${srcTag}${cleanName} (${sc})`;
      chartSel.appendChild(opt);
    });

    const renderChartsFor = (idx) => {
      const m = models[idx];
      if (!m) return;
      const srcLabel = m.dataSource === 'preprocessed' ? '預處理資料'
                     : m.dataSource === 'raw' ? '原始資料' : '';
      const cleanName = m.name.replace(/^\[(原始|預處理)\]\s*/, '');
      const sub = `— ${cleanName}${srcLabel ? ` · ${srcLabel}` : ''}`;
      document.getElementById('exp-chart-importance-sub').textContent = sub;
      document.getElementById('exp-chart-scatter-sub').textContent = sub;

      // 對 daniel ensemble:沒實際資料就隱藏整張 card,不要顯示「Pipeline 不揭露 / placeholder」
      // 提示文字。判斷:
      //   - featureImportance 空 → 隱藏特徵重要性 card
      //   - testTrue 空 → 隱藏 預測 vs 實際 card
      // 一般 sklearn 模型 (有資料的) → 兩張都正常顯示
      const hasFI = Array.isArray(m.featureImportance) && m.featureImportance.length > 0;
      const hasScatter = Array.isArray(m.testTrue) && m.testTrue.length > 0;
      document.getElementById('exp-importance-card')?.classList.toggle('hidden', !hasFI);
      document.getElementById('exp-scatter-card')?.classList.toggle('hidden', !hasScatter);
      // 兩個都隱藏 → 整 row 也藏,把版面空間還給下方
      document.getElementById('exp-charts-row')?.classList.toggle('hidden', !hasFI && !hasScatter);

      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          if (hasFI) renderExpFeatureImportance(m);
          if (hasScatter) renderExpPredictionChart(m, data);
        });
      });
    };

    chartSel.onchange = () => renderChartsFor(parseInt(chartSel.value));
    chartSel.value = '0';
    renderChartsFor(0);
  }

  // --- 批次預測區 ---
  initBatchPredict(models);
}

// 批次預測:選模型 + 上傳 CSV → 後端整批預測 → 下載含 prediction 欄的 CSV
function initBatchPredict(models) {
  const modelSel = document.getElementById('batch-model-select');
  const fileInput = document.getElementById('batch-csv-input');
  const sampleInput = document.getElementById('batch-sample-input');
  const btn = document.getElementById('btn-batch-predict');
  const statusEl = document.getElementById('batch-predict-status');
  if (!modelSel || !btn) return;

  // 只列有 id 且後端有對應 estimator 的模型。
  // - sklearn 模型:全部都可預測
  // - Daniel ensemble (canPredict=true):後端有完整 ensemble bundle,可走批次預測
  // - Daniel placeholder (沒有 canPredict):純歷史紀錄,沒實 estimator,排除
  const usable = models.filter(m => {
    if (!m.id) return false;
    if (m.type === 'daniel_pipeline') return false;   // 舊 placeholder (DB 只有 summary)
    return true;
  });
  modelSel.innerHTML = '';
  if (usable.length === 0) {
    modelSel.innerHTML = '<option value="">無可用模型 (需後端 API 模式訓練)</option>';
    btn.disabled = true;
    return;
  }
  btn.disabled = false;
  usable.forEach((m, i) => {
    const opt = document.createElement('option');
    opt.value = m.id;
    const srcTag = m.dataSource === 'preprocessed' ? '🟦預處理 ' : m.dataSource === 'raw' ? '🔵原始 ' : '';
    opt.textContent = `${i === 0 ? '⭐ ' : ''}${srcTag}${m.name.replace(/^\[(原始|預處理)\]\s*/, '')}`;
    modelSel.appendChild(opt);
  });

  btn.onclick = async () => {
    const modelId = modelSel.value;
    const file = fileInput.files[0];
    const sampleFile = sampleInput ? sampleInput.files[0] : null;
    const errCls = 'px-4 pb-4 pt-1 text-xs text-danger-400';
    const okCls = 'px-4 pb-4 pt-1 text-xs text-success-400';
    const infoCls = 'px-4 pb-4 pt-1 text-xs text-dark-400';
    if (!modelId) { statusEl.textContent = '✗ 請選擇模型'; statusEl.className = errCls; return; }
    if (!file) { statusEl.textContent = '✗ 請選擇測試 CSV 檔案'; statusEl.className = errCls; return; }
    if (typeof ApiClient === 'undefined' || !ApiClient.enabled) {
      statusEl.textContent = '✗ 批次預測需要開啟「使用 Python 後端 API」';
      statusEl.className = errCls;
      return;
    }

    btn.disabled = true;
    const origLabel = btn.textContent;
    btn.innerHTML = '<div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin inline-block mr-2"></div>預測中...';
    statusEl.textContent = '處理中...';
    statusEl.className = infoCls;
    setGlobalStatus('running', `批次預測中 — ${file.name}`);
    try {
      const blob = await ApiClient.predictBatch(modelId, file, sampleFile);
      // 有給範本 → submission.csv;沒給 → {原檔名}_predicted.csv
      const outName = sampleFile ? 'submission.csv' : `${file.name.replace(/\.[^.]+$/, '')}_predicted.csv`;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = outName;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      const note = sampleFile ? '(已套用範本 submission 格式)' : '(新增 prediction 欄)';
      statusEl.textContent = `✓ 預測完成,已下載 ${outName} ${note}`;
      statusEl.className = okCls;
      setGlobalStatus('success', `預測完成 — ${outName}`);
      notify('批次預測完成 ✓', `${file.name} → ${outName}`, 'success');
    } catch (e) {
      statusEl.textContent = `✗ ${e.message}`;
      statusEl.className = errCls;
      setGlobalStatus('error', '預測失敗');
      notify('批次預測失敗', e.message, 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = origLabel;
    }
  };
}

function renderExpFeatureImportance(model) {
  const chart = initChart('chart-exp-importance');
  if (!chart) return;

  // Daniel pipeline 不提供 per-feature importance — 顯示提示文字
  if (model.type === 'daniel_pipeline') {
    chart.clear();
    chart.setOption({
      title: { text: 'Pipeline 不揭露單一特徵重要性', subtext: '內部做了 PCA / FFT / KMeans 等變換', left: 'center', top: '40%', textStyle: { color: '#94a3b8', fontSize: 12 }, subtextStyle: { color: '#64748b', fontSize: 10 } },
    });
    return;
  }

  const names = model.featureNames || [];
  const values = model.featureImportance || [];
  // Sort descending
  const pairs = names.map((n, i) => ({ name: n, value: values[i] || 0 })).sort((a, b) => b.value - a.value).slice(0, 15);

  chart.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0', fontSize: 11 } },
    grid: { left: 110, right: 30, top: 10, bottom: 15 },
    xAxis: { type: 'value', axisLabel: { color: '#64748b', fontSize: 9 }, splitLine: { lineStyle: { color: '#1e293b' } } },
    yAxis: { type: 'category', data: pairs.map(p => p.name).reverse(), axisLabel: { color: '#e2e8f0', fontSize: 10, formatter: v => v.length > 15 ? v.slice(0, 13) + '...' : v }, axisLine: { show: false }, axisTick: { show: false } },
    series: [{
      type: 'bar', data: pairs.map(p => p.value).reverse(),
      itemStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
          { offset: 0, color: '#3b82f6' },
          { offset: 1, color: '#22d3ee' }
        ]),
        borderRadius: [0, 4, 4, 0],
      },
      barWidth: '55%',
      label: { show: true, position: 'right', color: '#94a3b8', fontSize: 9, formatter: p => (p.value * 100).toFixed(1) + '%' },
    }],
  });
  chart.resize();
}

function renderExpPredictionChart(model, data) {
  const chart = initChart('chart-exp-scatter');
  if (!chart) return;

  // Daniel pipeline 現在(回歸 + 分類)都會回 testTrueDecoded / testPredDecoded 給前端;
  // 只剩 testTrue 真的空才退化到「不畫」。舊版 daniel_pipeline placeholder (從 resultsSummary
  // 假造的 fake model row,沒實際 perSample 資料) 仍然顯示文字提示。
  const _isLegacyDanielPlaceholder = model.type === 'daniel_pipeline'
                                     && (!model.testTrue || model.testTrue.length === 0);
  if (_isLegacyDanielPlaceholder) {
    chart.clear();
    chart.setOption({
      title: {
        text: 'Pipeline placeholder',
        subtext: '此記錄沒存 per-sample 預測 (舊資料 / 訓練中斷)。重訓即可看圖。',
        left: 'center', top: '40%',
        textStyle: { color: '#94a3b8', fontSize: 12 },
        subtextStyle: { color: '#64748b', fontSize: 10 },
      },
    });
    return;
  }
  if (!model.testTrue || model.testTrue.length === 0) {
    chart.clear();
    return;
  }

  if (data.taskType === 'regression') {
    // Scatter: predicted vs actual
    const points = model.testTrue.map((actual, i) => [actual, model.testPred[i]]);
    const min = Math.min(...model.testTrue, ...model.testPred);
    const max = Math.max(...model.testTrue, ...model.testPred);

    chart.setOption({
      tooltip: { trigger: 'item', backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0', fontSize: 11 },
        formatter: p => `實際: ${p.value[0].toFixed(2)}<br/>預測: ${p.value[1].toFixed(2)}` },
      grid: { left: 55, right: 20, top: 20, bottom: 40 },
      xAxis: { type: 'value', name: '實際值', nameTextStyle: { color: '#64748b', fontSize: 10 }, axisLabel: { color: '#64748b', fontSize: 9 }, splitLine: { lineStyle: { color: '#1e293b' } }, min, max },
      yAxis: { type: 'value', name: '預測值', nameTextStyle: { color: '#64748b', fontSize: 10 }, axisLabel: { color: '#64748b', fontSize: 9 }, splitLine: { lineStyle: { color: '#1e293b' } }, min, max },
      series: [
        { type: 'scatter', data: points, symbolSize: 6, itemStyle: { color: '#3b82f6', opacity: 0.6 } },
        { type: 'line', data: [[min, min], [max, max]], lineStyle: { color: '#ef4444', type: 'dashed', width: 1.5 }, symbol: 'none', tooltip: { show: false } },
      ],
    });
  } else {
    // Classification: bar chart of per-class accuracy
    const classes = model.metrics.classes;
    const classCorrect = {};
    const classTotal = {};
    classes.forEach(c => { classCorrect[c] = 0; classTotal[c] = 0; });
    model.testTrue.forEach((t, i) => {
      classTotal[t] = (classTotal[t] || 0) + 1;
      if (model.testPred[i] === t) classCorrect[t] = (classCorrect[t] || 0) + 1;
    });
    const classAcc = classes.map(c => classTotal[c] > 0 ? classCorrect[c] / classTotal[c] : 0);

    chart.setOption({
      tooltip: { trigger: 'axis', backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0', fontSize: 11 } },
      grid: { left: 50, right: 20, top: 15, bottom: 30 },
      xAxis: { type: 'category', data: classes.map(c => `Class ${c}`), axisLabel: { color: '#64748b', fontSize: 10 } },
      yAxis: { type: 'value', max: 1, axisLabel: { color: '#64748b', fontSize: 9, formatter: v => (v * 100) + '%' }, splitLine: { lineStyle: { color: '#1e293b' } } },
      series: [{
        type: 'bar', data: classAcc,
        itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#22d3ee' }, { offset: 1, color: '#3b82f6' }]), borderRadius: [4, 4, 0, 0] },
        barWidth: '40%',
        label: { show: true, position: 'top', color: '#e2e8f0', fontSize: 10, formatter: p => (p.value * 100).toFixed(1) + '%' },
      }],
    });
  }
  chart.resize();
}

// 把 model.hyperparameters dict 轉成一排 chip HTML。
// 回傳 { count, html };沒有可顯示的超參數時回傳 ''(falsy,呼叫端可直接 if 判斷)。
function _hyperparamChipsHtml(hp, maxItems = 16) {
  if (!hp || typeof hp !== 'object') return '';
  const entries = Object.entries(hp)
    .filter(([_, v]) => v !== null && v !== undefined && !(Array.isArray(v) && v.length === 0))
    .sort((a, b) => a[0].localeCompare(b[0]));
  if (!entries.length) return '';
  const shown = entries.slice(0, maxItems);
  const chips = shown.map(([k, v]) => {
    const valStr = typeof v === 'number'
      ? (Number.isInteger(v) ? String(v) : v.toFixed(4))
      : (typeof v === 'object' ? JSON.stringify(v) : String(v));
    const valTrunc = valStr.length > 28 ? valStr.slice(0, 26) + '…' : valStr;
    return `<span class="inline-block text-[10px] px-1.5 py-0.5 rounded bg-dark-800 text-dark-300 mr-1 mb-1" title="${escapeHtml(k)} = ${escapeHtml(valStr)}"><span class="text-dark-500">${escapeHtml(k)}</span>=<span class="font-mono text-dark-200">${escapeHtml(valTrunc)}</span></span>`;
  }).join('');
  const more = entries.length > maxItems
    ? `<span class="text-[10px] text-dark-500">+${entries.length - maxItems} 個…</span>`
    : '';
  return { count: entries.length, html: chips + more };
}

// ===== REAL LEADERBOARD =====
function renderRealLeaderboard() {
  let models = MLEngine.trainedModels;
  if (models.length === 0) return;

  _lbWirePredictInputs();   // 確保上方 file input 的 change listener 綁好 (idempotent)

  // Apply filters
  const filterSource = document.getElementById('lb-filter-source')?.value || '';
  const filterType = document.getElementById('lb-filter-type')?.value || '';
  if (filterSource) models = models.filter(m => m.dataSource === filterSource);
  if (filterType) models = models.filter(m => m.taskType === filterType);

  // Wire filter listeners once
  const srcSel = document.getElementById('lb-filter-source');
  const typeSel = document.getElementById('lb-filter-type');
  const resetBtn = document.getElementById('lb-filter-reset');
  if (srcSel && !srcSel.__wired) {
    srcSel.__wired = true;
    srcSel.addEventListener('change', renderRealLeaderboard);
    typeSel?.addEventListener('change', renderRealLeaderboard);
    resetBtn?.addEventListener('click', () => {
      if (srcSel) srcSel.value = '';
      if (typeSel) typeSel.value = '';
      renderRealLeaderboard();
    });
  }

  const isReg = models[0]?.taskType === 'regression' || (filterType === 'regression');

  // 歷史訓練 — 三層級聯 (資料集 → target → 訓練紀錄)
  const lbCascade = document.getElementById('lb-history-cascade');
  if (lbCascade) {
    lbCascade.dataset.variant = 'compact'; // 排行榜空間有限,用小尺寸
    renderHistoryCascade(lbCascade);
  }

  // Update column headers
  const col1 = document.getElementById('lb-col1');
  const col2 = document.getElementById('lb-col2');
  const col3 = document.getElementById('lb-col3');
  if (col1 && col2 && col3) {
    if (isReg) {
      col1.innerHTML = 'R² <span class="sort-arrow">&#8597;</span>';
      col2.innerHTML = 'RMSE <span class="sort-arrow">&#8597;</span>';
      col3.innerHTML = 'MAE <span class="sort-arrow">&#8597;</span>';
    } else {
      col1.innerHTML = 'F1-Score <span class="sort-arrow">&#8597;</span>';
      col2.innerHTML = 'AUC-ROC <span class="sort-arrow">&#8597;</span>';
      col3.innerHTML = '準確率 <span class="sort-arrow">&#8597;</span>';
    }
  }

  const tbody = document.getElementById('leaderboard-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  if (models.length === 0) {
    const colspan = 11;
    tbody.innerHTML = `<tr><td colspan="${colspan}" class="py-8 text-center text-sm text-dark-500">篩選條件下沒有符合的模型</td></tr>`;
    return;
  }

  // 找出訓練最快的模型 (給「最快」標籤用) — 只看有成功訓練的
  const trained = models.filter(m => m.trainTime > 0);
  const fastestTrainTime = trained.length ? Math.min(...trained.map(m => m.trainTime)) : null;

  models.forEach((m, i) => {
    const tr = document.createElement('tr');
    tr.className = 'border-b border-dark-700/30';

    const rankEl = i === 0
      ? '<span class="inline-flex items-center justify-center w-7 h-7 rounded-full bg-gradient-to-br from-warning-400 to-warning-600 text-dark-950 text-xs font-bold">1</span>'
      : i === 1
        ? '<span class="inline-flex items-center justify-center w-7 h-7 rounded-full bg-dark-600 text-dark-200 text-xs font-bold">2</span>'
        : `<span class="text-dark-400 text-sm">${i + 1}</span>`;

    // metric 全部走「沒值就 —」防禦 — pipeline ensemble 在 test.csv 沒 target 時這些可能是 null
    const _fmt = (v, digits = 4, suffix = '') =>
      (typeof v === 'number' && isFinite(v))
        ? `${v.toFixed(digits)}${suffix}`
        : '<span class="text-dark-500">—</span>';
    const mm = m.metrics || {};
    let extraCols;
    if (isReg) {
      extraCols = `
        <td class="py-3 px-4 font-mono text-xs">${_fmt(mm.testR2, 4)}</td>
        <td class="py-3 px-4 font-mono text-xs">${_fmt(mm.testRMSE, 4)}</td>
        <td class="py-3 px-4 font-mono text-xs">${_fmt(mm.testMAE, 4)}</td>`;
    } else {
      extraCols = `
        <td class="py-3 px-4 font-mono text-xs">${_fmt(mm.f1, 4)}</td>
        <td class="py-3 px-4 font-mono text-xs ${mm.auc > 0 ? '' : 'text-dark-500'}">${mm.auc > 0 ? mm.auc.toFixed(4) : '—'}</td>
        <td class="py-3 px-4 font-mono text-xs">${_fmt(mm.testAccuracy * 100, 2, '%')}</td>`;
    }

    // 標籤:最佳 (rank 1) + 最快 (訓練時間最短)
    let tags = '';
    if (i === 0) tags += '<span class="badge badge-success mr-1">最佳</span>';
    if (fastestTrainTime !== null && m.trainTime === fastestTrainTime && m.trainTime > 0) {
      tags += '<span class="text-[10px] bg-accent-500/15 text-accent-400 px-1.5 py-0.5 rounded-full">⚡ 最快</span>';
    }
    if (!tags) tags = '<span class="text-dark-600 text-xs">—</span>';

    // 推論延遲 (每筆樣本平均 ms) — 後端量測
    const lat = (typeof m.inferLatency === 'number' && m.inferLatency > 0)
      ? (m.inferLatency < 0.01 ? '<0.01ms' : m.inferLatency.toFixed(3) + 'ms')
      : '<span class="text-dark-600">—</span>';

    // 操作:分析 → 跳洞察; 預測 → batch predict; ✕ → 刪除
    // Pipeline 舊 placeholder (type === 'daniel_pipeline') 是前端假 id,後端查 DB 會 404 → 不顯示預測按鈕
    const isLegacyPipelinePlaceholder = m.type === 'daniel_pipeline';
    const predictBtn = isLegacyPipelinePlaceholder
      ? `<span class="text-xs text-dark-600" title="舊版 Pipeline 紀錄,後端沒有 estimator pickle。重新訓練即可啟用預測。">預測 (需重訓)</span>`
      : `<button class="text-xs text-accent-400 hover:text-accent-300 transition-colors" onclick="leaderboardPredict('${m.id}', this)" title="用上方上傳的 test CSV 批次預測,下載對應 CSV">預測</button>`;
    const actions = m.id
      ? `<span class="flex items-center gap-2">` +
        `<button class="text-xs text-primary-400 hover:text-primary-300 transition-colors" onclick="analyzeModel('${m.id}')">分析</button>` +
        predictBtn +
        `<button class="text-xs text-danger-400 hover:text-danger-300 transition-colors" onclick="deleteModelById('${m.id}', this)" title="刪除模型">✕</button>` +
        `</span>`
      : '<span class="text-dark-600 text-xs">—</span>';

    // 資料來源 badge + 去掉名稱的 [原始]/[預處理] 文字前綴
    const cleanName = m.name.replace(/^\[(原始|預處理)\]\s*/, '');
    let srcBadge;
    if (m.dataSource === 'preprocessed') {
      srcBadge = '<span class="inline-block px-2 py-0.5 rounded-md text-[11px] font-semibold bg-accent-500/15 text-accent-400 border border-accent-500/30">預處理</span>';
      tr.classList.add('border-l-2', 'border-l-accent-500/40');
    } else if (m.dataSource === 'raw') {
      srcBadge = '<span class="inline-block px-2 py-0.5 rounded-md text-[11px] font-semibold bg-primary-500/15 text-primary-400 border border-primary-500/30">原始</span>';
      tr.classList.add('border-l-2', 'border-l-primary-500/40');
    } else {
      srcBadge = '<span class="text-dark-500 text-xs">—</span>';
    }

    // 訓練時間自適應格式: < 1s 用 ms,< 60s 用 s,< 60min 用 mm:ss,大於 1h 用 hh:mm
    const trainTimeStr = (typeof m.trainTime === 'number' && isFinite(m.trainTime))
      ? _fmtTrainTime(m.trainTime)
      : '<span class="text-dark-500">—</span>';
    // 模型大小:小於 1MB 不顯示 (sklearn 大多 < 1MB);> 100MB 改成 File 標籤提醒
    // 是 daniel ensemble 才有,sklearn 模型不會帶這欄
    const _mbVal = (typeof m.estimatorMb === 'number') ? m.estimatorMb : null;
    let sizeBadge = '';
    if (_mbVal !== null && _mbVal >= 1) {
      const isFile = _mbVal > 50;   // 對應 bootstrap.MODEL_BLOB_FILE_THRESHOLD_MB
      const cls = isFile
        ? 'bg-warning-500/15 text-warning-400 border-warning-500/30'
        : 'bg-dark-700/40 text-dark-400 border-dark-600/40';
      const tip = isFile
        ? `存在 model_blobs/ 檔案 (${_mbVal}MB) — 刪掉檔案模型就不能 batch predict / SHAP`
        : `存在 DB bytea 欄 (${_mbVal}MB)`;
      sizeBadge = `<span class="inline-block ml-2 px-1.5 py-0.5 rounded text-[9px] font-mono border ${cls}" title="${tip}">${_mbVal}MB${isFile ? ' · File' : ''}</span>`;
    }
    tr.innerHTML = `
      <td class="py-3 px-4"><input type="checkbox" class="model-select-cb" data-idx="${i}"></td>
      <td class="py-3 px-4">${rankEl}</td>
      <td class="py-3 px-4">${srcBadge}</td>
      <td class="py-3 px-4"><span class="font-medium">${escapeHtml(cleanName)}</span>${sizeBadge}</td>
      ${extraCols}
      <td class="py-3 px-4 font-mono text-xs">${trainTimeStr}</td>
      <td class="py-3 px-4 font-mono text-xs">${lat}</td>
      <td class="py-3 px-4">${tags}</td>
      <td class="py-3 px-4">${actions}</td>
    `;
    tbody.appendChild(tr);

    // 超參數展開列 — sklearn 模型有細部超參數;pipeline ensemble 通常無,則不插這列
    const hpInfo = _hyperparamChipsHtml(m.hyperparameters, 16);
    if (hpInfo) {
      const hpTr = document.createElement('tr');
      hpTr.className = 'bg-dark-900/20 border-b border-dark-700/30';
      hpTr.innerHTML = `
        <td></td>
        <td colspan="10" class="py-1.5 px-4">
          <details>
            <summary class="text-[11px] text-dark-400 cursor-pointer hover:text-primary-300 select-none inline-flex items-center gap-1">
              <span>⚙</span><span>超參數 (${hpInfo.count})</span>
            </summary>
            <div class="mt-1.5 leading-relaxed">${hpInfo.html}</div>
          </details>
        </td>
      `;
      tbody.appendChild(hpTr);
    }

    // U2:Daniel pipeline ensemble 額外展開「每個基模型的 OOF」可摺疊區塊。
    // perModel 在 bundle JSON 裡有,但目前只藏在 Pipeline 詳細卡;這裡讓使用者
    // 在主排行榜就能直接掃 5-7 個基模型的 OOF 排名,點任何一列也能對應到 algo 名稱。
    const perModel = m.danielPerModel || m.perModel || (m.metrics && m.metrics.perModel) || [];
    if (perModel.length > 0) {
      const sorted = [...perModel]
        .filter(pm => pm.oofScore != null)
        .sort((a, b) => (b.oofScore || 0) - (a.oofScore || 0));
      if (sorted.length > 0) {
        const bmTr = document.createElement('tr');
        bmTr.className = 'bg-dark-900/20 border-b border-dark-700/30';
        const rows = sorted.map((pm, j) => {
          const cleanTag = (pm.tag || '').replace(/^reg_/, '');
          const algo = cleanTag.split('_')[0];
          const algoUpper = algo.length > 8 ? algo : algo.toUpperCase();
          return `<div class="flex items-center justify-between text-[11px] py-0.5 ${j === 0 ? 'text-warning-300 font-semibold' : 'text-dark-300'}">
            <span class="flex items-center gap-2">
              <span class="font-mono text-[9px] text-dark-500 w-4 text-right">${j + 1}</span>
              <span class="font-mono">${escapeHtml(algoUpper)}</span>
              <span class="text-[9px] text-dark-600">${escapeHtml(cleanTag)}</span>
            </span>
            <span class="font-mono">${pm.oofScore.toFixed(4)}</span>
          </div>`;
        }).join('');
        bmTr.innerHTML = `
          <td></td>
          <td colspan="10" class="py-1.5 px-4">
            <details>
              <summary class="text-[11px] text-dark-400 cursor-pointer hover:text-primary-300 select-none inline-flex items-center gap-1">
                <span>📊</span><span>基模型 OOF 排名 (${sorted.length})</span>
                <span class="text-[10px] text-dark-600 ml-1">— ensemble 由這幾個基模型合成</span>
              </summary>
              <div class="mt-1.5 pl-6 max-h-48 overflow-y-auto">${rows}</div>
            </details>
          </td>
        `;
        tbody.appendChild(bmTr);
      }
    }
  });

  // Bind compare functionality for real models
  initRealCompare(models, isReg);
}

function initRealCompare(models, isReg) {
  const oldBtn = document.getElementById('btn-compare');
  const selectAllCb = document.getElementById('select-all-models');
  if (!oldBtn) return;

  const newBtn = oldBtn.cloneNode(true);
  oldBtn.parentNode.replaceChild(newBtn, oldBtn);

  const updateCompareState = () => {
    const checked = document.querySelectorAll('.model-select-cb:checked');
    newBtn.disabled = checked.length < 2;
  };

  newBtn.disabled = true;

  // Rebind checkboxes
  document.querySelectorAll('.model-select-cb').forEach(cb => {
    cb.addEventListener('change', updateCompareState);
  });

  if (selectAllCb) {
    selectAllCb.addEventListener('change', () => {
      const cbs = document.querySelectorAll('.model-select-cb');
      cbs.forEach((cb, i) => { cb.checked = selectAllCb.checked && i < 3; });
      updateCompareState();
    });
  }

  // Compare button click
  newBtn.addEventListener('click', () => {
    const checkedIdxs = [...document.querySelectorAll('.model-select-cb:checked')].map(cb => parseInt(cb.dataset.idx));
    if (checkedIdxs.length < 2) return;
    const selected = checkedIdxs.map(i => models[i]).filter(Boolean);
    const modal = document.getElementById('compare-modal');
    modal.classList.remove('hidden');
    setTimeout(() => renderRealCompareCharts(selected, isReg), 100);
  });
}

function renderRealCompareCharts(models, isReg) {
  // Radar chart
  const radarChart = initChart('chart-compare-radar');
  if (radarChart) {
    const indicators = isReg
      ? [{ name: 'R²', max: 1 }, { name: '1-RMSE(norm)', max: 1 }, { name: '1-MAE(norm)', max: 1 }]
      : [{ name: 'Accuracy', max: 1 }, { name: 'F1', max: 1 }, { name: 'Precision', max: 1 }];

    const maxRMSE = Math.max(...models.map(m => m.metrics.testRMSE || 1));
    const maxMAE = Math.max(...models.map(m => m.metrics.testMAE || 1));
    const colors = ['#3b82f6', '#22d3ee', '#f59e0b', '#10b981'];

    const series = models.map((m, i) => ({
      name: m.name,
      value: isReg
        ? [Math.max(0, m.metrics.testR2), 1 - (m.metrics.testRMSE / maxRMSE), 1 - (m.metrics.testMAE / maxMAE)]
        : [m.metrics.testAccuracy || 0, m.metrics.f1 || 0, m.metrics.precision || 0],
      lineStyle: { color: colors[i % colors.length] },
      itemStyle: { color: colors[i % colors.length] },
      areaStyle: { color: colors[i % colors.length], opacity: 0.1 },
    }));

    radarChart.setOption({
      tooltip: { backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0', fontSize: 11 } },
      legend: { data: models.map(m => m.name), bottom: 0, textStyle: { color: '#94a3b8', fontSize: 10 } },
      radar: { indicator: indicators, axisName: { color: '#94a3b8', fontSize: 10 }, splitLine: { lineStyle: { color: '#1e293b' } }, splitArea: { areaStyle: { color: ['transparent'] } } },
      series: [{ type: 'radar', data: series }],
    });
  }

  // Bar comparison
  const confTitle = document.getElementById('compare-modal-conf-title');
  if (confTitle) confTitle.textContent = isReg ? '模型分數比較' : '混淆矩陣比較';
  
  const rocSection = document.getElementById('compare-modal-roc-section');
  if (rocSection) {
    if (isReg) rocSection.classList.add('hidden');
    else rocSection.classList.remove('hidden');
  }

  const confChart = initChart('chart-compare-confusion');
  if (confChart) {
    const scoreLabel = isReg ? 'R²' : 'Accuracy';
    const scores = models.map(m => m.metrics.testScore);
    const colors = scores.map(s => s > 0.8 ? '#10b981' : s > 0.5 ? '#f59e0b' : '#f87171');

    confChart.setOption({
      tooltip: { trigger: 'axis', backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0', fontSize: 11 } },
      grid: { left: 50, right: 20, top: 15, bottom: 40 },
      xAxis: { type: 'category', data: models.map(m => m.name), axisLabel: { color: '#94a3b8', fontSize: 9, rotate: 20 } },
      yAxis: { type: 'value', name: scoreLabel, nameTextStyle: { color: '#64748b' }, axisLabel: { color: '#64748b', fontSize: 9 }, splitLine: { lineStyle: { color: '#1e293b' } } },
      series: [{
        type: 'bar', data: scores.map((s, i) => ({ value: s, itemStyle: { color: colors[i] } })),
        barWidth: '40%', itemStyle: { borderRadius: [4, 4, 0, 0] },
        label: { show: true, position: 'top', color: '#e2e8f0', fontSize: 9, formatter: p => p.value.toFixed(4) },
      }],
    });
  }
}

// ===== REAL INSIGHTS =====
function renderRealInsights() {
  const models = MLEngine.trainedModels;
  if (models.length === 0) return;

  // 確保 Pipeline 模式隱藏的 ECharts 面板重新顯示
  ['real-feature-importance-section', 'real-model-compare-section',
   'real-pred-scatter-section', 'real-whatif-section',
  ].forEach(id => { const el = document.getElementById(id); if (el) el.classList.remove('hidden'); });

  // 歷史訓練 — 三層級聯
  renderHistoryCascade(document.getElementById('insights-history-cascade'));

  const best = models[0];
  const isReg = best.taskType === 'regression';

  // Summary cards
  document.getElementById('insight-best-model').textContent = best.name;
  document.getElementById('insight-score-label').textContent = isReg ? 'R² 分數' : '準確率';
  document.getElementById('insight-best-score').textContent = isReg
    ? (typeof best.metrics?.testR2 === 'number' ? best.metrics.testR2.toFixed(4) : '—')
    : (typeof best.metrics?.testAccuracy === 'number' ? (best.metrics.testAccuracy * 100).toFixed(2) + '%' : '—');
  document.getElementById('insight-feature-count').textContent = best.featureNames.length;
  document.getElementById('insight-model-count').textContent = models.length;

  // SHAP section (隊友 AutoMLVisualizer)
  initShapSection(models, best);

  // Defer chart rendering so DOM has time to compute container sizes
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      renderRealFeatureImportance(best);
      renderRealModelCompare(models, isReg);
      renderRealPredScatter(best, isReg);
      if (isReg) renderRealResiduals(best);
      // What-If simulator (DOM-only, no ECharts; safe to call immediately too)
      renderRealWhatIf(models);
    });
  });
}

// ===== INSIGHTS 空白狀態 (完全沒有模型時) =====
function _renderInsightsEmptyState() {
  const safeSet = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  safeSet('insight-best-model', '—');
  safeSet('insight-best-score', '—');
  safeSet('insight-feature-count', '—');
  safeSet('insight-model-count', '0');

  // 在 SHAP section 顯示引導提示
  const errEl = document.getElementById('shap-error');
  if (errEl) {
    errEl.classList.remove('hidden');
    errEl.textContent = '尚未訓練任何模型。請先在「實驗室」訓練 sklearn 模型，或在「Pipeline」完成訓練後點選「SHAP 分析」按鈕。';
  }
}

// ===== PIPELINE SHAP-ONLY INSIGHTS (當無 sklearn 模型但有 pipeline bestModelId) =====
async function _renderPipelineShapOnly(modelId) {
  const safeSet = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };

  // 隱藏 ECharts 面板（特徵重要性 / 模型比較 / 預測散點圖）— pipeline 無這些資料
  // ⚠️ 不隱藏 real-whatif-section — 等下面從後端拿到特徵統計後再決定是否顯示
  ['real-feature-importance-section', 'real-model-compare-section',
   'real-pred-scatter-section', 'insights-history-cascade',
  ].forEach(id => { const el = document.getElementById(id); if (el) el.classList.add('hidden'); });

  // 顯示 SHAP section（有時被 demo 模式 CSS 隱藏）
  const shapSection = document.getElementById('real-shap-section');
  if (shapSection) shapSection.classList.remove('hidden');

  // Summary cards
  safeSet('insight-best-model', 'Pipeline 最佳模型');
  safeSet('insight-score-label', 'OOF 分數');
  safeSet('insight-best-score', '—');      // 後端沒有回傳到這裡，顯示 '—'
  safeSet('insight-feature-count', '—');   // loadShapFigures 後會更新
  safeSet('insight-model-count', '1');

  // 合成 model entry（testAccuracy=0 → dropdown 只顯示名稱，不顯示 Acc）
  const syntheticModel = {
    id: modelId,
    name: 'Pipeline 最佳模型',
    featureNames: [],
    metrics: { testAccuracy: 0, testScore: 0, testR2: 0 },
    taskType: 'classification',
  };

  _shapRequestedModelId = modelId;
  initShapSection([syntheticModel], syntheticModel);

  // 自動觸發一次載入（requestAnimationFrame 確保 DOM 已更新）
  requestAnimationFrame(() => loadShapFigures());

  // ── What-If：從後端拿特徵統計，合成可用的 model entry ──────────
  try {
    const info = await ApiClient.get(`/api/model/${modelId}/info`);
    if (info && info.featureNames && info.featureNames.length > 0) {
      const n = info.featureNames.length;
      // 建立 featureStats（{name, mean, std, min, max}）
      const featureStats = info.featureNames.map((name, i) => ({
        name,
        mean: info.featureMeans[i] ?? 0,
        std:  info.featureStds[i]  ?? 1,
        min:  info.featureMin[i]   ?? (info.featureMeans[i] - 3),
        max:  info.featureMax[i]   ?? (info.featureMeans[i] + 3),
      }));
      const whatIfModel = {
        id:           modelId,
        name:         info.modelName || 'Pipeline 最佳模型',
        featureNames: info.featureNames,
        featureStats,
        means:        info.featureMeans,
        stds:         info.featureStds,
        taskType:     info.taskType || 'regression',
        targetName:   info.targetName || '目標變數',
        metrics:      { testR2: 0, testAccuracy: 0 },
      };
      // 更新摘要卡的特徵數
      safeSet('insight-feature-count', n);
      // 渲染 What-If 模擬器
      renderRealWhatIf([whatIfModel]);
    } else {
      // 後端沒有特徵資訊，隱藏 What-If 區塊
      const ws = document.getElementById('real-whatif-section');
      if (ws) ws.classList.add('hidden');
    }
  } catch (e) {
    console.warn('[What-If] Pipeline model info 載入失敗:', e);
    const ws = document.getElementById('real-whatif-section');
    if (ws) ws.classList.add('hidden');
  }
}

// ===== SHAP SECTION (Plotly figures from /api/visualize/shap) =====
let _shapInitialized = false;
let _shapRequestedModelId = null;   // 排行榜「分析」按鈕指定要看的模型
let _shapLastResponse = null;       // 預計算模式:存上一次 response,給 family 切換用


// 從 fold tag (如 'reg_xgb_raw_stat_c0_ccb08f_yrs') 還原乾淨演算法名 'XGB'
function _shapExtractAlgoName(tag) {
  if (!tag) return '?';
  const t = tag.startsWith('reg_') ? tag.slice(4) : tag;
  // 跟後端 _shap_family_of_tag 同一份名單,longest-prefix 匹配,跟 daniel 命名對齊
  const KNOWN = [
    // Tabular(ML)
    ['extra_trees', 'ExtraTrees'],
    ['catboost', 'CatBoost'],
    ['logreg', 'LogReg'],
    ['xgb', 'XGB'],
    ['lgbm', 'LGBM'],
    ['ridge', 'Ridge'],
    ['knn', 'KNN'],
    ['rf', 'RF'],
    // DL
    ['resnet1d', 'ResNet1D'],
    ['patchtst', 'PatchTST'],
    ['transformer', 'Transformer'],
    ['cnn1d', 'CNN1D'],
    ['tcn', 'TCN'],
    ['tsnet', 'TSNet'],
    ['mlp', 'MLP'],
  ];
  for (const [key, label] of KNOWN) {
    if (t.startsWith(key + '_')) return label;
  }
  return (t.split('_')[0] || '?').toUpperCase();
}


// 預計算 SHAP — ML + DL 並列顯示 (6 張 = 2×3),不打後端
// ML 在上、DL 在下;沒 DL 就只顯示 ML section。Header 各自顯示演算法 + OOF 分數
function _renderShapPrecomputed() {
  if (!_shapLastResponse || !_shapLastResponse.shapPlots) return;
  const sp = _shapLastResponse.shapPlots;
  const cfg = { responsive: true, displaylogo: false };

  // ── ML (tabular) section ──
  const mlSection = document.getElementById('shap-section-ml');
  const mlHeader  = document.getElementById('shap-ml-header');
  if (sp.tabular) {
    const algoName = _shapExtractAlgoName(sp.tabular.modelTag);
    const score = sp.tabular.oofScore != null ? sp.tabular.oofScore.toFixed(4) : '?';
    document.getElementById('shap-ml-algo').textContent = `最佳 ML: ${algoName}`;
    document.getElementById('shap-ml-meta').innerHTML =
      `OOF=${score} <span class="text-dark-600 ml-1" title="${escapeHtml(sp.tabular.modelTag || '')}">${escapeHtml((sp.tabular.modelTag || '').slice(0, 32))}</span>`;
    mlHeader?.classList.remove('hidden');
    if (sp.tabular.global)     Plotly.newPlot('shap-fig-global',     sp.tabular.global.data,     sp.tabular.global.layout,     cfg);
    if (sp.tabular.waterfall)  Plotly.newPlot('shap-fig-waterfall',  sp.tabular.waterfall.data,  sp.tabular.waterfall.layout,  cfg);
    if (sp.tabular.dependence) Plotly.newPlot('shap-fig-dependence', sp.tabular.dependence.data, sp.tabular.dependence.layout, cfg);
    mlSection?.classList.remove('hidden');
  } else {
    mlSection?.classList.add('hidden');
    mlHeader?.classList.add('hidden');
  }

  // ── DL section ──
  const dlSection = document.getElementById('shap-section-dl');
  if (sp.dl) {
    const algoName = _shapExtractAlgoName(sp.dl.modelTag);
    const score = sp.dl.oofScore != null ? sp.dl.oofScore.toFixed(4) : '?';
    document.getElementById('shap-dl-algo').textContent = `最佳 DL: ${algoName}`;
    document.getElementById('shap-dl-meta').innerHTML =
      `OOF=${score} <span class="text-dark-600 ml-1" title="${escapeHtml(sp.dl.modelTag || '')}">${escapeHtml((sp.dl.modelTag || '').slice(0, 32))}</span>`;
    if (sp.dl.global)     Plotly.newPlot('shap-fig-global-dl',     sp.dl.global.data,     sp.dl.global.layout,     cfg);
    if (sp.dl.waterfall)  Plotly.newPlot('shap-fig-waterfall-dl',  sp.dl.waterfall.data,  sp.dl.waterfall.layout,  cfg);
    if (sp.dl.dependence) Plotly.newPlot('shap-fig-dependence-dl', sp.dl.dependence.data, sp.dl.dependence.layout, cfg);
    dlSection?.classList.remove('hidden');
  } else {
    dlSection?.classList.add('hidden');
  }

  // 交互特徵下拉:用 ML section 的 featureNames(優先);DL 沒有就用 DL 的
  const featSel = document.getElementById('shap-target-feature');
  const featList = (sp.tabular?.featureNames) || (sp.dl?.featureNames) || [];
  if (featSel && featList.length) {
    featSel.innerHTML = '';
    featList.forEach(fn => {
      const o = document.createElement('option');
      o.value = fn; o.textContent = fn;
      featSel.appendChild(o);
    });
  }
}

// 舊 API 名稱保留 — 內部直接轉呼新版(沒 family 切換概念了)
function _renderShapForFamily(_unused) { _renderShapPrecomputed(); }


// 排行榜 / 結果表的「分析」按鈕呼叫:跳到洞察頁並指定 SHAP 要顯示哪個模型
function analyzeModel(modelId) {
  _shapRequestedModelId = modelId || null;
  navigateTo('insights');
}

// 排行榜上方的 file input — 給 leaderboardPredict() 用
function _lbPredictTestFile() {
  return document.getElementById('lb-predict-test-csv')?.files?.[0] || null;
}
function _lbPredictSampleFile() {
  return document.getElementById('lb-predict-sample-csv')?.files?.[0] || null;
}
function _lbSetStatus(msg, level) {
  const el = document.getElementById('lb-predict-status');
  if (!el) return;
  const colorMap = { info: 'text-dark-400', success: 'text-success-400',
                     error: 'text-danger-400', warning: 'text-warning-400' };
  el.className = `text-xs px-4 pb-3 ${colorMap[level] || colorMap.info}`;
  el.textContent = msg || '';
  el.classList.toggle('hidden', !msg);
}

// ── 訓練 elapsed timer ────────────────────────────────────────────────
// 對所有引擎(sklearn / daniel / autogluon)通用 — setInterval 1s 更新顯示。
// 用全域 _trainingTimerId 確保任何時候只有一個 timer 在跑。
let _trainingTimerId = null;
let _trainingTimerStart = null;

function _trainingTimerStart_(timeLimitSec) {
  _trainingTimerStop();   // 防重入,先清舊的
  _trainingTimerStart = Date.now();
  const wrap = document.getElementById('exp-elapsed-timer');
  const sep  = document.getElementById('exp-time-limit-sep');
  const lim  = document.getElementById('exp-time-limit-value');
  if (!wrap) return;
  wrap.classList.remove('hidden');
  if (typeof timeLimitSec === 'number' && timeLimitSec > 0) {
    sep?.classList.remove('hidden');
    lim?.classList.remove('hidden');
    if (lim) lim.textContent = _fmtMmSs(timeLimitSec);
  } else {
    sep?.classList.add('hidden');
    lim?.classList.add('hidden');
  }
  // 立刻更新一次 + 每秒推
  _trainingTimerTick(timeLimitSec);
  _trainingTimerId = setInterval(() => _trainingTimerTick(timeLimitSec), 1000);
}

function _trainingTimerTick(timeLimitSec) {
  const valEl = document.getElementById('exp-elapsed-value');
  if (!valEl || _trainingTimerStart == null) return;
  const elapsed = Math.floor((Date.now() - _trainingTimerStart) / 1000);
  valEl.textContent = _fmtMmSs(elapsed);
  // 超過 time_limit 改紅色提醒(訓練應該快結束了)
  if (timeLimitSec && elapsed > timeLimitSec) {
    valEl.className = 'text-danger-400';
  } else if (timeLimitSec && elapsed > timeLimitSec * 0.8) {
    valEl.className = 'text-warning-300';
  } else {
    valEl.className = 'text-dark-200';
  }
}

function _trainingTimerStop() {
  if (_trainingTimerId != null) {
    clearInterval(_trainingTimerId);
    _trainingTimerId = null;
  }
  _trainingTimerStart = null;
}

function _fmtMmSs(sec) {
  sec = Math.max(0, Math.floor(sec));
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

// 訓練時間 (ms) 自適應顯示 — sklearn 通常秒級,daniel/autogluon 分鐘起跳,
// 純 'XXXXXms' 太難讀
function _fmtTrainTime(ms) {
  if (!ms || ms < 0) return '—';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const sec = ms / 1000;
  if (sec < 60) return `${sec.toFixed(1)}s`;
  const minutes = sec / 60;
  if (minutes < 60) {
    const m = Math.floor(minutes);
    const s = Math.round(sec - m * 60);
    return `${m}m ${s}s`;
  }
  const h = Math.floor(minutes / 60);
  const m = Math.round(minutes - h * 60);
  return `${h}h ${m}m`;
}

// 批次預測 SSE 進度條 helpers
function _lbProgressShow() {
  document.getElementById('lb-predict-progress-wrap')?.classList.remove('hidden');
}
function _lbProgressHide() {
  document.getElementById('lb-predict-progress-wrap')?.classList.add('hidden');
  _lbProgressSet(0, '');
}
function _lbProgressSet(pct, label) {
  const bar = document.getElementById('lb-predict-progress-bar');
  const pctEl = document.getElementById('lb-predict-progress-pct');
  const lblEl = document.getElementById('lb-predict-progress-label');
  if (bar) bar.style.width = Math.max(0, Math.min(100, pct)) + '%';
  if (pctEl) pctEl.textContent = Math.round(pct) + '%';
  if (lblEl && label != null) lblEl.textContent = label;
}

// 排行榜每列的「預測」按鈕觸發 — 用上面 file input 的測試 CSV + 範本跑批次預測
// Daniel ensemble 走 SSE 版 (邊跑邊推進度 + 可取消);非 ensemble 走舊版一次性 endpoint
async function leaderboardPredict(modelId, btnEl) {
  const testFile   = _lbPredictTestFile();
  const sampleFile = _lbPredictSampleFile();
  if (!testFile) {
    _lbSetStatus('✗ 請先在上方選擇「測試 CSV」', 'error');
    return;
  }
  if (typeof ApiClient === 'undefined' || !ApiClient.enabled) {
    _lbSetStatus('✗ 批次預測需要開啟「使用 Python 後端 API」', 'error');
    return;
  }

  // 判斷:Daniel ensemble (canPredict=true 且 type=daniel_pipeline_ensemble) → SSE 版
  // 從 _trainingHistory 找 model 看 type
  let useSSE = false;
  for (const h of (_trainingHistory || [])) {
    const m = (h.models || []).find(mm => mm.id === modelId);
    if (m && (m.type === 'daniel_pipeline_ensemble' || m.canPredict)) {
      useSSE = true; break;
    }
  }

  const origLabel = btnEl?.textContent;
  const setPredictingBtn = (txt) => {
    if (btnEl) { btnEl.disabled = true; btnEl.textContent = txt; }
  };
  const resetBtn = () => {
    if (btnEl) { btnEl.disabled = false; btnEl.textContent = origLabel || '預測'; }
  };

  if (!useSSE) {
    // 舊版一次性 path (sklearn 模型用,通常很快)
    setPredictingBtn('預測中…');
    _lbSetStatus(`處理中 — 模型 ${modelId} × ${testFile.name}`, 'info');
    setGlobalStatus('running', `批次預測中 — ${testFile.name}`);
    try {
      const blob = await ApiClient.predictBatch(modelId, testFile, sampleFile);
      const outName = sampleFile
        ? 'submission.csv'
        : `${testFile.name.replace(/\.[^.]+$/, '')}_predicted.csv`;
      _triggerDownload(blob, outName);
      _lbSetStatus(`✓ 預測完成,已下載 ${outName}`, 'success');
      setGlobalStatus('success', `預測完成 — ${outName}`);
      notify('批次預測完成 ✓', `${testFile.name} → ${outName}`, 'success');
    } catch (e) {
      _lbSetStatus(`✗ 預測失敗: ${e.message}`, 'error');
      setGlobalStatus('error', '預測失敗');
      notify('批次預測失敗', e.message, 'error');
    } finally {
      resetBtn();
    }
    return;
  }

  // SSE 版:可顯示 per-config 進度 + 可取消
  setPredictingBtn('取消');
  const controller = new AbortController();
  // 把取消綁到按鈕 — 點按鈕第二次 = abort fetch = 後端 cancel_token
  const cancelHandler = (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    controller.abort();
  };
  if (btnEl) {
    btnEl.removeAttribute('onclick');
    btnEl.addEventListener('click', cancelHandler);
  }

  _lbSetStatus(`啟動中 — 模型 ${modelId} × ${testFile.name}`, 'info');
  setGlobalStatus('running', `批次預測中 — ${testFile.name}`);

  try {
    const fd = new FormData();
    fd.append('modelId', modelId);
    fd.append('file', testFile);
    if (sampleFile) fd.append('sampleFile', sampleFile);
    const apiBase = (typeof ApiClient !== 'undefined' && ApiClient.baseUrl) ? ApiClient.baseUrl : '';
    // 跟既有 training stream 對齊 — 不要 credentials='include' (後端 CORS 是 *,
    // credentials 模式下瀏覽器會擋),改用 Bearer token 傳 auth
    const headers = {};
    if (typeof AuthClient !== 'undefined' && AuthClient.token) {
      headers['Authorization'] = `Bearer ${AuthClient.token}`;
    }
    const resp = await fetch(`${apiBase}/api/predict/batch/stream`, {
      method: 'POST', body: fd, headers, signal: controller.signal,
    });
    if (!resp.ok) {
      const err = await resp.text().catch(() => `HTTP ${resp.status}`);
      throw new Error(err.slice(0, 300));
    }

    // SSE 解析:event: + data: ...  雙換行為界
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finalEvent = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // 拆 SSE chunks
      let sep;
      while ((sep = buffer.indexOf('\n\n')) >= 0) {
        const chunk = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const ev = _parseSSE(chunk);
        if (!ev) continue;
        if (ev.event === 'progress') {
          const p = ev.data;
          if (p.phase === 'start') {
            _lbSetStatus(`開始 — ${p.totalConfigs} 個 config × N 個 fold,N=${p.nSamples} 樣本`, 'info');
            _lbProgressShow();
            _lbProgressSet(0, `等待 ${p.totalConfigs} 個 config 跑完`);
          } else if (p.phase === 'config_start') {
            _lbSetStatus(`[${p.done}/${p.total}] 跑 ${p.tag} (${p.nFolds} folds)...`, 'info');
            _lbProgressSet(Math.round((p.done / Math.max(p.total, 1)) * 95),
              `跑 ${p.tag}`);   // 進度條進到「該 config 開始前」的位置;留 5% 給 ensemble combine
          } else if (p.phase === 'config_done') {
            _lbSetStatus(`[${p.done}/${p.total}] ${p.tag} 完成 (${p.elapsedSec}s)`, 'info');
            _lbProgressSet(Math.round((p.done / Math.max(p.total, 1)) * 95),
              `${p.done}/${p.total} configs 完成`);
          } else if (p.phase === 'ensemble_combine') {
            _lbSetStatus(`所有 config 跑完,blender + stacker 合併中...`, 'info');
            _lbProgressSet(97, 'blender + stacker 合併中');
          }
        } else if (ev.event === 'error') {
          throw new Error(`${ev.data.kind}: ${ev.data.msg}`);
        } else if (ev.event === 'done') {
          finalEvent = ev.data;
        }
      }
    }

    if (!finalEvent) throw new Error('SSE 流結束但沒收到 done 事件');

    // base64 → blob → 觸發下載
    const bin = atob(finalEvent.csvBase64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    const blob = new Blob([arr], { type: 'text/csv;charset=utf-8' });
    _triggerDownload(blob, finalEvent.filename);
    _lbProgressSet(100, '✓ 完成');
    _lbSetStatus(`✓ 預測完成 (${finalEvent.rowCount} 列) — 已下載 ${finalEvent.filename}`, 'success');
    setGlobalStatus('success', `預測完成 — ${finalEvent.filename}`);
    notify('批次預測完成 ✓', `${testFile.name} → ${finalEvent.filename}`, 'success');
    // 3 秒後自動收起進度條
    setTimeout(() => _lbProgressHide(), 3000);
  } catch (e) {
    _lbProgressHide();
    if (e.name === 'AbortError') {
      _lbSetStatus('✗ 已取消批次預測', 'warning');
      setGlobalStatus('warning', '預測已取消');
    } else {
      _lbSetStatus(`✗ 預測失敗: ${e.message}`, 'error');
      setGlobalStatus('error', '預測失敗');
      notify('批次預測失敗', e.message, 'error');
    }
  } finally {
    if (btnEl) {
      btnEl.removeEventListener('click', cancelHandler);
      // 還原原 onclick (inline)
      btnEl.setAttribute('onclick', `leaderboardPredict('${modelId}', this)`);
    }
    resetBtn();
  }
}

// SSE chunk 解析:`event: <name>\ndata: <json>` → {event, data}
function _parseSSE(chunk) {
  const lines = chunk.split('\n');
  let event = 'message';
  const dataLines = [];
  for (const line of lines) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return null;
  try { return { event, data: JSON.parse(dataLines.join('\n')) }; }
  catch { return { event, data: dataLines.join('\n') }; }
}

function _triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

// 排行榜 header 中的緊湊 file picker — change 時把 label 文字換成檔名 + 顯示清除鈕
// 重點:input 用 display:none 隱藏時,某些瀏覽器不會自動把 label click 派發過去 →
//      改用 JS 顯式 input.click(),確保任何點擊都能開檔案選取視窗
function _lbWirePredictInputs() {
  const wire = (labelId, inputId, textId, clearId, defaultText) => {
    const label  = document.getElementById(labelId);
    const input  = document.getElementById(inputId);
    const text   = document.getElementById(textId);
    const clear  = document.getElementById(clearId);
    if (!input || input.__wired) return;
    input.__wired = true;

    const refresh = () => {
      const f = input.files?.[0];
      if (f && text) {
        const short = f.name.length > 24 ? f.name.slice(0, 22) + '…' : f.name;
        text.textContent = short;
        text.classList.remove('text-dark-400');
        text.classList.add('text-primary-300');
        clear?.classList.remove('hidden');
      } else if (text) {
        text.textContent = defaultText;
        text.classList.add('text-dark-400');
        text.classList.remove('text-primary-300');
        clear?.classList.add('hidden');
      }
    };
    input.addEventListener('change', refresh);

    // label 整塊都可點 → 觸發 input
    if (label) {
      label.addEventListener('click', (e) => {
        // 點 ✕ 清除鈕時,不要也觸發檔案選取
        if (clear && (e.target === clear || clear.contains(e.target))) return;
        // 點到 input 自己時 (Chrome 偶爾會 native 觸發),不要重複觸發
        if (e.target === input) return;
        e.preventDefault();
        input.click();
      });
    }

    if (clear) {
      clear.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        input.value = '';
        refresh();
      });
    }
  };
  wire('lb-predict-test-label',   'lb-predict-test-csv',   'lb-predict-test-text',
       'lb-predict-test-clear',   '測試 CSV');
  wire('lb-predict-sample-label', 'lb-predict-sample-csv', 'lb-predict-sample-text',
       'lb-predict-sample-clear', 'submission 範本 (選填)');
}

async function deleteModelById(modelId, btnEl) {
  if (!modelId) return;
  if (!confirm('確定刪除此模型？此操作無法復原。')) return;
  try {
    if (btnEl) { btnEl.disabled = true; btnEl.textContent = '…'; }
    if (typeof ApiClient !== 'undefined' && ApiClient.enabled) {
      await ApiClient.modelDelete(modelId);
    }
    // Remove from in-memory stores
    MLEngine.trainedModels = MLEngine.trainedModels.filter(m => m.id !== modelId);
    _trainingHistory.forEach(h => {
      if (h.models) h.models = h.models.filter(m => m.id !== modelId);
    });
    showToast('模型已刪除', { type: 'success' });
    renderRealLeaderboard();
    updateDashboardRealMetrics();
  } catch (e) {
    showToast('刪除失敗', { type: 'error', msg: e.message });
    if (btnEl) { btnEl.disabled = false; btnEl.textContent = '✕'; }
  }
}

// 依選定模型重填「交互特徵」下拉 (raw / 預處理 的特徵名不同)
function _populateShapFeatureSelect(model) {
  const featSel = document.getElementById('shap-target-feature');
  if (!featSel) return;
  featSel.innerHTML = '';
  (model?.featureNames || []).forEach(fn => {
    const opt = document.createElement('option');
    opt.value = fn; opt.textContent = fn;
    featSel.appendChild(opt);
  });
}

function initShapSection(models, best) {
  const modelSel = document.getElementById('shap-model-select');
  const btn = document.getElementById('btn-shap-refresh');
  if (!modelSel || !btn) return;  // 沒掛這段 HTML 就跳過

  // 填模型下拉
  modelSel.innerHTML = '';
  models.forEach(m => {
    if (!m.id) return;  // 訓練失敗的模型沒 id
    const opt = document.createElement('option');
    opt.value = m.id;
    // Pipeline 合成 model (testAccuracy=0) 只顯示名稱，不附分數
    const hasScore = m.metrics && (m.metrics.testAccuracy > 0 || m.metrics.testR2 > 0);
    const scoreStr = !hasScore ? '' : m.taskType === 'regression'
      ? ' (R²=' + m.metrics.testR2.toFixed(3) + ')'
      : ' (Acc=' + (m.metrics.testAccuracy * 100).toFixed(1) + '%)';
    opt.textContent = m.name + scoreStr;
    modelSel.appendChild(opt);
  });

  // 決定預設選哪個:有指定 (從「分析」按鈕進來) 就用指定的,否則用最佳模型
  let targetId = best && best.id ? best.id : (models[0] && models[0].id);
  if (_shapRequestedModelId && models.some(m => m.id === _shapRequestedModelId)) {
    targetId = _shapRequestedModelId;
  }
  _shapRequestedModelId = null;  // 用完清掉
  if (targetId) modelSel.value = targetId;

  // 交互特徵下拉 — 依「目前選定的模型」而非 best (raw/預處理特徵不同)
  const curModel = models.find(m => m.id === modelSel.value) || best;
  _populateShapFeatureSelect(curModel);

  // 切換模型 → 重填交互特徵下拉 + 重算
  modelSel.onchange = () => {
    const m = models.find(x => x.id === modelSel.value);
    _populateShapFeatureSelect(m);
    loadShapFigures();
  };

  if (!_shapInitialized) {
    // 「重新計算」按鈕走強制 path:先清 lastLoadedKey 讓 reqKey 防禦失效,確保真的重發
    btn.addEventListener('click', () => {
      _shapLastLoadedKey = null;
      loadShapFigures();
    });
    // Item 1D:family 切換按鈕 (Tabular / DL) — 只在 source='precomputed' 時顯示
    document.getElementById('shap-family-tabular')?.addEventListener('click', () => _renderShapForFamily('tabular'));
    document.getElementById('shap-family-dl')?.addEventListener('click', (e) => {
      if (e.currentTarget.disabled) return;
      _renderShapForFamily('dl');
    });
    _shapInitialized = true;
  }

  // 自動觸發第一次
  loadShapFigures();
}

// 清空 SHAP 三張 Plotly 圖 — 切換歷史 / 模型時先清,避免使用者看到舊圖以為沒更新
function _clearShapFigures() {
  // 6 個 figure div 都清(ML + DL)
  ['shap-fig-global', 'shap-fig-waterfall', 'shap-fig-dependence',
   'shap-fig-global-dl', 'shap-fig-waterfall-dl', 'shap-fig-dependence-dl'].forEach(id => {
    const el = document.getElementById(id);
    if (el && typeof Plotly !== 'undefined') {
      try { Plotly.purge(el); } catch (e) {}
      el.innerHTML = '';
    }
  });
}

// SHAP 重入防護:1) 同參數的 in-flight 請求不再發 2) 切頁觸發新的會 abort 舊的
let _shapInFlightController = null;
let _shapInFlightKey = null;
let _shapLastLoadedKey = null;

async function loadShapFigures() {
  if (typeof Plotly === 'undefined') {
    document.getElementById('shap-error').classList.remove('hidden');
    document.getElementById('shap-error').textContent = 'Plotly.js 未載入,請檢查網路或 CDN。';
    return;
  }
  const modelId = document.getElementById('shap-model-select').value;
  const sampleIndex = parseInt(document.getElementById('shap-sample-index').value) || 0;
  const targetFeature = document.getElementById('shap-target-feature').value || null;
  const errEl = document.getElementById('shap-error');
  const loadEl = document.getElementById('shap-loading');

  if (!modelId) {
    errEl.classList.remove('hidden');
    errEl.textContent = '請先選擇模型';
    _clearShapFigures();
    return;
  }

  // 重入防護:
  //   ① 同 model + 同 sample + 同 feature 已經 in-flight → 跳過(避免雙開 explainer)
  //   ② 同參數剛載完 + 圖還在 DOM → 跳過(換頁回來會誤觸,後端 sklearn 無法取消,
  //                                   會在 server 真的跑兩遍)
  //   ③ 「重新計算」按鈕走另一個 path,會直接呼叫 loadShapFigures 強制重發 → 不擋
  const reqKey = `${modelId}|${sampleIndex}|${targetFeature || ''}`;
  if (_shapInFlightKey === reqKey) {
    console.log('[shap] 同參數已 in-flight,跳過重發');
    return;
  }
  // 圖還在 DOM (Plotly 會塞個 .plotly 子元素) + 上次成功載入是同 key → 不重發
  const globalDiv = document.getElementById('shap-fig-global');
  if (_shapLastLoadedKey === reqKey && globalDiv && globalDiv.querySelector('.plotly')) {
    console.log('[shap] 同參數已載入,跳過重發 (要 refresh 請按「重新計算」)');
    return;
  }
  // 切到不同 model/params:abort 舊的 client-side fetch (server 還在算,但我們不再等回應)
  if (_shapInFlightController) {
    console.log('[shap] abort 上一個 fetch (server 端可能還在算)');
    try { _shapInFlightController.abort(); } catch (_) {}
  }

  errEl.classList.add('hidden');
  loadEl.classList.remove('hidden');
  // 先清空舊圖,讓使用者看到 loading 狀態,避免誤以為沒切換
  _clearShapFigures();

  _shapInFlightController = new AbortController();
  _shapInFlightKey = reqKey;
  const myController = _shapInFlightController;
  try {
    const res = await ApiClient.visualizeShap({
      modelId, sampleIndex, targetFeature, maxSamples: getSettings().shapSamples,
      signal: myController.signal,
    });
    // 自己被別人 abort 了 → 不要繼續 render
    if (myController.signal.aborted) return;
    const cfg = { responsive: true, displaylogo: false };

    // ── 新格式:訓練時預計算的雙 family SHAP (source='precomputed')──
    // ML + DL 並列同時顯示 (6 張 = 2×3),不再用 family selector 切換
    if (res.source === 'precomputed' && res.shapPlots) {
      _shapLastResponse = res;
      // 持久化徽章(取代 family selector)
      document.getElementById('shap-family-wrap')?.classList.add('hidden');   // 不再需要切換按鈕
      document.getElementById('shap-best-tag')?.classList.add('hidden');      // 標題改在 section header
      document.getElementById('shap-source-badge')?.classList.remove('hidden');
      // 兩個 section 一起渲染
      _renderShapPrecomputed();
      return;
    }

    // ── 舊格式:即時計算的單組 SHAP ──
    // 顯示 ML section(沒 header),隱藏 DL section
    _shapLastResponse = null;
    document.getElementById('shap-family-wrap')?.classList.add('hidden');
    document.getElementById('shap-source-badge')?.classList.add('hidden');
    document.getElementById('shap-best-tag')?.classList.add('hidden');
    document.getElementById('shap-ml-header')?.classList.add('hidden');   // 即時版沒分 family,不顯示「ML XGB」標題
    document.getElementById('shap-section-ml')?.classList.remove('hidden');
    document.getElementById('shap-section-dl')?.classList.add('hidden');
    Plotly.newPlot('shap-fig-global',     res.global.data,     res.global.layout,     cfg);
    Plotly.newPlot('shap-fig-waterfall',  res.waterfall.data,  res.waterfall.layout,  cfg);
    Plotly.newPlot('shap-fig-dependence', res.dependence.data, res.dependence.layout, cfg);

    // ── Pipeline 模式補丁：後端回傳 featureNames 後，補填「交互特徵」下拉 ──
    const featSel = document.getElementById('shap-target-feature');
    if (featSel && featSel.options.length === 0 && res.featureNames && res.featureNames.length) {
      res.featureNames.forEach(fn => {
        const o = document.createElement('option');
        o.value = fn; o.textContent = fn;
        featSel.appendChild(o);
      });
      // 更新 insight-feature-count card（pipeline 模式下原本顯示 '—'）
      const fcEl = document.getElementById('insight-feature-count');
      if (fcEl && fcEl.textContent === '—') fcEl.textContent = res.featureNames.length;
    }
  } catch (e) {
    // AbortError 不算錯誤,是我們主動取消的(切到別模型)
    if (e.name === 'AbortError' || /abort/i.test(e.message || '')) {
      console.log('[shap] 請求被 abort (正常)');
      return;
    }
    errEl.classList.remove('hidden');
    // 訊息明顯一點 — 通常是後端 model 找不到 (重啟過 / 歷史紀錄但 model 沒持久化)
    const msg = e.message.includes('404') || e.message.includes('不存在')
      ? '⚠ 此歷史模型已不在後端記憶體 (uvicorn 重啟後會丟失)。請重新訓練以查看 SHAP 解釋。'
      : `SHAP 載入失敗:${e.message}`;
    errEl.textContent = msg;
    _clearShapFigures();
  } finally {
    loadEl.classList.add('hidden');
    // 只清掉「自己」的 in-flight 標記;若中間又有新請求進來,別誤清新的
    if (_shapInFlightController === myController) {
      _shapInFlightController = null;
      _shapInFlightKey = null;
      _shapLastLoadedKey = reqKey;
    }
  }
}

// ===== REAL WHAT-IF SIMULATOR =====
const RealWhatIfState = { currentModel: null, values: [] };

function renderRealWhatIf(models) {
  const section = document.getElementById('real-whatif-section');
  if (!section) return;

  // Filter to models that have prediction infrastructure
  // - JS 模式: 需要 m.predict (callable function)
  // - API 模式: 需要 m.id (用來呼叫 /api/predict)
  const usable = models.filter(m =>
    m.featureStats?.length > 0 && m.means?.length > 0 && m.stds?.length > 0 &&
    (typeof m.predict === 'function' || m.id)
  );
  if (usable.length === 0) {
    section.classList.add('hidden');
    return;
  }
  section.classList.remove('hidden');

  // Populate model selector
  const modelSel = document.getElementById('real-whatif-model');
  modelSel.innerHTML = '';
  usable.forEach((m, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = i === 0 ? `${m.name} (最佳)` : m.name;
    modelSel.appendChild(opt);
  });

  const setModel = (idx) => {
    RealWhatIfState.currentModel = usable[idx];
    RealWhatIfState.values = usable[idx].featureStats.map(s => s.mean);
    buildRealWhatIfSliders();
    updateRealWhatIfPrediction();
  };

  modelSel.onchange = () => setModel(parseInt(modelSel.value));
  document.getElementById('real-whatif-reset').onclick = () => {
    RealWhatIfState.values = RealWhatIfState.currentModel.featureStats.map(s => s.mean);
    buildRealWhatIfSliders();
    updateRealWhatIfPrediction();
  };

  setModel(0);
}

// 算「漂亮的步進值」— 讓數字框上下鍵一次調整一個合理的量 (1 / 2 / 5 × 10^n)
function _niceStep(range) {
  if (!range || range <= 0) return 0.01;
  const raw = range / 100;                              // 目標:整個範圍約 100 階
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const nice = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
  return nice * mag;
}

function buildRealWhatIfSliders() {
  const wrap = document.getElementById('real-whatif-sliders');
  const m = RealWhatIfState.currentModel;
  if (!wrap || !m) return;

  wrap.innerHTML = '';
  m.featureNames.forEach((name, i) => {
    const stats = m.featureStats[i];
    const range = stats.max - stats.min;
    // 二元 0/1 欄位 (預處理 OneHot 出來的) → step=1,直接在 0/1 間切換
    const isBinary = (stats.min === 0 && stats.max === 1);
    // 滑桿:細步進 (好拖);數字框:漂亮步進 (上下鍵一次調整有感)
    const sliderStep = isBinary ? 1 : (range / 200 || 0.01);
    const numStep = isBinary ? 1 : _niceStep(range);
    const cur = RealWhatIfState.values[i];

    const div = document.createElement('div');
    div.className = 'slider-group';
    div.innerHTML = `
      <div class="flex items-center justify-between mb-1 gap-2">
        <label class="text-xs font-medium text-dark-200 truncate" title="${escapeHtml(name)}">${escapeHtml(name)}${isBinary ? ' <span class="text-dark-500">(0/1)</span>' : ''}</label>
        <input type="number" min="${stats.min}" max="${stats.max}" step="${numStep}" value="${formatRwifVal(cur)}" data-val-input="${i}" data-num-step="${numStep}"
          title="範圍 ${formatRwifVal(stats.min)} ~ ${formatRwifVal(stats.max)},上下鍵每次 ±${numStep}"
          class="w-24 bg-dark-800 border border-dark-600 rounded px-2 py-0.5 text-xs font-mono text-accent-400 text-right focus:border-accent-500 outline-none flex-shrink-0">
      </div>
      <input type="range" min="${stats.min}" max="${stats.max}" step="${sliderStep}" value="${cur}" data-feat-idx="${i}" class="slider-input w-full">
      <div class="flex justify-between text-[9px] text-dark-500 mt-0.5">
        <span>${formatRwifVal(stats.min)}</span>
        <span class="text-dark-600">μ ${formatRwifVal(stats.mean)}</span>
        <span>${formatRwifVal(stats.max)}</span>
      </div>
    `;
    wrap.appendChild(div);
  });

  // 滑桿拖動 → 更新數字框 + 狀態 + 重新預測
  wrap.querySelectorAll('input[type="range"]').forEach(input => {
    input.addEventListener('input', (e) => {
      const idx = parseInt(e.target.dataset.featIdx);
      const v = parseFloat(e.target.value);
      RealWhatIfState.values[idx] = v;
      const numInput = wrap.querySelector(`[data-val-input="${idx}"]`);
      if (numInput) numInput.value = formatRwifVal(v);
      updateRealWhatIfPrediction();
    });
  });

  // 數字框輸入 → 夾在 [min, max] 範圍內 + 同步滑桿 + 重新預測
  wrap.querySelectorAll('input[type="number"]').forEach(input => {
    const idx = parseInt(input.dataset.valInput);
    const stats = m.featureStats[idx];
    // 即時打字:更新狀態+滑桿,但不強制夾(讓使用者打完)
    input.addEventListener('input', (e) => {
      const v = parseFloat(e.target.value);
      if (isNaN(v)) return;
      RealWhatIfState.values[idx] = v;
      const slider = wrap.querySelector(`input[type="range"][data-feat-idx="${idx}"]`);
      if (slider) slider.value = v;  // 超出範圍時滑桿自動停在邊界
      updateRealWhatIfPrediction();
    });
    // 失焦 / 按 Enter / 點上下鍵:夾回範圍 + 對齊到 step 網格 (避免出現一堆醜小數)
    const numStep = parseFloat(input.dataset.numStep) || 0.01;
    input.addEventListener('change', (e) => {
      let v = parseFloat(e.target.value);
      if (isNaN(v)) v = stats.mean;
      v = Math.min(stats.max, Math.max(stats.min, v));        // clamp 到 [min,max]
      v = Math.round(v / numStep) * numStep;                  // snap 到 step 網格
      v = Math.min(stats.max, Math.max(stats.min, v));        // snap 後可能微超界,再夾一次
      e.target.value = formatRwifVal(v);
      RealWhatIfState.values[idx] = v;
      const slider = wrap.querySelector(`input[type="range"][data-feat-idx="${idx}"]`);
      if (slider) slider.value = v;
      updateRealWhatIfPrediction();
    });
  });
}

// 用 token 防止 slider 拖動時的舊請求蓋過新請求
let _whatIfReqToken = 0;

async function updateRealWhatIfPrediction() {
  const m = RealWhatIfState.currentModel;
  if (!m) return;

  const isApi = (typeof ApiClient !== 'undefined' && ApiClient.enabled && m.id);
  const isReg = m.taskType === 'regression';

  // ── 顯示 loading 狀態 ──────────────────────────────────────────
  const predEl     = document.getElementById('real-whatif-prediction');
  const baseEl     = document.getElementById('real-whatif-baseline');
  const deltaEl    = document.getElementById('real-whatif-delta');
  const probaEl    = document.getElementById('real-whatif-proba');   // may not exist in old HTML
  if (isApi) {
    if (predEl) predEl.textContent = '…';
    if (baseEl) baseEl.textContent = '…';
  }

  let pred = NaN, baseline = NaN;
  let predProba = null, baseProba = null;
  let predRespStd = null;   // 回歸用:5 fold std → ±1σ 區間

  if (isApi) {
    // API 模式: raw scale 值直接傳給後端,後端內部做 scaler.transform
    const myToken = ++_whatIfReqToken;
    try {
      const [predResp, baseResp] = await Promise.all([
        ApiClient.predict({ modelId: m.id, features: RealWhatIfState.values }),
        // baseline = scaler.mean_ (raw-scale training means) → transform 後 = 零向量
        ApiClient.predict({ modelId: m.id, features: m.means }),
      ]);
      // 過時的 response (slider 還在拖動) 直接丟掉
      if (myToken !== _whatIfReqToken) return;
      pred     = predResp.prediction;
      baseline = baseResp.prediction;
      // 回歸:接 predictionStd (5 fold std) → 給 UI 畫 ±1σ 信心區間
      if (isReg && typeof predResp.predictionStd === 'number') {
        predRespStd = predResp.predictionStd;
      }
      // 分類機率：取 pred 對應的 class 的機率
      if (!isReg && Array.isArray(predResp.proba) && Array.isArray(predResp.classes)) {
        const idx = predResp.classes.indexOf(pred);
        if (idx >= 0) predProba = predResp.proba[idx];
      }
      if (!isReg && Array.isArray(baseResp.proba) && Array.isArray(baseResp.classes)) {
        const idx = baseResp.classes.indexOf(baseline);
        if (idx >= 0) baseProba = baseResp.proba[idx];
      }
    } catch (err) {
      if (myToken !== _whatIfReqToken) return;
      console.warn('What-If predict 失敗:', err);
      if (predEl) predEl.textContent = '錯誤';
      if (baseEl) baseEl.textContent = '—';
      return;
    }
  } else {
    // 本地 JS 模式 (沿用原邏輯)
    const xNorm = RealWhatIfState.values.map((v, i) => (v - m.means[i]) / (m.stds[i] || 1));
    try { pred = m.predict(xNorm); } catch (e) { pred = NaN; }
    const baselineNorm = new Array(m.featureNames.length).fill(0);
    try { baseline = m.predict(baselineNorm); } catch (e) { baseline = NaN; }
  }

  // ── 渲染結果 ──────────────────────────────────────────────────
  const fmtCls = (v, proba) => {
    const label = (v === null || v === undefined || (typeof v === 'number' && isNaN(v))) ? '—' : String(v);
    const pct = proba != null ? ` (${(proba * 100).toFixed(1)}%)` : '';
    return label + pct;
  };

  // 回歸:顯示預測值 + ±1σ 信心區間 (從 5 fold std 來)
  if (predEl) {
    if (isReg && typeof predRespStd === 'number' && predRespStd > 0) {
      const main = formatRwifVal(pred);
      const stdStr = formatRwifVal(predRespStd);
      predEl.innerHTML = `${main}<span class="ml-2 text-[11px] text-dark-400">±${stdStr}</span>`;
    } else {
      predEl.textContent = isReg ? formatRwifVal(pred) : fmtCls(pred, predProba);
    }
  }
  document.getElementById('real-whatif-target-label').textContent = m.targetName || '目標變數';
  if (baseEl) baseEl.textContent = isReg ? formatRwifVal(baseline) : fmtCls(baseline, baseProba);
  document.getElementById('real-whatif-model-name').textContent = m.name;

  if (isReg && !isNaN(pred) && !isNaN(baseline)) {
    const delta = pred - baseline;
    const sign = delta >= 0 ? '+' : '';
    if (deltaEl) {
      // 多顯示一個信心區間提示 (66% / 95% CI)
      let ciHint = '';
      if (typeof predRespStd === 'number' && predRespStd > 0) {
        const lo = formatRwifVal(pred - 1.96 * predRespStd);
        const hi = formatRwifVal(pred + 1.96 * predRespStd);
        ciHint = ` <span class="text-[10px] text-dark-500 ml-1">95% CI: [${lo}, ${hi}]</span>`;
      }
      deltaEl.innerHTML = `${sign}${formatRwifVal(delta)}${ciHint}`;
      deltaEl.className = `font-mono ${delta > 0 ? 'text-success-400' : delta < 0 ? 'text-danger-400' : 'text-dark-300'}`;
    }
  } else if (!isReg && predProba != null && baseProba != null) {
    // 分類：顯示機率差值
    const delta = predProba - baseProba;
    const sign = delta >= 0 ? '+' : '';
    if (deltaEl) {
      deltaEl.textContent = `${sign}${(delta * 100).toFixed(1)}%`;
      deltaEl.className = `font-mono ${delta > 0 ? 'text-success-400' : delta < 0 ? 'text-danger-400' : 'text-dark-300'}`;
    }
  } else {
    if (deltaEl) deltaEl.textContent = '—';
  }
}

function formatRwifVal(v) {
  if (v === null || v === undefined || isNaN(v)) return '—';
  const abs = Math.abs(v);
  if (abs >= 1000) return v.toFixed(0);
  if (abs >= 100) return v.toFixed(1);
  if (abs >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

function renderRealFeatureImportance(model) {
  const chart = initChart('chart-real-feature-importance');
  if (!chart) return;

  const names = model.featureNames;
  const values = model.featureImportance;
  // 兩種格式都接受:
  //   A. 純 array,跟 featureNames 等長對齊 (sklearn 引擎)
  //   B. [[name, score], ...] 已配對 (daniel ensemble — pipeline 轉換後的特徵名跟 raw 不一樣)
  let pairs;
  if (Array.isArray(values) && values.length > 0 && Array.isArray(values[0])) {
    pairs = values.map(([n, v]) => ({ name: String(n), value: Number(v) || 0 }));
  } else if (names && values && names.length > 0) {
    pairs = names.map((n, i) => ({ name: n, value: values[i] }));
  } else {
    _chartPlaceholderMessage(chart, '此歷史紀錄的特徵重要性資料已被壓縮\n請重新訓練以查看完整圖表');
    return;
  }
  pairs = pairs.sort((a, b) => b.value - a.value).slice(0, 15);

  chart.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0', fontSize: 11 } },
    grid: { left: 120, right: 30, top: 10, bottom: 15 },
    xAxis: { type: 'value', axisLabel: { color: '#64748b', fontSize: 9 }, splitLine: { lineStyle: { color: '#1e293b' } } },
    yAxis: { type: 'category', data: pairs.map(p => p.name).reverse(), axisLabel: { color: '#e2e8f0', fontSize: 10, formatter: v => v.length > 18 ? v.slice(0, 16) + '...' : v }, axisLine: { show: false }, axisTick: { show: false } },
    series: [{
      type: 'bar', data: pairs.map(p => p.value).reverse(),
      itemStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
          { offset: 0, color: '#3b82f6' },
          { offset: 1, color: '#22d3ee' }
        ]),
        borderRadius: [0, 4, 4, 0],
      },
      barWidth: '55%',
      label: { show: true, position: 'right', color: '#94a3b8', fontSize: 9, formatter: p => (p.value * 100).toFixed(1) + '%' },
    }],
  });
}

function renderRealModelCompare(models, isReg) {
  const chart = initChart('chart-real-model-compare');
  if (!chart) return;

  const names = models.map(m => m.name);
  const scores = models.map(m => isReg ? m.metrics.testR2 : m.metrics.testScore);
  const colors = scores.map(s => s > 0.8 ? '#10b981' : s > 0.5 ? '#f59e0b' : s > 0 ? '#f87171' : '#64748b');

  chart.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0', fontSize: 11 } },
    grid: { left: 30, right: 20, top: 10, bottom: 80 },
    xAxis: { type: 'category', data: names, axisLabel: { color: '#94a3b8', fontSize: 9, rotate: 35, interval: 0 }, axisLine: { lineStyle: { color: '#1e293b' } } },
    yAxis: { type: 'value', name: isReg ? 'R²' : 'Score', nameTextStyle: { color: '#64748b', fontSize: 10 }, axisLabel: { color: '#64748b', fontSize: 9 }, splitLine: { lineStyle: { color: '#1e293b' } } },
    series: [{
      type: 'bar', data: scores.map((s, i) => ({ value: s, itemStyle: { color: colors[i] } })),
      barWidth: '50%',
      itemStyle: { borderRadius: [4, 4, 0, 0] },
      label: { show: true, position: 'top', color: '#e2e8f0', fontSize: 9, formatter: p => p.value.toFixed(4) },
    }],
  });
}

// 共用:在 chart 上印一段中央訊息 (data 不全時的 fallback)
function _chartPlaceholderMessage(chart, msg) {
  chart.setOption({
    title: {
      text: msg,
      left: 'center', top: 'center',
      textStyle: { color: '#64748b', fontSize: 13, lineHeight: 20 },
    },
  }, true);
}

function renderRealPredScatter(model, isReg) {
  const chart = initChart('chart-real-pred-scatter');
  if (!chart) return;

  // 防禦:從 localStorage 還原的歷史 run 可能因 quota 觸發瘦身,testTrue/testPred 被砍
  if (!model.testTrue || !model.testPred || model.testTrue.length === 0) {
    _chartPlaceholderMessage(chart, '此歷史紀錄的測試集資料已被壓縮\n請重新訓練以查看散點圖');
    return;
  }

  if (!isReg) {
    // Classification: show confusion-like accuracy per class
    chart.setOption({
      title: { text: '（分類模式 - 請參考排行榜）', left: 'center', top: 'center', textStyle: { color: '#64748b', fontSize: 13 } },
    });
    return;
  }

  const trueVals = model.testTrue;
  const predVals = model.testPred;
  const data = trueVals.map((t, i) => [t, predVals[i]]);
  const allVals = [...trueVals, ...predVals];
  const mn = Math.min(...allVals);
  const mx = Math.max(...allVals);

  chart.setOption({
    tooltip: { trigger: 'item', backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0', fontSize: 11 }, formatter: p => `實際: ${p.value[0].toFixed(2)}<br>預測: ${p.value[1].toFixed(2)}` },
    grid: { left: 50, right: 20, top: 15, bottom: 40 },
    xAxis: { type: 'value', name: '實際值', nameTextStyle: { color: '#64748b', fontSize: 10 }, axisLabel: { color: '#64748b', fontSize: 9 }, splitLine: { lineStyle: { color: '#1e293b' } } },
    yAxis: { type: 'value', name: '預測值', nameTextStyle: { color: '#64748b', fontSize: 10 }, axisLabel: { color: '#64748b', fontSize: 9 }, splitLine: { lineStyle: { color: '#1e293b' } } },
    series: [
      { type: 'scatter', data, symbolSize: 6, itemStyle: { color: '#22d3ee', opacity: 0.6 } },
      { type: 'line', data: [[mn, mn], [mx, mx]], lineStyle: { color: '#f59e0b', type: 'dashed', width: 1.5 }, symbol: 'none', tooltip: { show: false } },
    ],
  });
}

function renderRealResiduals(model) {
  const chart = initChart('chart-real-residuals');
  if (!chart) return;

  if (!model.testTrue || !model.testPred || model.testTrue.length === 0) {
    _chartPlaceholderMessage(chart, '此歷史紀錄的測試集資料已被壓縮\n請重新訓練以查看殘差分佈');
    return;
  }
  const residuals = model.testTrue.map((t, i) => t - model.testPred[i]);
  // Histogram
  const binCount = 20;
  const mn = Math.min(...residuals);
  const mx = Math.max(...residuals);
  const binWidth = (mx - mn) / binCount || 1;
  const bins = new Array(binCount).fill(0);
  const binLabels = [];
  for (let i = 0; i < binCount; i++) {
    binLabels.push((mn + (i + 0.5) * binWidth).toFixed(1));
  }
  residuals.forEach(r => {
    const idx = Math.min(Math.floor((r - mn) / binWidth), binCount - 1);
    bins[idx]++;
  });

  chart.setOption({
    tooltip: { trigger: 'axis', backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0', fontSize: 11 } },
    grid: { left: 40, right: 20, top: 15, bottom: 35 },
    xAxis: { type: 'category', data: binLabels, name: '殘差', nameTextStyle: { color: '#64748b', fontSize: 10 }, axisLabel: { color: '#64748b', fontSize: 8, rotate: 30 }, axisLine: { lineStyle: { color: '#1e293b' } } },
    yAxis: { type: 'value', name: '頻次', nameTextStyle: { color: '#64748b', fontSize: 10 }, axisLabel: { color: '#64748b', fontSize: 9 }, splitLine: { lineStyle: { color: '#1e293b' } } },
    series: [{
      type: 'bar', data: bins,
      itemStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: '#22d3ee' },
          { offset: 1, color: '#3b82f6' }
        ]),
        borderRadius: [3, 3, 0, 0],
      },
      barWidth: '80%',
    }],
  });
}

// ===== DASHBOARD REAL METRICS =====
function updateDashboardRealMetrics() {
  const models = MLEngine.trainedModels;
  const cards = document.querySelectorAll('#page-dashboard .metric-card');
  if (cards.length < 4) return;

  // Card 0: 活躍數據集 — DataEngine.datasets 總數
  cards[0].querySelector('.text-3xl').textContent = DataEngine.datasets.length.toString();
  const sub0 = cards[0].querySelector('.text-success-400, .text-dark-400.text-xs, .text-xs');
  if (sub0 && DataEngine.currentDataset) {
    const rows = DataEngine.currentDataset.data?.length || DataEngine.currentDataset.rowCount || 0;
    const rowStr = rows > 0 ? ` · ${rows.toLocaleString()} 列` : '';
    sub0.textContent = `當前: ${DataEngine.currentDataset.fileName || ''}${rowStr}`;
  }

  // Card 1: 已完成實驗 — 歷史訓練「次數」(每次 run = 1 個實驗)
  const totalRuns = _trainingHistory.length;
  cards[1].querySelector('.text-3xl').textContent = totalRuns.toString();
  const sub1 = cards[1].querySelectorAll('.text-success-400, .text-dark-400.text-xs, .text-xs');
  if (sub1.length && totalRuns > 0) {
    const totalModels = _trainingHistory.reduce((s, h) => s + (h.modelCount || 0), 0);
    sub1[sub1.length - 1].textContent = `共訓練 ${totalModels} 個模型`;
  }

  // Card 2: 最佳模型分數 — 用「歷史中最高分」而非「上次最佳」
  let best = models[0];
  let bestRun = null;
  for (const h of _trainingHistory) {
    if (!bestRun || (h.bestModel?.score ?? 0) > (bestRun.bestModel?.score ?? 0)) {
      bestRun = h;
    }
  }
  if (bestRun) {
    const isReg = bestRun.taskType === 'regression';
    const score = bestRun.bestModel.score;
    cards[2].querySelector('.text-3xl').innerHTML = isReg
      ? score.toFixed(3) + '<span class="text-lg text-dark-400"> R²</span>'
      : (score * 100).toFixed(1) + '<span class="text-lg text-dark-400">%</span>';
    cards[2].querySelector('.text-dark-400.text-sm').textContent = isReg ? '歷史最佳 R²' : '歷史最佳準確率';
  } else if (best) {
    const isReg = best.taskType === 'regression';
    cards[2].querySelector('.text-3xl').innerHTML = isReg
      ? best.metrics.testR2.toFixed(3) + '<span class="text-lg text-dark-400"> R²</span>'
      : (best.metrics.testAccuracy * 100).toFixed(1) + '<span class="text-lg text-dark-400">%</span>';
  }

  // Card 2 sub-line: 與上一次相比的進步幅度
  if (bestRun) {
    const sub2 = cards[2].querySelectorAll('.text-success-400, .text-dark-400.text-xs, .text-xs');
    const sub2El = sub2[sub2.length - 1];
    if (sub2El && _trainingHistory.length >= 2) {
      const prevRun = _trainingHistory[1];  // second most recent
      const isReg = bestRun.taskType === 'regression';
      const delta = (bestRun.bestModel.score ?? 0) - (prevRun.bestModel?.score ?? 0);
      if (Math.abs(delta) > 0.0001) {
        const sign = delta > 0 ? '+' : '';
        const deltaStr = isReg ? `${sign}${delta.toFixed(3)} R²` : `${sign}${(delta * 100).toFixed(1)}% Acc`;
        sub2El.className = `flex items-center gap-1 mt-1 text-xs ${delta > 0 ? 'text-success-400' : 'text-warning-400'}`;
        sub2El.innerHTML = delta > 0
          ? `<svg class="w-3 h-3"><use href="#i-arrow-up"/></svg>${deltaStr} 較上次`
          : `<svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" transform="rotate(180 12 12)"/></svg>${deltaStr} 較上次`;
      }
    }
    // Show AUC in sub-label if available and classification
    const bestMetrics = bestRun.bestModel?.metrics || bestRun.bestModel || {};
    if (!bestRun.taskType || bestRun.taskType === 'classification') {
      const auc = bestMetrics.auc;
      const lbl2 = cards[2].querySelector('.text-dark-400.text-sm');
      if (lbl2 && auc > 0) lbl2.textContent = `歷史最佳 AUC: ${auc.toFixed(3)}`;
    }
  }

  // Card 3: 已部署模型 → 改成「目前載入模型數」(更實用)
  const allHistoryModels = _trainingHistory.reduce((s, h) => s + (h.models?.length || h.modelCount || 0), 0);
  cards[3].querySelector('.text-3xl').textContent = (models.length || allHistoryModels).toString();
  const lbl3 = cards[3].querySelector('.text-dark-400.text-sm');
  if (lbl3) lbl3.textContent = models.length > 0 ? '當前訓練模型數' : `歷史合計 ${allHistoryModels} 個`;
  const sub3 = cards[3].querySelectorAll('.text-success-400, .text-dark-400.text-xs, .text-xs');
  const sub3El = sub3[sub3.length - 1];
  if (sub3El && models.length > 0) sub3El.textContent = `API 端點已就緒`;

  // 最近實驗清單 — 用 _trainingHistory 取代 mock data
  renderRecentExperimentsCard();
}

// 模型效能趨勢 (real mode) — 用 _trainingHistory,每個資料集一條線、X 軸是訓練時間、Y 軸是該次最佳分數
function renderRealPerformanceTrend() {
  const chart = initChart('chart-performance-trend');
  if (!chart) return;
  if (_trainingHistory.length === 0) {
    chart.setOption({ title: { text: '尚無訓練紀錄', textStyle: { color: '#94a3b8', fontSize: 14 }, left: 'center', top: 'center' } });
    return;
  }

  // 倒過來 — 時間軸是「舊 → 新」
  const runsOldFirst = [..._trainingHistory].reverse();

  // 依資料集分組
  const byDataset = new Map();
  runsOldFirst.forEach(h => {
    if (!byDataset.has(h.datasetName)) byDataset.set(h.datasetName, []);
    byDataset.get(h.datasetName).push(h);
  });

  // X 軸:用所有訓練的時間戳 (合併 + 排序)
  const xLabels = runsOldFirst.map(h => {
    const d = new Date(h.timestamp);
    return `${String(d.getMonth()+1).padStart(2,'0')}/${String(d.getDate()).padStart(2,'0')} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
  });

  const palette = ['#3b82f6', '#06b6d4', '#8b5cf6', '#f59e0b', '#10b981'];
  const series = [];
  let colorIdx = 0;
  byDataset.forEach((runs, datasetName) => {
    const color = palette[colorIdx++ % palette.length];
    // 對應到 xLabels 的 sparse data (沒跑該資料集的位置給 null)
    const data = runsOldFirst.map(h => h.datasetName === datasetName ? h.bestModel.score : null);
    series.push({
      name: datasetName,
      type: 'line',
      data,
      smooth: false,
      connectNulls: true,
      symbol: 'circle',
      symbolSize: 6,
      lineStyle: { width: 2, color },
      itemStyle: { color },
    });
  });

  chart.setOption({
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0', fontSize: 11 },
      formatter: (params) => {
        const idx = params[0].dataIndex;
        const run = runsOldFirst[idx];
        const isReg = run.taskType === 'regression';
        return `<b>${run.datasetName}</b><br>` +
               `target: ${run.target}<br>` +
               `${run.modelCount} 模型 · ${run.bestModel.name}<br>` +
               `${isReg ? 'R²' : 'Acc'}: ${isReg ? run.bestModel.score.toFixed(4) : (run.bestModel.score*100).toFixed(1)+'%'}`;
      },
    },
    legend: { data: [...byDataset.keys()], top: 5, right: 10, textStyle: { color: '#94a3b8', fontSize: 11 } },
    grid: { left: 50, right: 20, top: 40, bottom: 30 },
    xAxis: { type: 'category', data: xLabels, axisLine: { lineStyle: { color: '#1e293b' } }, axisLabel: { color: '#64748b', fontSize: 10 } },
    yAxis: { type: 'value', min: 0, max: 1, axisLine: { show: false }, axisLabel: { color: '#64748b', fontSize: 10, formatter: v => v.toFixed(2) }, splitLine: { lineStyle: { color: '#1e293b' } } },
    series,
  }, true);
}

// 任務類型分佈 (real mode) — 從歷史統計 classification vs regression
function renderRealTaskDistribution() {
  const chart = initChart('chart-task-distribution');
  if (!chart) return;
  if (_trainingHistory.length === 0) {
    chart.setOption({ title: { text: '尚無訓練紀錄', textStyle: { color: '#94a3b8', fontSize: 14 }, left: 'center', top: 'center' } });
    return;
  }

  const counts = { classification: 0, regression: 0 };
  _trainingHistory.forEach(h => {
    if (h.taskType === 'classification') counts.classification++;
    else if (h.taskType === 'regression') counts.regression++;
  });

  const data = [
    { value: counts.classification, name: '分類', itemStyle: { color: '#3b82f6' } },
    { value: counts.regression,     name: '迴歸', itemStyle: { color: '#06b6d4' } },
  ].filter(d => d.value > 0);

  chart.setOption({
    tooltip: { trigger: 'item', backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0' } },
    series: [{
      type: 'pie', radius: ['50%', '75%'], center: ['50%', '50%'],
      avoidLabelOverlap: true,
      itemStyle: { borderRadius: 6, borderColor: '#0f172a', borderWidth: 3 },
      label: { show: true, color: '#94a3b8', fontSize: 11, formatter: '{b}\n{c} 次 ({d}%)' },
      labelLine: { lineStyle: { color: '#334155' } },
      data,
    }],
  }, true);
}

// ============================================================
// ③ DEPLOYMENTS — REAL MODE
// 把所有訓練過的模型列為「可查詢的 API 端點」，每張卡顯示：
//   模型名稱、類型標籤、核心指標、訓練時間戳、curl 呼叫範例
// ============================================================
function renderRealDeployments() {
  const wrap = document.getElementById('deployments-real-content');
  if (!wrap) return;

  // Collect all models from current session + history
  const allModels = [];
  // Current session models
  (MLEngine.trainedModels || []).forEach(m => {
    if (m.id) allModels.push({ m, ts: Date.now(), fromSession: true });
  });
  // History models (de-dupe by id)
  const seenIds = new Set(allModels.map(e => e.m.id));
  _trainingHistory.forEach(h => {
    (h.models || []).forEach(m => {
      if (m.id && !seenIds.has(m.id)) {
        seenIds.add(m.id);
        allModels.push({ m, ts: h.timestamp, fromSession: false });
      }
    });
  });

  if (allModels.length === 0) {
    wrap.innerHTML = '';
    // Fall back to empty state
    const realEmptyEl = document.getElementById('deployments-real-empty');
    if (realEmptyEl) realEmptyEl.classList.remove('hidden');
    wrap.classList.add('hidden');
    return;
  }

  wrap.classList.remove('hidden');
  const isReg = m => m.taskType === 'regression';

  // Summary stats
  const total = allModels.length;
  const bestScore = allModels.reduce((best, { m }) => {
    const s = isReg(m) ? (m.metrics?.testR2 ?? 0) : (m.metrics?.testAccuracy ?? 0);
    return s > best ? s : best;
  }, 0);
  const apiBase = (typeof ApiClient !== 'undefined' && ApiClient.baseUrl) ? ApiClient.baseUrl : 'http://localhost:8000';

  const fmtScore = (m) => {
    if (isReg(m)) {
      const r2 = m.metrics?.testR2;
      return r2 != null ? `R² ${r2.toFixed(3)}` : '—';
    }
    const acc = m.metrics?.testAccuracy;
    const auc = m.metrics?.auc;
    if (auc > 0) return `AUC ${auc.toFixed(3)}`;
    return acc != null ? `Acc ${(acc * 100).toFixed(1)}%` : '—';
  };

  const fmtTs = (ts) => {
    if (!ts) return '—';
    const d = new Date(ts);
    return d.toLocaleDateString('zh-TW') + ' ' + d.toLocaleTimeString('zh-TW', { hour: '2-digit', minute: '2-digit' });
  };

  const curlSnippet = (modelId) =>
    `curl -X POST ${apiBase}/api/predict \\\n  -H "Content-Type: application/json" \\\n  -d '{"modelId":"${modelId}","features":[...]}'`;

  wrap.innerHTML = `
    <!-- Summary row -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
      <div class="metric-card">
        <div class="flex items-center justify-between mb-3">
          <span class="text-dark-400 text-sm">已訓練模型</span>
          <span class="badge badge-success">可查詢</span>
        </div>
        <div class="text-3xl font-bold">${total}</div>
        <div class="text-xs text-dark-400 mt-1">${allModels.filter(e => e.fromSession).length} 本次工作階段</div>
      </div>
      <div class="metric-card">
        <div class="flex items-center justify-between mb-3">
          <span class="text-dark-400 text-sm">最佳分數</span>
        </div>
        <div class="text-3xl font-bold font-mono">${bestScore.toFixed(3)}</div>
        <div class="text-xs text-dark-400 mt-1">所有模型中最高</div>
      </div>
      <div class="metric-card">
        <div class="flex items-center justify-between mb-3">
          <span class="text-dark-400 text-sm">API 端點</span>
        </div>
        <div class="text-sm font-mono text-primary-300 mt-1 truncate">${apiBase}/api/predict</div>
        <div class="text-xs text-dark-400 mt-1">POST · JSON · 需 Authorization</div>
      </div>
    </div>

    <!-- Model cards -->
    <div class="space-y-3">
      ${allModels.map(({ m, ts }) => {
        const taskBadge = isReg(m)
          ? `<span class="text-xs px-2 py-0.5 rounded-full bg-accent-500/15 text-accent-400">迴歸</span>`
          : `<span class="text-xs px-2 py-0.5 rounded-full bg-primary-500/15 text-primary-400">分類</span>`;
        const srcBadge = m.dataSource === 'preprocessed'
          ? `<span class="text-xs px-2 py-0.5 rounded-full bg-success-500/15 text-success-400">預處理</span>`
          : `<span class="text-xs px-2 py-0.5 rounded-full bg-dark-600 text-dark-300">原始</span>`;
        const snippet = escapeHtml(curlSnippet(m.id));
        return `
        <div class="card">
          <div class="flex items-start gap-4 p-4">
            <div class="w-10 h-10 rounded-xl bg-primary-500/10 flex items-center justify-center flex-shrink-0 mt-0.5">
              <div class="w-3 h-3 rounded-full bg-primary-400"></div>
            </div>
            <div class="flex-1 min-w-0">
              <div class="flex items-center gap-2 flex-wrap">
                <h4 class="font-semibold text-sm">${escapeHtml(m.name || m.type || '模型')}</h4>
                ${taskBadge}${srcBadge}
              </div>
              <p class="text-xs text-dark-400 mt-0.5">ID: <code class="text-[11px] bg-dark-800 px-1.5 py-0.5 rounded text-dark-200">${m.id}</code></p>
              <details class="mt-2">
                <summary class="text-xs text-dark-500 cursor-pointer hover:text-dark-300 select-none">curl 範例</summary>
                <pre class="mt-1.5 text-[10px] bg-dark-900 rounded p-2 overflow-x-auto text-dark-200 whitespace-pre-wrap">${snippet}</pre>
              </details>
            </div>
            <div class="text-right flex-shrink-0 ml-2">
              <p class="text-sm font-mono text-success-300">${fmtScore(m)}</p>
              <p class="text-xs text-dark-500 mt-0.5">${fmtTs(ts)}</p>
              <button onclick="analyzeModel('${m.id}')" class="mt-2 text-xs text-primary-400 hover:text-primary-300 transition-colors">洞察 →</button>
            </div>
          </div>
        </div>`;
      }).join('')}
    </div>

    <!-- Batch predict hint -->
    <div class="card mt-6 p-4 bg-dark-800/50">
      <p class="text-xs text-dark-400">
        <span class="text-primary-400 font-medium">批次預測：</span>
        上傳 CSV 檔案至
        <code class="text-[11px] bg-dark-800 px-1.5 py-0.5 rounded">${apiBase}/api/predict/batch</code>
        可一次取得整批預測結果。詳見實驗室 → 批次預測。
      </p>
    </div>
  `;
}

// 系統通知卡片 — 把 dashboard 上的「系統通知」改用真實 _notifications
function renderSystemNotificationsCard() {
  const cards = document.querySelectorAll('#page-dashboard .card');
  let targetCard = null;
  cards.forEach(c => {
    const title = c.querySelector('.card-title');
    if (title && title.textContent.trim() === '系統通知') targetCard = c;
  });
  if (!targetCard) return;
  const body = targetCard.querySelector('.space-y-3');
  if (!body) return;
  body.innerHTML = '';
  if (_notifications.length === 0) {
    body.innerHTML = '<p class="text-xs text-dark-500 py-6 text-center">暫無通知</p>';
    return;
  }
  const iconMap = {
    success: { svg: '<svg class="w-5 h-5 text-success-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>', wrap: 'bg-success-500/5 border border-success-500/10' },
    error:   { svg: '<svg class="w-5 h-5 text-danger-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>', wrap: 'bg-danger-500/5 border border-danger-500/10' },
    warning: { svg: '<svg class="w-5 h-5 text-warning-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>', wrap: 'bg-warning-500/5 border border-warning-500/10' },
    info:    { svg: '<svg class="w-5 h-5 text-primary-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>', wrap: 'bg-primary-500/5 border border-primary-500/10' },
  };
  _notifications.slice(0, 5).forEach(n => {
    const ic = iconMap[n.type] || iconMap.info;
    const div = document.createElement('div');
    div.className = `flex gap-3 p-3 rounded-lg ${ic.wrap}`;
    div.innerHTML = `${ic.svg}<div><p class="text-sm font-medium">${escapeHtml(n.title)}</p><p class="text-xs text-dark-400 mt-1">${escapeHtml(n.message)}</p></div>`;
    body.appendChild(div);
  });
}

// 全域:當前的最近實驗 filter (空字串 = 不篩選)
let _recentExpFilter = { datasetName: '', target: '' };

// Audit JSON 匯出 — 把 DataEngine.lastAuditReport 直接序列化下載,
// 供外部工具(pandas / R / 自家 dashboard)讀取分析。daniel run_data_audit 結構:
//   { health_score?, warnings: [...], info: [...], missing_summary: {...},
//     sentinel_summary: [...], schema_inferred: {...}, ... }
function exportAuditReportJson() {
  const audit = DataEngine.lastAuditReport;
  if (!audit || audit.error) {
    showToast('請先執行健檢', { type: 'warning' });
    return;
  }
  const ds = DataEngine.currentDataset;
  const dsName = ds?.fileName?.replace(/\.[^.]+$/, '') || 'dataset';
  const now = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);

  const payload = {
    exportedAt: new Date().toISOString(),
    dataset: { fileName: ds?.fileName, target: ds?.target || null,
               rowCount: ds?.rowCount, colCount: ds?.colCount },
    audit: audit,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)],
                        { type: 'application/json;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `audit_${dsName}_${now}.json`;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
  showToast('已下載 JSON 報告', { type: 'success' });
}


function exportAuditReportHtml() {
  const audit = DataEngine.lastAuditReport;
  if (!audit || audit.error) {
    showToast('請先執行健檢', { type: 'warning' });
    return;
  }
  const ds = DataEngine.currentDataset;
  const dsName = ds?.fileName || '資料集';
  const now = new Date().toLocaleString('zh-TW');

  // Build rows for missing summary
  const missingRows = Object.entries(audit.missing_summary || {}).map(([col, info]) => {
    const cnt = info.count || info.missing_count || '?';
    const pct = info.pct != null ? info.pct.toFixed(1) : (info.missing_pct != null ? info.missing_pct.toFixed(1) : '?');
    return `<tr><td>${escapeHtml(col)}</td><td style="text-align:right">${cnt}</td><td style="text-align:right">${pct}%</td></tr>`;
  }).join('');

  // Sentinel summary
  const sentinelRows = (audit.sentinel_summary || []).map(s => {
    const col = s.column || s.col || String(s);
    const vals = (s.sentinel_values || []).join(', ');
    const cnt = s.count != null ? s.count : '?';
    return `<tr><td>${escapeHtml(col)}</td><td>${escapeHtml(vals)}</td><td style="text-align:right">${cnt}</td></tr>`;
  }).join('');

  // Warnings list
  const warnItems = (audit.warnings || []).map(w => `<li>${escapeHtml(w)}</li>`).join('');

  const html = `<!DOCTYPE html>
<html lang="zh-TW"><head><meta charset="utf-8">
<title>資料健檢報告 — ${escapeHtml(dsName)}</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;background:#fff;color:#1e293b}
  h1{font-size:1.5rem;font-weight:700;margin-bottom:.25rem}
  .meta{color:#64748b;font-size:.875rem;margin-bottom:2rem}
  h2{font-size:1.1rem;font-weight:600;margin:1.5rem 0 .75rem;border-bottom:1px solid #e2e8f0;padding-bottom:.25rem}
  .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.5rem}
  .stat{border:1px solid #e2e8f0;border-radius:.5rem;padding:1rem}
  .stat .num{font-size:1.75rem;font-weight:700}
  .stat .lbl{color:#64748b;font-size:.8rem}
  table{width:100%;border-collapse:collapse;font-size:.875rem}
  th,td{text-align:left;padding:.5rem .75rem;border-bottom:1px solid #f1f5f9}
  th{background:#f8fafc;font-weight:600;color:#475569}
  .warn{background:#fffbeb;border-left:3px solid #f59e0b;padding:.5rem .75rem;margin:.25rem 0;border-radius:0 .25rem .25rem 0;font-size:.875rem}
  .ok{color:#10b981;font-weight:600}
  .footer{margin-top:3rem;padding-top:1rem;border-top:1px solid #e2e8f0;color:#94a3b8;font-size:.75rem}
</style></head><body>
<h1>資料健檢報告</h1>
<div class="meta">資料集: ${escapeHtml(dsName)} &nbsp;·&nbsp; 產生時間: ${now}</div>

<h2>統計摘要</h2>
<div class="stats">
  <div class="stat"><div class="num">${audit.total_rows ?? '—'}</div><div class="lbl">總列數</div></div>
  <div class="stat"><div class="num">${audit.total_columns ?? '—'}</div><div class="lbl">總欄位數</div></div>
  <div class="stat"><div class="num" style="color:#10b981">${audit.perfect_columns ?? '—'}</div><div class="lbl">完整欄位</div></div>
  <div class="stat"><div class="num" style="color:#f59e0b">${Object.keys(audit.missing_summary || {}).length}</div><div class="lbl">含缺失欄位</div></div>
</div>

<h2>缺失值摘要</h2>
${missingRows ? `<table><thead><tr><th>欄位</th><th style="text-align:right">缺失數</th><th style="text-align:right">缺失率</th></tr></thead><tbody>${missingRows}</tbody></table>` : '<p class="ok">✓ 無缺失值</p>'}

${sentinelRows ? `<h2>哨兵值偵測 (-9/-8/-7 等)</h2><table><thead><tr><th>欄位</th><th>哨兵值</th><th style="text-align:right">筆數</th></tr></thead><tbody>${sentinelRows}</tbody></table>` : ''}

<h2>警告事項</h2>
${warnItems ? `<ul style="list-style:none;padding:0">${(audit.warnings || []).map(w => `<div class="warn">⚠ ${escapeHtml(w)}</div>`).join('')}</ul>` : '<p class="ok">✓ 無警告</p>'}

<div class="footer">由 AutoML Studio 自動產生 · ${now}</div>
</body></html>`;

  const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `audit_report_${(dsName).replace(/[^a-zA-Z0-9]/g, '_').slice(0,30)}_${Date.now()}.html`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('健檢報告已匯出', { type: 'success' });
}

function renderRecentExperimentsCard() {
  const cards = document.querySelectorAll('#page-dashboard .card');
  let targetCard = null;
  cards.forEach(c => {
    const title = c.querySelector('.card-title');
    if (title && title.textContent.trim() === '最近實驗') targetCard = c;
  });
  if (!targetCard) return;
  const body = targetCard.querySelector('.space-y-3');
  if (!body) return;

  // 套用篩選 — 永遠依時間新→舊排序
  const filtered = _trainingHistory.filter(h => {
    if (_recentExpFilter.datasetName && h.datasetName !== _recentExpFilter.datasetName) return false;
    if (_recentExpFilter.target && h.target !== _recentExpFilter.target) return false;
    return true;
  });

  // 同步更新兩個下拉選單的選項 (datasets / targets) — 從整個歷史抓 unique
  syncRecentExperimentFilters();

  body.innerHTML = '';
  if (filtered.length === 0) {
    const emptyMsg = _trainingHistory.length === 0
      ? '尚未有訓練紀錄'
      : '篩選條件下沒有結果';
    body.innerHTML = `<p class="text-xs text-dark-500 py-6 text-center">${emptyMsg}</p>`;
    return;
  }

  filtered.slice(0, 5).forEach((h, i) => {
    const elapsedMin = Math.floor((Date.now() - h.timestamp) / 60000);
    const elapsedStr = elapsedMin < 1 ? '剛剛' : elapsedMin < 60 ? `${elapsedMin} 分鐘前` : `${Math.floor(elapsedMin/60)} 小時前`;
    const isReg = h.taskType === 'regression';
    const scoreStr = isReg
      ? `R²=${h.bestModel.score.toFixed(4)}`
      : `Acc=${(h.bestModel.score*100).toFixed(1)}%`;
    const dotColor = i === 0 ? 'bg-primary-500 animate-pulse' : 'bg-success-500';
    const item = document.createElement('div');
    item.className = 'flex items-center gap-3 p-3 rounded-lg bg-dark-800/50 hover:bg-dark-800 transition-colors cursor-pointer';
    item.innerHTML = `
      <div class="w-2 h-2 rounded-full ${dotColor}"></div>
      <div class="flex-1 min-w-0">
        <p class="text-sm font-medium truncate">${escapeHtml(h.datasetName)} <span class="text-dark-500 text-xs">→ ${escapeHtml(h.target)}</span></p>
        <p class="text-xs text-dark-400">${isReg ? '迴歸' : '分類'} | ${escapeHtml(h.bestModel.name)} | ${scoreStr}</p>
      </div>
      <span class="text-xs text-dark-400 shrink-0">${elapsedStr}</span>
    `;
    item.addEventListener('click', () => {
      applyHistoricalRun(h.id);
      navigateTo('leaderboard');
    });
    body.appendChild(item);
  });
}

// 重新填兩個 filter 下拉 (datasets / targets),保留目前選的值
function syncRecentExperimentFilters() {
  const dsSel = document.getElementById('recent-exp-filter-ds');
  const tgtSel = document.getElementById('recent-exp-filter-target');
  if (!dsSel || !tgtSel) return;

  // 一次性 wire listener
  if (!dsSel.__wired) {
    dsSel.__wired = true;
    dsSel.addEventListener('change', () => {
      _recentExpFilter.datasetName = dsSel.value;
      renderRecentExperimentsCard();
    });
    tgtSel.addEventListener('change', () => {
      _recentExpFilter.target = tgtSel.value;
      renderRecentExperimentsCard();
    });
  }

  // 從目前歷史抓 unique datasets / targets
  const allDatasets = [...new Set(_trainingHistory.map(h => h.datasetName).filter(Boolean))];
  // target 列表跟隨 dataset 篩選 — 選了 dataset 後 target 只列那個 dataset 出現過的
  const targetPool = _recentExpFilter.datasetName
    ? _trainingHistory.filter(h => h.datasetName === _recentExpFilter.datasetName)
    : _trainingHistory;
  const allTargets = [...new Set(targetPool.map(h => h.target).filter(Boolean))];

  // 重建 options 但保留 selected value
  const rebuildOptions = (sel, items, allLabel) => {
    const cur = sel.value;
    sel.innerHTML = `<option value="">${allLabel}</option>`;
    items.forEach(v => {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = v;
      sel.appendChild(opt);
    });
    // 還原 selection (如果 value 還在新列表中)
    if (cur && items.includes(cur)) sel.value = cur;
    else { sel.value = ''; }
  };
  rebuildOptions(dsSel, allDatasets, '所有資料集');
  rebuildOptions(tgtSel, allTargets, '所有 target');

  // 同步 internal state 跟 DOM (處理「dataset 變了之後 target 不在新列表中」的情況)
  if (dsSel.value !== _recentExpFilter.datasetName) _recentExpFilter.datasetName = dsSel.value;
  if (tgtSel.value !== _recentExpFilter.target)    _recentExpFilter.target = tgtSel.value;
}

// ===== SYSTEM SETTINGS =====
// ===== SETTINGS — 全域存取 =====
// 預設值 (localStorage 沒有時用這套)
const DEFAULT_SETTINGS = {
  testSize: 0.2,
  seed: 42,
  taskType: 'auto',
  srcRaw: true,
  srcPp: false,
  // 預設啟用的演算法 key (對應 MLEngine.ALGORITHMS) — 預設全開
  activeAlgos: ['linear_regression', 'ridge', 'lasso', 'elastic_net',
                'knn_3', 'knn_5', 'knn_7',
                'decision_tree', 'random_forest', 'gradient_boosting',
                'hist_gradient_boosting', 'xgboost', 'lightgbm', 'catboost',
                'naive_bayes', 'logistic', 'svr', 'svc',
                'voting', 'stacking'],
  shapSamples: 50,
};

// 其他頁面要讀設定就呼叫這個 — 一律回傳完整物件 (缺的欄位用預設補)
function getSettings() {
  let saved = {};
  try {
    saved = JSON.parse(localStorage.getItem('automl_settings') || '{}');
  } catch (e) { saved = {}; }
  return { ...DEFAULT_SETTINGS, ...saved };
}

function initSettings() {
  const testSizeRange = document.getElementById('setting-test-size');
  const testSizeLabel = document.getElementById('setting-test-size-label');
  const seedInput = document.getElementById('setting-seed');
  const taskTypeSel = document.getElementById('setting-task-type');
  const srcRawCb = document.getElementById('setting-src-raw');
  const srcPpCb = document.getElementById('setting-src-pp');
  const shapSamplesInput = document.getElementById('setting-shap-samples');
  const algoContainer = document.getElementById('setting-algos-container');
  const saveBtn = document.getElementById('btn-save-settings');
  const resetBtn = document.getElementById('btn-reset-settings');

  // ---- API 區 (本來就有效,保留) ----
  const apiToggle = document.getElementById('setting-use-api');
  const apiUrlInput = document.getElementById('setting-api-url');
  const apiUrlPreset = document.getElementById('setting-api-url-preset');
  const apiPingBtn = document.getElementById('btn-api-ping');
  const apiStatusEl = document.getElementById('setting-api-status');
  if (apiToggle && typeof ApiClient !== 'undefined') {
    apiToggle.checked = ApiClient.enabled;
    apiToggle.addEventListener('change', () => ApiClient.setEnabled(apiToggle.checked));

    // ---- 預設清單 + 自訂位址 (localStorage 持久化) ----
    const CUSTOM_KEY = 'apiBaseUrlCustomList';
    const loadCustom = () => {
      try { return JSON.parse(localStorage.getItem(CUSTOM_KEY) || '[]').filter(Boolean); }
      catch { return []; }
    };
    const saveCustom = (arr) => {
      try { localStorage.setItem(CUSTOM_KEY, JSON.stringify(arr)); } catch (e) {}
    };

    const rebuildPresetOptions = () => {
      if (!apiUrlPreset) return;
      // 預設三個選項保留(它們已在 HTML 裡),這裡只重新插入自訂條目
      // 先清掉 dataset='custom' 的舊條目,避免重複
      apiUrlPreset.querySelectorAll('option[data-custom="1"]').forEach(o => o.remove());
      const customs = loadCustom();
      const addCustomOpt = apiUrlPreset.querySelector('option[value="__custom__"]');
      customs.forEach(url => {
        const opt = document.createElement('option');
        opt.value = url;
        opt.dataset.custom = '1';
        opt.textContent = url.replace(/^https?:\/\//, '');
        apiUrlPreset.insertBefore(opt, addCustomOpt);
      });
    };

    // ---- 本地模式開關 (一鍵在「本機 localhost」與「雲端 HF」之間切) ----
    const localModeToggle = document.getElementById('setting-local-mode');
    const localHint = document.getElementById('setting-local-hint');
    // 雲端預設 = 預設選單第一個 (HuggingFace);抓不到就 fallback 寫死
    const HF_DEFAULT_URL = (apiUrlPreset.querySelector('option') || {}).value
      || 'https://honglideng-i2ai-automl-backend.hf.space';
    const isLocalUrl = (u) => /localhost|127\.0\.0\.1|\[::1\]/i.test(u || '');
    const syncLocalUi = (url) => {
      if (localModeToggle) localModeToggle.checked = isLocalUrl(url);
      if (localHint) localHint.classList.toggle('hidden', !isLocalUrl(url));
    };

    const setActiveUrl = (url) => {
      if (!url) return;
      apiUrlInput.value = url;
      ApiClient.setBaseUrl(url);
      // 同步 select:若 url 在 options 裡就選它,否則選自訂佔位
      const matched = Array.from(apiUrlPreset.options).find(o => o.value === url);
      apiUrlPreset.value = matched ? url : '__custom__';
      syncLocalUi(url);  // 也同步「本地模式」開關 + 提示
    };

    if (localModeToggle) {
      localModeToggle.addEventListener('change', () => {
        if (localModeToggle.checked) {
          // 開本地模式:確保 API 有啟用,並指向本機後端
          if (!ApiClient.enabled) { ApiClient.setEnabled(true); if (apiToggle) apiToggle.checked = true; }
          // 用 127.0.0.1 而非 localhost — Windows 上 localhost 常解析到 IPv6 ::1,
          // 但 uvicorn 預設只聽 IPv4 → 會 Failed to fetch。127.0.0.1 強制走 IPv4。
          setActiveUrl('http://127.0.0.1:8000');
        } else {
          setActiveUrl(HF_DEFAULT_URL);   // 關掉 → 回雲端
        }
      });
    }

    rebuildPresetOptions();
    setActiveUrl(ApiClient.baseUrl);

    apiUrlPreset.addEventListener('change', () => {
      const v = apiUrlPreset.value;
      if (v === '__custom__') {
        const newUrl = prompt('輸入新的 API Base URL (例如 https://my-backend.example.com):');
        if (!newUrl || !newUrl.trim()) {
          // 取消 → 回到目前 active 的 URL
          setActiveUrl(ApiClient.baseUrl);
          return;
        }
        const clean = newUrl.trim().replace(/\/$/, '');
        const customs = loadCustom();
        if (!customs.includes(clean)) {
          customs.push(clean);
          saveCustom(customs);
          rebuildPresetOptions();
        }
        setActiveUrl(clean);
      } else {
        setActiveUrl(v);
      }
    });

    apiUrlInput.addEventListener('change', () => {
      const v = apiUrlInput.value.trim().replace(/\/$/, '');
      if (!v) return;
      // 手動編輯 input → 自動加進自訂清單 (若不是預設項)
      const isBuiltin = Array.from(apiUrlPreset.options)
        .filter(o => !o.dataset.custom && o.value !== '__custom__')
        .some(o => o.value === v);
      if (!isBuiltin) {
        const customs = loadCustom();
        if (!customs.includes(v)) {
          customs.push(v);
          saveCustom(customs);
          rebuildPresetOptions();
        }
      }
      setActiveUrl(v);
    });

    apiPingBtn.addEventListener('click', async () => {
      apiStatusEl.textContent = '測試中...';
      const ok = await ApiClient.health();
      apiStatusEl.textContent = ok ? `✓ 已連線 (${ApiClient.baseUrl})` : `✗ 無法連線到 ${ApiClient.baseUrl}`;
      apiStatusEl.className = ok ? 'text-xs text-success-400' : 'text-xs text-danger-400';
    });
  }

  // ---- 演算法 chips — 依 MLEngine.ALGORITHMS 動態產生 ----
  let activeAlgos = getSettings().activeAlgos.slice();

  const renderAlgoChips = () => {
    if (!algoContainer || typeof MLEngine === 'undefined') return;
    algoContainer.innerHTML = '';
    Object.entries(MLEngine.ALGORITHMS).forEach(([key, algo]) => {
      const on = activeAlgos.includes(key);
      const chip = document.createElement('span');
      chip.className = on
        ? 'setting-algo-chip text-xs bg-primary-500/15 text-primary-400 border border-primary-500/30 px-2.5 py-1 rounded-lg cursor-pointer select-none transition-colors'
        : 'setting-algo-chip text-xs bg-dark-700 text-dark-400 border border-dark-600 px-2.5 py-1 rounded-lg cursor-pointer select-none transition-colors';
      chip.dataset.val = key;
      chip.textContent = algo.label || algo.name || key;
      algoContainer.appendChild(chip);
    });
  };

  if (algoContainer) {
    algoContainer.addEventListener('click', (e) => {
      const chip = e.target.closest('.setting-algo-chip');
      if (!chip) return;
      const val = chip.dataset.val;
      if (activeAlgos.includes(val)) activeAlgos = activeAlgos.filter(v => v !== val);
      else activeAlgos.push(val);
      renderAlgoChips();
    });
  }

  // ---- 載入已存設定到 UI ----
  const applyToUI = (s) => {
    if (testSizeRange) {
      testSizeRange.value = s.testSize;
      if (testSizeLabel) testSizeLabel.textContent = `${Math.round(s.testSize * 100)}%`;
    }
    if (seedInput) seedInput.value = s.seed;
    if (taskTypeSel) taskTypeSel.value = s.taskType;
    if (srcRawCb) srcRawCb.checked = s.srcRaw;
    if (srcPpCb) srcPpCb.checked = s.srcPp;
    if (shapSamplesInput) shapSamplesInput.value = s.shapSamples;
    activeAlgos = s.activeAlgos.slice();
    renderAlgoChips();
  };
  applyToUI(getSettings());

  // test-size slider 即時更新標籤
  if (testSizeRange && testSizeLabel) {
    testSizeRange.addEventListener('input', () => {
      testSizeLabel.textContent = `${Math.round(testSizeRange.value * 100)}%`;
    });
  }

  // ---- 儲存 ----
  if (saveBtn) {
    saveBtn.addEventListener('click', () => {
      const cfg = {
        testSize: testSizeRange ? parseFloat(testSizeRange.value) : 0.2,
        seed: seedInput ? parseInt(seedInput.value) || 42 : 42,
        taskType: taskTypeSel ? taskTypeSel.value : 'auto',
        srcRaw: srcRawCb ? srcRawCb.checked : true,
        srcPp: srcPpCb ? srcPpCb.checked : false,
        activeAlgos: activeAlgos.slice(),
        shapSamples: shapSamplesInput ? (parseInt(shapSamplesInput.value) || 50) : 50,
      };
      localStorage.setItem('automl_settings', JSON.stringify(cfg));
      showToast('設定已儲存', { type: 'success' });
    });
  }

  // ---- 重置 ----
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      if (!confirm('確定要清除所有已儲存的設定,回到預設值嗎?')) return;
      localStorage.removeItem('automl_settings');
      applyToUI({ ...DEFAULT_SETTINGS });
      showToast('已重置為預設值', { type: 'info' });
    });
  }

}

// ============================================================
// PIPELINE PAGE (測試模式專用) — 跑 Daniel 的 AutoML pipeline
// ============================================================

// 9 個階段的卡片資料 — 對應 daniel_runner.py 的 _STAGE_PATTERNS
const PIPELINE_STAGES = [
  { id: 'scout',     label: 'Scout HPO',           desc: '單次 holdout 快篩弱模型',          pct: 5,  color: 'primary' },
  { id: 'full-hpo',  label: 'Full HPO',            desc: 'Optuna TPE × 5-Fold CV',          pct: 15, color: 'primary' },
  { id: 'nas',       label: 'MLP / TSNet NAS',     desc: 'OneShot 共享權重超網路',          pct: 35, color: 'accent'  },
  { id: 'mlp-hpo',   label: 'MLP Training HPO',    desc: 'lr / dropout / wd 搜尋',          pct: 50, color: 'accent'  },
  { id: 'cnn-hpo',   label: 'CNN1D / TCN HPO',     desc: '時序模式跑 TCN (擴張因果卷積)',   pct: 65, color: 'accent'  },
  { id: 'tx-hpo',    label: 'Transformer HPO',     desc: '時序模式跑 PatchTST',             pct: 78, color: 'accent'  },
  { id: '5fold',     label: '5-Fold CV',           desc: 'OOF + Test 預測 (含快取)',        pct: 88, color: 'warning' },
  { id: 'blend',     label: 'Nelder-Mead Blend',   desc: 'log-space softmax 權重最佳化',    pct: 93, color: 'warning' },
  { id: 'stack',     label: 'Meta-Learner Stack',  desc: 'OOF 拼接 → LGBM/XGB meta',        pct: 97, color: 'success' },
];

let _pipelineState = {
  file: null,
  running: false,
  reader: null,         // ReadableStreamDefaultReader,給取消用
  abortController: null,
};

function renderPipelinePage() {
  renderPipelineStages();
  loadPipelineBenchmark();
}

function renderPipelineStages() {
  const wrap = document.getElementById('pipeline-stages');
  if (!wrap) return;
  // 已渲染過就不重複渲染 (狀態 class 由 progress 事件動態切)
  if (wrap.children.length) return;
  wrap.innerHTML = PIPELINE_STAGES.map((s, i) => `
    <div class="pipeline-stage-card" data-stage="${s.id}" data-color="${s.color}">
      <div class="pipeline-stage-num">${i + 1}</div>
      <p class="pipeline-stage-label">${s.label}</p>
      <p class="pipeline-stage-desc">${s.desc}</p>
    </div>
  `).join('');
}

function updateStageProgress(pct) {
  // 依目前進度標亮對應 stage
  document.querySelectorAll('.pipeline-stage-card').forEach(card => {
    const stageId = card.dataset.stage;
    const stage = PIPELINE_STAGES.find(s => s.id === stageId);
    if (!stage) return;
    card.classList.remove('pipeline-stage-active', 'pipeline-stage-done');
    if (pct >= stage.pct + 8) card.classList.add('pipeline-stage-done');
    else if (pct >= stage.pct - 2) card.classList.add('pipeline-stage-active');
  });
}

async function loadPipelineBenchmark() {
  const tbody = document.getElementById('pipeline-bench-tbody');
  const meta = document.getElementById('pipeline-bench-meta');
  if (!tbody) return;
  if (typeof ApiClient === 'undefined' || !ApiClient.enabled) {
    tbody.innerHTML = `<tr><td colspan="8" class="text-center py-6 text-xs text-dark-500">請先在系統設定開啟「使用 Python 後端 API」</td></tr>`;
    if (meta) meta.textContent = '需要後端 API';
    return;
  }
  tbody.innerHTML = `<tr><td colspan="8" class="text-center py-6 text-xs text-dark-500">載入中...</td></tr>`;
  try {
    const resp = await fetch(`${ApiClient.baseUrl}/api/train/pipeline/benchmark`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    renderBenchmarkRows(data.rows || []);
    if (meta) meta.textContent = `${data.count || 0} 筆評估 (OpenML-CC18 + UCR 80)`;
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="8" class="text-center py-6 text-xs text-danger-400">載入失敗: ${escapeHtml(e.message)}</td></tr>`;
    if (meta) meta.textContent = '載入失敗';
  }
}

function renderBenchmarkRows(rows) {
  const tbody = document.getElementById('pipeline-bench-tbody');
  if (!tbody) return;
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="text-center py-6 text-xs text-dark-500">尚無評估結果</td></tr>`;
    return;
  }
  // 依 dataset 配對 pipeline / baseline
  const byDataset = {};
  rows.forEach(r => {
    const key = r.dataset;
    byDataset[key] = byDataset[key] || { dataset: key, type: r.type };
    byDataset[key][r.source] = r;
  });
  const html = Object.values(byDataset).map(grp => {
    const p = grp.pipeline || {};
    const b = grp.baseline || {};
    const pF1 = p.f1_macro ?? null;
    const bF1 = b.f1_macro ?? null;
    const delta = (pF1 != null && bF1 != null) ? (pF1 - bF1) : null;
    const dColor = delta == null ? 'text-dark-500'
                  : delta > 0  ? 'text-success-400'
                  :              'text-danger-400';
    const dSign = delta == null ? '—' : (delta > 0 ? '+' : '') + delta.toFixed(4);
    const typeColor = grp.type === 'TS' ? 'bg-warning-500/10 text-warning-300 border-warning-500/30'
                                        : 'bg-primary-500/10 text-primary-300 border-primary-500/30';
    return `
      <tr class="border-b border-dark-700/30 hover:bg-dark-800/30">
        <td class="py-2 px-4 font-mono text-xs">${escapeHtml(grp.dataset)}</td>
        <td class="py-2 px-3"><span class="text-[10px] px-1.5 py-0.5 rounded border ${typeColor}">${grp.type || '—'}</span></td>
        <td class="py-2 px-3 text-right text-xs text-dark-300">${p.n_train ?? '—'}</td>
        <td class="py-2 px-3 text-right text-xs text-dark-300">${p.n_test ?? '—'}</td>
        <td class="py-2 px-3 text-right font-mono text-warning-300">${pF1 != null ? pF1.toFixed(4) : '—'}</td>
        <td class="py-2 px-3 text-right font-mono text-dark-300">${bF1 != null ? bF1.toFixed(4) : '—'}</td>
        <td class="py-2 px-3 text-right font-mono ${dColor}">${dSign}</td>
        <td class="py-2 px-3 text-right text-xs text-dark-400">${p.elapsed_s != null ? p.elapsed_s.toFixed(1) : '—'}</td>
      </tr>
    `;
  }).join('');
  tbody.innerHTML = html;
}

function initPipelinePage() {
  // File picker
  const zone = document.getElementById('pipeline-upload-zone');
  const input = document.getElementById('pipeline-csv-input');
  const label = document.getElementById('pipeline-file-label');
  const meta = document.getElementById('pipeline-file-meta');
  if (!zone || !input) return;

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f) setPipelineFile(f);
  });
  input.addEventListener('change', () => {
    if (input.files.length > 0) setPipelineFile(input.files[0]);
  });

  function setPipelineFile(f) {
    _pipelineState.file = f;
    if (label) label.textContent = f.name;
    if (meta) meta.textContent = `${(f.size / 1024).toFixed(1)} KB · 點擊「執行 Pipeline」開始`;
  }

  // Run button
  document.getElementById('btn-pipeline-run')?.addEventListener('click', runPipelineFlow);
  document.getElementById('btn-pipeline-bench-reload')?.addEventListener('click', loadPipelineBenchmark);
}

async function runPipelineFlow() {
  if (_pipelineState.running) {
    showToast('Pipeline 正在執行中', { type: 'warning' });
    return;
  }
  const file = _pipelineState.file;
  if (!file) {
    showToast('請先選擇 CSV 檔案', { type: 'warning' });
    return;
  }
  if (typeof ApiClient === 'undefined' || !ApiClient.enabled) {
    showToast('請先在系統設定開啟 Python 後端 API', { type: 'error' });
    return;
  }

  // Reset UI
  const runtime = document.getElementById('pipeline-runtime');
  const logEl = document.getElementById('pipeline-log');
  const bar = document.getElementById('pipeline-progress-bar');
  const stepEl = document.getElementById('pipeline-progress-step');
  const statusEl = document.getElementById('pipeline-status');
  const runBtn = document.getElementById('btn-pipeline-run');
  runtime?.classList.remove('hidden');
  if (logEl) logEl.innerHTML = '';
  if (bar) bar.style.width = '0%';
  if (stepEl) stepEl.textContent = '上傳中...';
  if (statusEl) statusEl.textContent = '執行中';
  if (runBtn) { runBtn.disabled = true; runBtn.textContent = '執行中...'; }
  updateStageProgress(0);
  _pipelineState.running = true;

  const form = new FormData();
  form.append('file', file);
  const target = document.getElementById('pipeline-target')?.value.trim();
  if (target) form.append('target', target);
  form.append('timeSeries', document.getElementById('pipeline-ts')?.checked ? 'true' : 'false');
  form.append('metric', document.getElementById('pipeline-metric')?.value || 'f1');
  form.append('fast', document.getElementById('pipeline-fast')?.checked ? 'true' : 'false');
  form.append('timeLimit', document.getElementById('pipeline-time-limit')?.value || '0');
  form.append('skipDl', document.getElementById('pipeline-skip-dl')?.checked ? 'true' : 'false');
  form.append('noNas', document.getElementById('pipeline-no-nas')?.checked ? 'true' : 'false');

  try {
    const ctrl = new AbortController();
    _pipelineState.abortController = ctrl;
    const headers = {};
    if (typeof AuthClient !== 'undefined' && AuthClient.token) {
      headers['Authorization'] = `Bearer ${AuthClient.token}`;
    }
    const resp = await fetch(`${ApiClient.baseUrl}/api/train/pipeline/stream`, {
      method: 'POST', body: form, headers, signal: ctrl.signal,
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status} ${resp.statusText}`);

    // SSE parse
    const reader = resp.body.getReader();
    _pipelineState.reader = reader;
    const decoder = new TextDecoder('utf-8');
    let buf = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      // 每段以 "\n\n" 分隔
      let idx;
      while ((idx = buf.indexOf('\n\n')) !== -1) {
        const chunk = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        for (const line of chunk.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          let ev;
          try { ev = JSON.parse(line.slice(6)); } catch { continue; }
          handlePipelineEvent(ev);
        }
      }
    }
  } catch (e) {
    appendPipelineLog(`連線中斷: ${e.message}`, 'error');
    showToast('Pipeline 中斷', { type: 'error', msg: e.message });
  } finally {
    _pipelineState.running = false;
    _pipelineState.abortController = null;
    _pipelineState.reader = null;
    if (runBtn) { runBtn.disabled = false; runBtn.innerHTML = '<svg class="w-4 h-4"><use href="#i-arrow-right"/></svg>執行 Pipeline'; }
    if (statusEl) statusEl.textContent = '完成';
  }
}

function handlePipelineEvent(ev) {
  if (ev.type === 'log') {
    appendPipelineLog(ev.msg, ev.level || 'info');
  } else if (ev.type === 'progress') {
    const bar = document.getElementById('pipeline-progress-bar');
    const step = document.getElementById('pipeline-progress-step');
    if (bar) bar.style.width = `${ev.pct}%`;
    if (step) step.textContent = `${ev.step} (${ev.pct}%)`;
    updateStageProgress(ev.pct);
  } else if (ev.type === 'done') {
    appendPipelineLog('Pipeline 完成', 'success');
    const results = ev.results || (ev.result ? [ev.result] : []);
    const r = results[0] || {};
    renderPipelineResult(r);
    showToast('Pipeline 完成', { type: 'success', msg: `bestScore=${r.bestScore ?? '—'}` });
  } else if (ev.type === 'error') {
    appendPipelineLog(`錯誤: ${ev.message}`, 'error');
    showToast('Pipeline 失敗', { type: 'error', msg: ev.message });
  }
}

function appendPipelineLog(msg, level = 'info') {
  const logEl = document.getElementById('pipeline-log');
  if (!logEl) return;
  const colorMap = {
    info:    'text-dark-300',
    muted:   'text-dark-500',
    success: 'text-success-300',
    warning: 'text-warning-300',
    error:   'text-danger-300',
  };
  const p = document.createElement('p');
  p.className = colorMap[level] || colorMap.info;
  p.textContent = msg;
  logEl.appendChild(p);
  logEl.scrollTop = logEl.scrollHeight;
  if (logEl.children.length > 500) {
    while (logEl.children.length > 400) logEl.removeChild(logEl.firstChild);
  }
}

function renderPipelineResult(r) {
  const panel = document.getElementById('pipeline-result-panel');
  const metricEl = document.getElementById('pipeline-result-metric');
  if (!panel) return;
  if (!r.ok) {
    panel.innerHTML = `
      <div class="text-center py-6">
        <p class="text-xs text-danger-400 uppercase tracking-wider mb-2">執行失敗</p>
        <p class="text-sm text-dark-300">${escapeHtml(r.error || '未知錯誤')}</p>
      </div>`;
    if (metricEl) metricEl.textContent = '失敗';
    return;
  }
  if (metricEl) metricEl.textContent = r.metric || '—';
  const perModel = (r.perModel || []).filter(m => m.oofScore != null)
    .sort((a, b) => (b.oofScore || 0) - (a.oofScore || 0));
  panel.innerHTML = `
    <div class="text-center">
      <p class="text-[10px] text-dark-500 uppercase tracking-wider mb-1">Best (${escapeHtml(r.metric || 'score')})</p>
      <p class="text-5xl font-bold text-warning-300 mb-2 leading-none">${(r.bestScore ?? 0).toFixed(4)}</p>
      <div class="flex items-center justify-center gap-3 text-[11px] mt-3">
        <span class="text-dark-400">Blend</span><span class="font-mono text-dark-200">${(r.scoreBlend ?? 0).toFixed(4)}</span>
        <span class="text-dark-600">|</span>
        <span class="text-dark-400">Stack</span><span class="font-mono text-dark-200">${(r.scoreStack ?? 0).toFixed(4)}</span>
      </div>
    </div>
    <div class="grid grid-cols-3 gap-2 text-center pt-3 border-t border-dark-700/50">
      <div>
        <p class="text-[10px] text-dark-500">Accuracy</p>
        <p class="text-sm font-mono text-dark-100">${(r.accuracy ?? 0).toFixed(4)}</p>
      </div>
      <div>
        <p class="text-[10px] text-dark-500">F1</p>
        <p class="text-sm font-mono text-dark-100">${(r.f1 ?? 0).toFixed(4)}</p>
      </div>
      <div>
        <p class="text-[10px] text-dark-500">耗時</p>
        <p class="text-sm font-mono text-dark-100">${(r.elapsedSec ?? 0).toFixed(0)}s</p>
      </div>
    </div>
    <div class="text-[10px] text-dark-500 space-y-0.5 pt-3 border-t border-dark-700/50">
      <p>target: <code class="text-dark-300">${escapeHtml(r.target || '—')}</code></p>
      <p>split: ${escapeHtml(r.splitMode || '—')} · ${r.nTrain}/${r.nTest} · ${r.nClasses} class · ${r.nFeatures} features</p>
      <p>device: ${escapeHtml(r.device || '—')}</p>
    </div>
    ${perModel.length ? `
      <details class="pt-3 border-t border-dark-700/50">
        <summary class="text-[11px] text-dark-400 cursor-pointer hover:text-dark-200">各模型 OOF 分數 (${perModel.length})</summary>
        <div class="mt-2 space-y-1 max-h-40 overflow-y-auto">
          ${perModel.map(m => `
            <div class="flex items-center justify-between text-[11px]">
              <span class="font-mono text-dark-300 truncate" title="${escapeHtml(m.tag)}">${escapeHtml(m.tag)}</span>
              <span class="font-mono text-warning-300 ml-2 shrink-0">${(m.oofScore ?? 0).toFixed(4)}</span>
            </div>
          `).join('')}
        </div>
      </details>
    ` : ''}
  `;
}


// ===== 新介面 handshake =====
function initNewUiAuthHandshake() {
  const url = new URL(window.location.href);
  const need = url.searchParams.get('need_login');
  const next = url.searchParams.get('next');
  if (!need) return;

  const bounce = () => {
    const nextUrl = next ? decodeURIComponent(next) : 'index-new.html';
    url.searchParams.delete('need_login');
    url.searchParams.delete('next');
    url.searchParams.delete('reason');
    window.history.replaceState({}, '', url.pathname + (url.search || '') + url.hash);
    window.location.href = nextUrl;
  };

  if (typeof AuthClient !== 'undefined' && AuthClient.isAuthenticated()) {
    bounce();
    return;
  }

  showToast('請先登入以進入新介面', { type: 'info' });
  setTimeout(() => document.getElementById('btn-auth-open')?.click(), 300);

  window.addEventListener('auth:changed', (ev) => {
    if (ev.detail && ev.detail.user) bounce();
  });
}

// ===== EXPOSE navigateTo globally =====
window.navigateTo = navigateTo;
