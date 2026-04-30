// ===== MAIN APPLICATION LOGIC =====

// ---- Global State ----
let appMode = 'demo'; // 'demo' | 'real'

document.addEventListener('DOMContentLoaded', () => {
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
  // Show demo data on first load
  showDemoDataset();
  setTimeout(() => renderPageCharts('dashboard'), 100);
});

// ===== MODE TOGGLE =====
function initModeToggle() {
  const btn = document.getElementById('btn-mode-toggle');
  if (!btn) return;
  btn.addEventListener('click', () => {
    appMode = appMode === 'demo' ? 'real' : 'demo';
    updateModeUI();
    // Re-render the currently visible page
    const visiblePage = document.querySelector('.page-section:not(.hidden)');
    if (visiblePage) {
      const pageId = visiblePage.id.replace('page-', '');
      renderPageCharts(pageId);
    }
  });
  updateModeUI();
}

function updateModeUI() {
  const icon = document.getElementById('mode-icon');
  const label = document.getElementById('mode-label');
  if (appMode === 'demo') {
    icon.textContent = 'D';
    icon.className = 'flex items-center justify-center w-5 h-5 rounded-md bg-warning-500/15 text-warning-400 text-[10px] font-bold';
    label.textContent = 'Demo 模式';
    label.className = 'flex-1 text-left text-dark-300';
  } else {
    icon.textContent = 'R';
    icon.className = 'flex items-center justify-center w-5 h-5 rounded-md bg-success-500/15 text-success-400 text-[10px] font-bold';
    label.textContent = '實作模式';
    label.className = 'flex-1 text-left text-dark-300';
  }
}

// ===== NAVIGATION =====
const PAGE_NAMES = {
  dashboard: '總覽儀表板',
  datasets: '數據集管理',
  experiments: '實驗室',
  leaderboard: '模型排行榜',
  insights: '洞察與決策',
  // deployments: '部署與 API',
  settings: '系統設定',
};

function navigateTo(page) {
  document.querySelectorAll('.page-section').forEach(s => s.classList.add('hidden'));
  const target = document.getElementById(`page-${page}`);
  if (target) target.classList.remove('hidden');
  document.querySelectorAll('.nav-item').forEach(item => {
    item.classList.toggle('active', item.dataset.page === page);
  });
  document.getElementById('breadcrumb-current').textContent = PAGE_NAMES[page] || page;
  setTimeout(() => renderPageCharts(page), 50);
}

function initNavigation() {
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', e => {
      e.preventDefault();
      const page = item.dataset.page;
      if (page) navigateTo(page);
    });
  });
}

function renderPageCharts(page) {
  // Toggle demo/real content for pages with overlays
  updatePageModeVisibility(page);

  switch (page) {
    case 'dashboard':
      if (appMode === 'demo' || MLEngine.trainedModels.length > 0) {
        renderPerformanceTrend();
        renderTaskDistribution();
        if (appMode === 'real' && MLEngine.trainedModels.length > 0) {
          updateDashboardRealMetrics();
        }
      }
      break;
    case 'datasets':
      renderDatasetPage();
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
      }
      break;
    case 'deployments':
      if (appMode === 'demo') renderApiUsage();
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
      if ((p === 'leaderboard' || p === 'dashboard') && MLEngine.trainedModels.length > 0) {
        realEmptyEl.classList.add('hidden');
        demoEl.classList.remove('hidden');
      } else if (p === 'insights' && MLEngine.trainedModels.length > 0) {
        realEmptyEl.classList.add('hidden');
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
    appMode = 'real';
    updateModeUI();
  }

  // Show loading state
  const zone = document.getElementById('upload-zone');
  zone.innerHTML = `
    <div class="flex items-center justify-center gap-3 py-4">
      <div class="w-5 h-5 border-2 border-primary-400 border-t-transparent rounded-full animate-spin"></div>
      <span class="text-sm text-primary-400">正在解析並分析 <strong>${escapeHtml(file.name)}</strong>...</span>
    </div>
  `;

  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const text = e.target.result;
      const dataset = DataEngine.loadCSV(text, file.name);

      // Show file info bar
      document.getElementById('upload-card').classList.add('hidden');
      const infoBar = document.getElementById('file-info-bar');
      infoBar.classList.remove('hidden');
      document.getElementById('file-info-name').textContent = file.name;
      const sizeMB = (file.size / 1024 / 1024).toFixed(2);
      const sizeKB = (file.size / 1024).toFixed(1);
      const sizeStr = file.size > 1048576 ? `${sizeMB} MB` : `${sizeKB} KB`;
      document.getElementById('file-info-meta').textContent = `${dataset.rowCount.toLocaleString()} 筆資料 | ${dataset.colCount} 個欄位 | ${sizeStr}`;

      // Render analysis
      renderDatasetPage();

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
    // Click to switch dataset
    div.addEventListener('click', (e) => {
      if (e.target.closest('.ds-remove-btn')) return;
      DataEngine.switchDataset(ds.id);
      renderDatasetPage();
    });
    // Remove button
    div.querySelector('.ds-remove-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      DataEngine.removeDataset(ds.id);
      renderDatasetPage();
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
  { action: 'fill_missing',   icon: 'M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10', label: '缺失值處理',  color: 'warning' },
  { action: 'handle_outlier', icon: 'M13 10V3L4 14h7v7l9-11h-7z',                                                                                                                                     label: '異常值處理',  color: 'danger' },
  { action: 'drop_column',    icon: 'M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16',                                   label: '高缺失欄位',  color: 'danger' },
  { action: 'encode',         icon: 'M7 20l4-16m2 16l4-16M6 9h14M4 15h14',                                                                                                                             label: '類別編碼',    color: 'primary' },
  { action: 'decompose_date', icon: 'M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z',                                                                         label: '日期特徵工程', color: 'accent' },
  { action: 'interaction',    icon: 'M4 6h16M4 12h16M4 18h7',                                                                                                                                           label: '交互特徵',    color: 'accent' },
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
    const colorMap = { warning: 'warning-400', danger: 'danger-400', primary: 'primary-400', accent: 'accent-400', success: 'success-400' };
    const textColor = colorMap[cat.color] || 'dark-300';
    const bgColor = cat.color + '-500/8';

    section.innerHTML = `
      <div class="proc-cat-header flex items-center gap-2.5 px-3 py-2 rounded-lg bg-${bgColor} cursor-pointer select-none" data-cat="${cat.action}">
        <svg class="w-4 h-4 text-${textColor} flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="${cat.icon}"/></svg>
        <span class="text-sm font-semibold text-${textColor}">${cat.label}</span>
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
        const status = document.getElementById('global-status');
        if (status) { const txt = document.getElementById('global-status-text'); if (txt) txt.textContent = '訓練完成'; }
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

  // Populate target select with all numeric columns
  const targetSel = document.getElementById('exp-target-select');
  targetSel.innerHTML = '';
  ds.analysis.forEach(col => {
    if (col.type === 'numeric') {
      const opt = document.createElement('option');
      opt.value = col.name;
      opt.textContent = `${col.name} (${col.type})`;
      targetSel.appendChild(opt);
    }
  });

  // --- Feature checkboxes ---
  const featBox = document.getElementById('exp-feature-checkboxes');
  const featCountEl = document.getElementById('exp-feature-count');

  const buildFeatureCheckboxes = () => {
    const targetCol = targetSel.value;
    const numericCols = ds.analysis.filter(c => c.type === 'numeric' && c.name !== targetCol);
    const dateCols = ds.analysis.filter(c => c.type === 'datetime' && c.name !== targetCol);
    featBox.innerHTML = '';
    // Numeric features
    numericCols.forEach(col => {
      const lbl = document.createElement('label');
      lbl.className = 'flex items-center gap-1.5 text-xs text-dark-300 cursor-pointer hover:text-dark-100 transition-colors';
      lbl.innerHTML = `<input type="checkbox" class="exp-feat-cb accent-primary-500" value="${escapeHtml(col.name)}" checked> ${escapeHtml(col.name)}`;
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
        lbl.innerHTML = `<input type="checkbox" class="exp-feat-cb accent-primary-500" value="${escapeHtml(fn)}" checked> <span class="text-cyan-400">${escapeHtml(col.name)}</span>_${escapeHtml(label)}`;
        featBox.appendChild(lbl);
      });
    });
    updateFeatureCount();
  };

  const updateFeatureCount = () => {
    const total = featBox.querySelectorAll('.exp-feat-cb').length;
    const checked = featBox.querySelectorAll('.exp-feat-cb:checked').length;
    featCountEl.textContent = `(${checked}/${total})`;
  };

  featBox.addEventListener('change', updateFeatureCount);
  document.getElementById('exp-feat-all').onclick = () => { featBox.querySelectorAll('.exp-feat-cb').forEach(cb => cb.checked = true); updateFeatureCount(); };
  document.getElementById('exp-feat-none').onclick = () => { featBox.querySelectorAll('.exp-feat-cb').forEach(cb => cb.checked = false); updateFeatureCount(); };

  // --- Algorithm checkboxes ---
  const algoBox = document.getElementById('exp-algo-checkboxes');
  const algoCountEl = document.getElementById('exp-algo-count');

  const buildAlgoCheckboxes = () => {
    algoBox.innerHTML = '';
    Object.entries(MLEngine.ALGORITHMS).forEach(([key, algo]) => {
      const lbl = document.createElement('label');
      lbl.className = 'flex items-center gap-1.5 text-xs text-dark-300 cursor-pointer hover:text-dark-100 transition-colors';
      const typeTag = algo.type === 'regression' ? '回歸' : algo.type === 'classification' ? '分類' : '通用';
      const tagColor = algo.type === 'regression' ? 'text-blue-400' : algo.type === 'classification' ? 'text-amber-400' : 'text-green-400';
      lbl.innerHTML = `<input type="checkbox" class="exp-algo-cb accent-primary-500" value="${key}" data-algo-type="${algo.type}" checked> ${escapeHtml(algo.label)} <span class="${tagColor} text-[10px]">(${typeTag})</span>`;
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
    const uniqueVals = new Set(ds.data.map(r => r[ds.headers.indexOf(colName)])).size;
    const detectedType = uniqueVals <= 10 ? 'classification' : 'regression';
    const label = detectedType === 'classification' ? `分類 (${uniqueVals} 類)` : '回歸 (連續值)';
    document.getElementById('exp-task-label').textContent = label;
    document.getElementById('exp-target-info').textContent = col.stats
      ? `範圍: ${col.stats.min} ~ ${col.stats.max}, 平均: ${col.stats.mean}`
      : '';
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

  // Train button
  const btn = document.getElementById('btn-real-train');
  const newBtn = btn.cloneNode(true);
  btn.parentNode.replaceChild(newBtn, btn);
  newBtn.addEventListener('click', () => {
    // Collect options from UI
    const selectedFeatures = [...featBox.querySelectorAll('.exp-feat-cb:checked')].map(cb => cb.value);
    const selectedAlgos = [...algoBox.querySelectorAll('.exp-algo-cb:checked')].map(cb => cb.value);
    const taskTypeRadio = document.querySelector('input[name="exp-task-type"]:checked');
    const taskType = taskTypeRadio ? taskTypeRadio.value : 'auto';
    const timeSeries = document.getElementById('exp-time-series').checked;

    if (selectedFeatures.length === 0) { alert('請至少選擇一個特徵欄位'); return; }
    if (selectedAlgos.length === 0) { alert('請至少選擇一個演算法'); return; }

    const options = {
      features: selectedFeatures,
      algorithms: selectedAlgos,
      taskType: taskType,
      timeSeries: timeSeries,
    };
    startRealTraining(ds, targetSel.value, options);
  });

  // If already trained, show results
  if (MLEngine.trainedModels.length > 0) {
    document.getElementById('exp-results').classList.remove('hidden');
  }
}

async function startRealTraining(ds, targetCol, options = {}) {
  const btn = document.getElementById('btn-real-train');
  btn.disabled = true;
  btn.innerHTML = '<div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin inline-block mr-2"></div>訓練中...';

  const progressCard = document.getElementById('exp-training-progress');
  const resultsCard = document.getElementById('exp-results');
  progressCard.classList.remove('hidden');
  resultsCard.classList.add('hidden');

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
    const { models, data } = await MLEngine.runExperiment(ds, targetCol, options, onProgress);

    if (models.length === 0) {
      addLog('所有演算法均訓練失敗，請檢查資料或調整設定', 'error');
      btn.innerHTML = '重試訓練';
      btn.disabled = false;
      return;
    }

    // Show results — must un-hide BEFORE rendering charts so ECharts can compute size
    resultsCard.classList.remove('hidden');
    renderExperimentResults(models, data);

    btn.innerHTML = '重新訓練';
    btn.disabled = false;
    btn.classList.remove('opacity-75');

    // Update global status
    const txt = document.getElementById('global-status-text');
    if (txt) txt.textContent = '訓練完成';

  } catch (err) {
    addLog(`錯誤: ${err.message}`, 'error');
    btn.innerHTML = '重試訓練';
    btn.disabled = false;
  }
}

function renderExperimentResults(models, data) {
  const isReg = data.taskType === 'regression';

  // Table headers
  if (isReg) {
    document.getElementById('exp-metric-col1').textContent = 'R² (測試)';
    document.getElementById('exp-metric-col2').textContent = 'RMSE';
    document.getElementById('exp-metric-col3').textContent = 'MAE';
  } else {
    document.getElementById('exp-metric-col1').textContent = 'Accuracy';
    document.getElementById('exp-metric-col2').textContent = 'F1-Score';
    document.getElementById('exp-metric-col3').textContent = 'Precision';
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
    if (isReg) {
      col1 = m.metrics.testR2.toFixed(4);
      col2 = m.metrics.testRMSE.toFixed(4);
      col3 = m.metrics.testMAE.toFixed(4);
    } else {
      col1 = (m.metrics.testAccuracy * 100).toFixed(2) + '%';
      col2 = m.metrics.f1.toFixed(4);
      col3 = m.metrics.precision.toFixed(4);
    }

    const score = m.metrics.testScore;
    const barWidth = Math.max(5, Math.round(score * 100));
    const barColor = score > 0.8 ? 'bg-success-500' : score > 0.5 ? 'bg-warning-500' : 'bg-danger-500';

    tr.innerHTML = `
      <td class="py-3 px-4">${rankBadge}</td>
      <td class="py-3 px-4">
        <p class="font-medium text-sm">${escapeHtml(m.name)}</p>
        <div class="w-24 bg-dark-800 rounded-full h-1.5 mt-1"><div class="${barColor} h-1.5 rounded-full" style="width:${barWidth}%"></div></div>
      </td>
      <td class="py-3 px-4 font-mono text-sm ${i === 0 ? 'text-accent-400 font-semibold' : ''}">${col1}</td>
      <td class="py-3 px-4 font-mono text-sm">${col2}</td>
      <td class="py-3 px-4 font-mono text-sm">${col3}</td>
      <td class="py-3 px-4 text-xs text-dark-400">${m.trainTime.toFixed(0)}ms</td>
    `;
    tbody.appendChild(tr);
  });

  // Feature importance chart (best model) — defer using rAF so DOM has time to layout before ECharts measures
  if (models.length > 0) {
    const best = models[0];
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        renderExpFeatureImportance(best);
        renderExpPredictionChart(best, data);
      });
    });
  }
}

function renderExpFeatureImportance(model) {
  const chart = initChart('chart-exp-importance');
  if (!chart) return;

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

// ===== REAL LEADERBOARD =====
function renderRealLeaderboard() {
  const models = MLEngine.trainedModels;
  if (models.length === 0) return;

  const isReg = models[0].taskType === 'regression';

  // Update dataset dropdown
  const lbSelect = document.getElementById('lb-dataset-select');
  if (lbSelect) {
    const dsName = DataEngine.currentDataset ? DataEngine.currentDataset.fileName : '當前數據集';
    lbSelect.innerHTML = `<option>${escapeHtml(dsName)}</option>`;
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

  models.forEach((m, i) => {
    const tr = document.createElement('tr');
    tr.className = 'border-b border-dark-700/30';

    const rankEl = i === 0
      ? '<span class="inline-flex items-center justify-center w-7 h-7 rounded-full bg-gradient-to-br from-warning-400 to-warning-600 text-dark-950 text-xs font-bold">1</span>'
      : i === 1
        ? '<span class="inline-flex items-center justify-center w-7 h-7 rounded-full bg-dark-600 text-dark-200 text-xs font-bold">2</span>'
        : `<span class="text-dark-400 text-sm">${i + 1}</span>`;

    let extraCols;
    if (isReg) {
      extraCols = `
        <td class="py-3 px-4 font-mono text-xs">${m.metrics.testR2.toFixed(4)}</td>
        <td class="py-3 px-4 font-mono text-xs">${m.metrics.testRMSE.toFixed(4)}</td>
        <td class="py-3 px-4 font-mono text-xs">${m.metrics.testMAE.toFixed(4)}</td>`;
    } else {
      extraCols = `
        <td class="py-3 px-4 font-mono text-xs">${m.metrics.f1.toFixed(4)}</td>
        <td class="py-3 px-4 font-mono text-xs">—</td>
        <td class="py-3 px-4 font-mono text-xs">${(m.metrics.testAccuracy * 100).toFixed(2)}%</td>`;
    }

    const badge = i === 0 ? '<span class="badge badge-success">最佳</span>' : '';

    tr.innerHTML = `
      <td class="py-3 px-4"><input type="checkbox" class="model-select-cb" data-idx="${i}"></td>
      <td class="py-3 px-4">${rankEl}</td>
      <td class="py-3 px-4"><span class="font-medium">${escapeHtml(m.name)}</span></td>
      ${extraCols}
      <td class="py-3 px-4 font-mono text-xs">${m.trainTime.toFixed(0)}ms</td>
      <td class="py-3 px-4 font-mono text-xs">—</td>
      <td class="py-3 px-4">${badge}</td>
      <td class="py-3 px-4">—</td>
    `;
    tbody.appendChild(tr);
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

  const best = models[0];
  const isReg = best.taskType === 'regression';

  // Summary cards
  document.getElementById('insight-best-model').textContent = best.name;
  document.getElementById('insight-score-label').textContent = isReg ? 'R² 分數' : '準確率';
  document.getElementById('insight-best-score').textContent = isReg
    ? best.metrics.testR2.toFixed(4)
    : (best.metrics.testAccuracy * 100).toFixed(2) + '%';
  document.getElementById('insight-feature-count').textContent = best.featureNames.length;
  document.getElementById('insight-model-count').textContent = models.length;

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

// ===== REAL WHAT-IF SIMULATOR =====
const RealWhatIfState = { currentModel: null, values: [] };

function renderRealWhatIf(models) {
  const section = document.getElementById('real-whatif-section');
  if (!section) return;

  // Filter to models that have prediction infrastructure
  const usable = models.filter(m => m.predict && m.featureStats && m.means && m.stds);
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

function buildRealWhatIfSliders() {
  const wrap = document.getElementById('real-whatif-sliders');
  const m = RealWhatIfState.currentModel;
  if (!wrap || !m) return;

  wrap.innerHTML = '';
  m.featureNames.forEach((name, i) => {
    const stats = m.featureStats[i];
    const range = stats.max - stats.min;
    const step = range > 100 ? 1 : (range / 200 || 0.01);
    const cur = RealWhatIfState.values[i];

    const div = document.createElement('div');
    div.className = 'slider-group';
    div.innerHTML = `
      <div class="flex items-center justify-between mb-1">
        <label class="text-xs font-medium text-dark-200 truncate" title="${escapeHtml(name)}">${escapeHtml(name)}</label>
        <span class="text-xs font-mono text-accent-400" data-val-display="${i}">${formatRwifVal(cur)}</span>
      </div>
      <input type="range" min="${stats.min}" max="${stats.max}" step="${step}" value="${cur}" data-feat-idx="${i}" class="slider-input w-full">
      <div class="flex justify-between text-[9px] text-dark-500 mt-0.5">
        <span>${formatRwifVal(stats.min)}</span>
        <span class="text-dark-600">μ ${formatRwifVal(stats.mean)}</span>
        <span>${formatRwifVal(stats.max)}</span>
      </div>
    `;
    wrap.appendChild(div);
  });

  wrap.querySelectorAll('input[type="range"]').forEach(input => {
    input.addEventListener('input', (e) => {
      const idx = parseInt(e.target.dataset.featIdx);
      const v = parseFloat(e.target.value);
      RealWhatIfState.values[idx] = v;
      const display = wrap.querySelector(`[data-val-display="${idx}"]`);
      if (display) display.textContent = formatRwifVal(v);
      updateRealWhatIfPrediction();
    });
  });
}

function updateRealWhatIfPrediction() {
  const m = RealWhatIfState.currentModel;
  if (!m) return;

  const xNorm = RealWhatIfState.values.map((v, i) => (v - m.means[i]) / (m.stds[i] || 1));
  let pred;
  try { pred = m.predict(xNorm); } catch (e) { pred = NaN; }

  const baselineNorm = new Array(m.featureNames.length).fill(0);
  let baseline;
  try { baseline = m.predict(baselineNorm); } catch (e) { baseline = NaN; }

  const isReg = m.taskType === 'regression';

  document.getElementById('real-whatif-prediction').textContent = isReg ? formatRwifVal(pred) : `Class ${pred}`;
  document.getElementById('real-whatif-target-label').textContent = m.targetName || '目標變數';
  document.getElementById('real-whatif-baseline').textContent = isReg ? formatRwifVal(baseline) : `Class ${baseline}`;
  document.getElementById('real-whatif-model-name').textContent = m.name;

  if (isReg && !isNaN(pred) && !isNaN(baseline)) {
    const delta = pred - baseline;
    const deltaEl = document.getElementById('real-whatif-delta');
    const sign = delta >= 0 ? '+' : '';
    deltaEl.textContent = `${sign}${formatRwifVal(delta)}`;
    deltaEl.className = `font-mono ${delta > 0 ? 'text-success-400' : delta < 0 ? 'text-danger-400' : 'text-dark-300'}`;
  } else {
    document.getElementById('real-whatif-delta').textContent = '—';
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
  const pairs = names.map((n, i) => ({ name: n, value: values[i] })).sort((a, b) => b.value - a.value).slice(0, 15);

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

function renderRealPredScatter(model, isReg) {
  const chart = initChart('chart-real-pred-scatter');
  if (!chart) return;

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
  // Update the 4 metric cards (they're the first metric-card elements in the dashboard)
  const cards = document.querySelectorAll('#page-dashboard .metric-card');
  if (cards.length >= 4) {
    // Active datasets
    cards[0].querySelector('.text-3xl').textContent = DataEngine.datasets.length.toString();
    // Completed experiments
    cards[1].querySelector('.text-3xl').textContent = models.length.toString();
    // Best model score
    const best = models[0];
    const isReg = best.taskType === 'regression';
    const scoreStr = isReg
      ? (best.metrics.testR2 * 100).toFixed(1) + '<span class="text-lg text-dark-400">%</span>'
      : (best.metrics.testScore * 100).toFixed(1) + '<span class="text-lg text-dark-400">%</span>';
    cards[2].querySelector('.text-3xl').innerHTML = scoreStr;
    cards[2].querySelector('.text-dark-400.text-sm').textContent = isReg ? '最佳模型 R²' : '最佳模型準確率';
    // Deployed models
    cards[3].querySelector('.text-3xl').textContent = '0';
  }
}

// ===== SYSTEM SETTINGS =====
function initSettings() {
  const timeoutSel = document.getElementById('setting-timeout');
  const autofeCb = document.getElementById('setting-autofe');
  const hpoInput = document.getElementById('setting-hpotrials');
  const emailCb = document.getElementById('setting-email');
  const kfoldSel = document.getElementById('setting-kfold');
  const algoContainer = document.getElementById('setting-algos-container');
  const saveBtn = document.getElementById('btn-save-settings');
  const toast = document.getElementById('settings-save-toast');

  let activeAlgos = ['XGBoost', 'LightGBM', 'CatBoost'];

  const renderAlgos = () => {
    if (!algoContainer) return;
    algoContainer.querySelectorAll('.setting-algo-chip').forEach(chip => {
      const val = chip.dataset.val;
      if (activeAlgos.includes(val)) {
        chip.className = 'setting-algo-chip text-xs bg-primary-500/10 text-primary-400 px-2 py-1 rounded-lg cursor-pointer select-none';
      } else {
        chip.className = 'setting-algo-chip text-xs bg-dark-600 text-dark-400 px-2 py-1 rounded-lg cursor-pointer select-none';
      }
    });
  };

  if (algoContainer) {
    algoContainer.addEventListener('click', (e) => {
      const chip = e.target.closest('.setting-algo-chip');
      if (chip) {
        const val = chip.dataset.val;
        if (activeAlgos.includes(val)) {
          activeAlgos = activeAlgos.filter(v => v !== val);
        } else {
          activeAlgos.push(val);
        }
        renderAlgos();
      }
    });
  }

  // Load from localStorage
  const configStr = localStorage.getItem('automl_settings');
  if (configStr) {
    try {
      const config = JSON.parse(configStr);
      if (timeoutSel && config.timeout) timeoutSel.value = config.timeout;
      if (autofeCb && config.autofe !== undefined) autofeCb.checked = config.autofe;
      if (hpoInput && config.hpo) hpoInput.value = config.hpo;
      if (emailCb && config.email !== undefined) emailCb.checked = config.email;
      if (kfoldSel && config.kfold) kfoldSel.value = config.kfold;
      if (config.activeAlgos) {
        activeAlgos = config.activeAlgos;
        renderAlgos();
      }
    } catch (e) {}
  } else {
    renderAlgos(); // Initial render for defaults
  }

  // Save to localStorage
  if (saveBtn) {
    saveBtn.addEventListener('click', () => {
      const newConfig = {
        timeout: timeoutSel ? timeoutSel.value : '1 小時',
        autofe: autofeCb ? autofeCb.checked : true,
        hpo: hpoInput ? hpoInput.value : 200,
        email: emailCb ? emailCb.checked : true,
        kfold: kfoldSel ? kfoldSel.value : '5-Fold',
        activeAlgos: activeAlgos
      };
      localStorage.setItem('automl_settings', JSON.stringify(newConfig));
      
      // Toast animation
      if (toast) {
        toast.style.opacity = '1';
        setTimeout(() => { toast.style.opacity = '0'; }, 2000);
      }
    });
  }
}

// ===== EXPOSE navigateTo globally =====
window.navigateTo = navigateTo;
