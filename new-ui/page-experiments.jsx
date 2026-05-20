// =============================================================================
// Page: Experiments — left list + right detail/new-exp form
// =============================================================================

// 把後端 training run 映射成 UI 用的 experiment 形狀,讓既有 component 不用大改
function _runToExp(run, allModels) {
  const summary = run.resultsSummary || {};
  const top = summary.topModels?.[0] || summary.perSource?.[0] || null;
  const elapsed = run.elapsedSec ?? (run.finishedAt && run.startedAt ? run.finishedAt - run.startedAt : 0);
  const myModels = (allModels || []).filter(m => m.trainingRunId === run.id);
  return {
    id: run.id,
    name: `${run.datasetName || run.datasetId || '?'} · ${run.target}`,
    dataset: run.datasetName || run.datasetId,
    target: run.target,
    task: run.taskType === 'regression' ? 'regression' : 'classification',
    status: run.status,
    startedAt: _expRelTime(run.startedAt),
    startedAtSec: run.startedAt,
    duration: _fmtElapsed(elapsed),
    error: run.errorMsg,
    models: myModels.length || (run.modelIds || []).length,
    modelIds: run.modelIds || [],
    myModels,
    engine: run.engine,
    sources: run.sources || [],
    options: run.options || {},
    best: top ? {
      algo: (top.name || '').replace(/^\[(原始|預處理)\]\s*/, '') || top.algo || '?',
      metric: top.metric || (run.taskType === 'regression' ? 'R²' : 'F1'),
      value: typeof top.score === 'number' ? top.score.toFixed(3)
           : typeof top.bestScore === 'number' ? top.bestScore.toFixed(3) : '—',
    } : null,
    progress: run.status === 'running' ? 0.5 : 1,
  };
}

function _expRelTime(sec) {
  if (!sec) return '?';
  const diff = Math.floor(Date.now() / 1000 - sec);
  if (diff < 60)   return `${diff}s 前`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m 前`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h 前`;
  return `${Math.floor(diff / 86400)} 天前`;
}
function _fmtElapsed(sec) {
  if (!sec || sec === 0) return '—';
  if (sec < 60) return `${sec.toFixed(0)}s`;
  const m = Math.floor(sec / 60);
  const s = Math.round(sec - m * 60);
  return `${m}m ${s.toString().padStart(2, '0')}s`;
}

function PageExperiments({ onNavigate }) {
  const [mode, setMode] = React.useState('detail');
  const [selectedId, setSelectedId] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError]     = React.useState(null);
  const [runs, setRuns]       = React.useState([]);
  const [models, setModels]   = React.useState([]);
  const [search, setSearch]   = React.useState('');
  const [statusFilter, setStatusFilter] = React.useState('all');

  const loadRuns = React.useCallback((force = false) => {
    setLoading(true);
    Promise.all([
      NewUI.api.getCached('/api/training-runs?limit=200', { force }).then(r => r.runs || []).catch(() => []),
      NewUI.api.getCached('/api/models?limit=1000', { force }).then(r => (r.models || []).map(m => ({ ...m, bundle: m.bundle || {} }))).catch(() => []),
    ]).then(([rRuns, rModels]) => {
      setRuns(rRuns);
      setModels(rModels);
      setLoading(false);
      if (!selectedId && rRuns.length > 0) setSelectedId(rRuns[0].id);
    }).catch(err => { setError(err.message); setLoading(false); });
  }, [selectedId]);

  React.useEffect(() => { loadRuns(false); }, []);

  const experiments = React.useMemo(() => runs.map(r => _runToExp(r, models)), [runs, models]);
  const counts = React.useMemo(() => ({
    all: experiments.length,
    running: experiments.filter(e => e.status === 'running').length,
    completed: experiments.filter(e => e.status === 'completed').length,
    failed: experiments.filter(e => e.status === 'failed').length,
  }), [experiments]);

  const filtered = React.useMemo(() => {
    return experiments.filter(e => {
      if (statusFilter !== 'all' && e.status !== statusFilter) return false;
      if (search && !(`${e.name} ${e.dataset} ${e.target}`).toLowerCase().includes(search.toLowerCase())) return false;
      return true;
    });
  }, [experiments, search, statusFilter]);

  const selected = experiments.find(e => e.id === selectedId);

  function startNew() { setMode('new'); }

  if (loading) {
    return <div style={{ padding: 24, textAlign: 'center', color: 'var(--fg-muted)', minHeight: 300 }}>
      <span className="t-label">載入實驗中...</span>
    </div>;
  }
  if (error) {
    return <div style={{ padding: 24 }}>
      <Surface style={{ padding: 24 }}>
        <Row gap={10}><Icon name="warning" size={20} style={{ color: 'var(--bad)' }} /><span className="t-label">{error}</span></Row>
      </Surface>
    </div>;
  }
  if (experiments.length === 0) {
    return (
      <div style={{ padding: 24 }}>
        <Row align="end" style={{ marginBottom: 16 }}>
          <div><h1 className="t-h1">實驗</h1><p className="t-label">尚無任何訓練紀錄</p></div>
        </Row>
        <Surface style={{ padding: 48, textAlign: 'center' }}>
          <Icon name="flask" size={40} style={{ color: 'var(--primary)', marginBottom: 12 }} />
          <div className="t-title">尚無實驗</div>
          <div className="t-label" style={{ marginTop: 6, marginBottom: 20 }}>
            目前新介面尚未實作訓練表單,請先在舊介面上跑一次,結果會自動同步回來
          </div>
          <Row gap={8} style={{ justifyContent: 'center' }}>
            <Button variant="primary" icon="arrowLeft" onClick={() => { window.location.href = 'index.html#experiments'; }}>
              到舊介面開始實驗
            </Button>
          </Row>
        </Surface>
      </div>
    );
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', height: 'calc(100vh - 49px)' }} >
      {/* ===== Left: experiment list ===== */}
      <div style={{ borderRight: '1px solid var(--bd-subtle)', display: 'flex', flexDirection: 'column' }}>
        <div style={{ padding: 16 }}>
          <Row style={{ justifyContent: 'space-between', marginBottom: 12 }}>
            <h1 className="t-h1">實驗</h1>
            <Button variant="primary" icon="plus" size="sm" onClick={startNew}>新實驗</Button>
          </Row>
          <input className="input" placeholder="搜尋實驗..." value={search} onChange={e => setSearch(e.target.value)} style={{ marginBottom: 10, width: '100%' }} />
          <Row gap={4} style={{ flexWrap: 'wrap' }}>
            {[
              { v: 'all', l: `全部 ${counts.all}` },
              { v: 'running', l: `running ${counts.running}` },
              { v: 'completed', l: `done ${counts.completed}` },
              { v: 'failed', l: `failed ${counts.failed}` },
            ].map(f => (
              <button key={f.v} onClick={() => setStatusFilter(f.v)}
                      className={`chip mono ${statusFilter === f.v ? 'chip-primary' : ''}`}
                      style={{ cursor: 'pointer', border: 'none' }}>
                {f.l}
              </button>
            ))}
          </Row>
        </div>
        <div style={{ overflow: 'auto', flex: 1 }}>
          {filtered.length === 0
            ? <div style={{ padding: 24, textAlign: 'center', color: 'var(--fg-muted)' }}>
                <span className="t-label">沒有符合條件的實驗</span>
              </div>
            : filtered.map(exp => (
                <ExpListItem key={exp.id} exp={exp}
                             active={mode === 'detail' && selectedId === exp.id}
                             onClick={() => { setSelectedId(exp.id); setMode('detail'); }} />
              ))}
        </div>
      </div>

      {/* ===== Right: detail or new ===== */}
      <div style={{ overflow: 'auto' }}>
        {mode === 'new'
          ? <NewExperimentReal onCancel={() => setMode('detail')}
                               onDone={(runId) => {
                                 // 訓練完成 → 強制重抓 runs 列表,再切到該 run 的 detail
                                 loadRuns(true);
                                 setMode('detail');
                                 if (runId) setSelectedId(runId);
                               }} />
          : (selected ? <ExperimentDetail exp={selected} onNavigate={onNavigate} /> : null)
        }
      </div>
    </div>
  );
}

// 後端接受的演算法 keys (對應 api/train/train.py _ALGO_LABELS)
const SKLEARN_ALGOS = [
  { id: 'logistic',              name: 'Logistic Regression', kind: 'linear', task: 'cls' },
  { id: 'naive_bayes',          name: 'Naive Bayes',         kind: 'prob',   task: 'cls' },
  { id: 'knn_5',                name: 'KNN (k=5)',           kind: 'instance', task: 'both' },
  { id: 'decision_tree',        name: 'Decision Tree',       kind: 'tree',   task: 'both' },
  { id: 'random_forest',        name: 'Random Forest',       kind: 'tree',   task: 'both' },
  { id: 'gradient_boosting',    name: 'Gradient Boosting',   kind: 'gbm',    task: 'both' },
  { id: 'hist_gradient_boosting', name: 'HistGradientBoosting', kind: 'gbm', task: 'both' },
  { id: 'xgboost',              name: 'XGBoost',             kind: 'gbm',    task: 'both' },
  { id: 'lightgbm',             name: 'LightGBM',            kind: 'gbm',    task: 'both' },
  { id: 'catboost',             name: 'CatBoost',            kind: 'gbm',    task: 'both' },
  { id: 'svc',                  name: 'SVC',                 kind: 'kernel', task: 'cls' },
  { id: 'voting',               name: 'Voting Ensemble',     kind: 'ens',    task: 'both' },
  { id: 'stacking',             name: 'Stacking Ensemble',   kind: 'ens',    task: 'both' },
  // regression-only
  { id: 'linear_regression',    name: 'Linear Regression',   kind: 'linear', task: 'reg' },
  { id: 'ridge',                name: 'Ridge',               kind: 'linear', task: 'reg' },
  { id: 'lasso',                name: 'Lasso',               kind: 'linear', task: 'reg' },
  { id: 'svr',                  name: 'SVR',                 kind: 'kernel', task: 'reg' },
];
const SKLEARN_DEFAULT = ['logistic', 'random_forest', 'xgboost', 'lightgbm'];

function NewExperimentReal({ onCancel, onDone }) {
  // ---- 載入 datasets ----
  const [datasets, setDatasets] = React.useState([]);
  const [loadingDs, setLoadingDs] = React.useState(true);
  const [datasetId, setDatasetId] = React.useState(null);
  const [detail, setDetail] = React.useState(null);     // 選定 dataset 的欄位資訊
  const [target, setTarget] = React.useState(null);

  const [engine, setEngine] = React.useState('sklearn');
  const [taskType, setTaskType] = React.useState('auto');
  const [algos, setAlgos] = React.useState(SKLEARN_DEFAULT);
  const [testSize, setTestSize] = React.useState(20);

  // pipeline 選項
  const [pFast, setPFast] = React.useState(true);
  const [pSkipDL, setPSkipDL] = React.useState(false);
  const [pSkipNAS, setPSkipNAS] = React.useState(false);
  const [pTimeSeries, setPTimeSeries] = React.useState(false);
  const [pMetric, setPMetric] = React.useState('f1');
  const [pTimeLimit, setPTimeLimit] = React.useState(0);

  // 訓練狀態
  const [training, setTraining] = React.useState(false);
  const [log, setLog] = React.useState([]);
  const [progress, setProgress] = React.useState(0);
  const [step, setStep] = React.useState('');
  const [doneRunId, setDoneRunId] = React.useState(null);
  const abortRef = React.useRef(null);
  const logBoxRef = React.useRef(null);

  React.useEffect(() => {
    NewUI.api.getCached('/api/dataset/list')
      .then(r => {
        const list = r.datasets || [];
        setDatasets(list);
        if (list.length > 0) setDatasetId(list[0].id);
      })
      .finally(() => setLoadingDs(false));
  }, []);

  // 選 dataset → 拉欄位 + 預設 target = 最後一欄
  React.useEffect(() => {
    if (!datasetId) { setDetail(null); return; }
    NewUI.api.getCached(`/api/dataset/${encodeURIComponent(datasetId)}`)
      .then(d => {
        setDetail(d);
        const headers = d.headers || [];
        if (headers.length > 0) setTarget(headers[headers.length - 1]);
      })
      .catch(() => setDetail(null));
  }, [datasetId]);

  // log 自動捲到底
  React.useEffect(() => {
    if (logBoxRef.current) logBoxRef.current.scrollTop = logBoxRef.current.scrollHeight;
  }, [log]);

  const headers = detail?.headers || [];
  const ds = datasets.find(d => d.id === datasetId);
  const visibleAlgos = SKLEARN_ALGOS.filter(a => {
    if (taskType === 'classification') return a.task === 'cls' || a.task === 'both';
    if (taskType === 'regression') return a.task === 'reg' || a.task === 'both';
    return a.task !== 'reg';  // auto 預設展示分類 + both (回歸演算法 auto 時不主動列)
  });

  function addLog(msg, level = 'info') {
    const ts = new Date().toLocaleTimeString('zh-TW', { hour12: false });
    setLog(prev => [...prev.slice(-200), { ts, msg, level }]);
  }

  async function startTraining() {
    if (!datasetId || !target) { alert('請選 dataset 和 target'); return; }
    if (engine === 'sklearn' && algos.length === 0) { alert('請至少選一個演算法'); return; }
    setTraining(true);
    setLog([]);
    setProgress(0);
    setStep('準備中...');
    setDoneRunId(null);
    abortRef.current = new AbortController();

    addLog(`啟動 ${engine} 訓練 — dataset=${ds?.fileName} target=${target}`, 'info');

    try {
      if (engine === 'sklearn') {
        await NewUI.api.streamSSE('/api/train/stream', {
          datasetId, target,
          features: null,                  // null = 用全部特徵
          algorithms: algos,
          options: { taskType, testSize: testSize / 100, randomState: 42 },
          sources: ['raw'],
          preprocessorId: null,
        }, onEvent, abortRef.current.signal);
      } else {
        // pipeline 走 FormData
        const form = new FormData();
        form.append('datasetId', datasetId);
        form.append('target', target);
        form.append('sources', JSON.stringify(['raw']));   // 後端用 json.loads 解析
        form.append('fast', String(pFast));
        form.append('skipDl', String(pSkipDL));
        form.append('noNas', String(pSkipNAS));
        form.append('timeSeries', String(pTimeSeries));
        form.append('metric', pMetric);
        form.append('timeLimit', String(pTimeLimit));
        await NewUI.api.streamSSE('/api/train/pipeline/stream', form, onEvent, abortRef.current.signal);
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        addLog('已取消訓練', 'warning');
      } else {
        addLog(`錯誤: ${err.message}`, 'error');
      }
    } finally {
      setTraining(false);
      // 訓練結束 → 清 cache + 通知 sidebar 重抓
      NewUI.api.invalidate('/api/training-runs');
      NewUI.api.invalidate('/api/models');
      window.dispatchEvent(new CustomEvent('newui:refresh-counts'));
    }
  }

  function onEvent(ev) {
    if (ev.type === 'log') {
      addLog(ev.msg, ev.level || ev.logType || 'info');
    } else if (ev.type === 'progress') {
      if (typeof ev.pct === 'number') setProgress(ev.pct);
      if (ev.step) setStep(ev.step);
    } else if (ev.type === 'model') {
      // sklearn 每個 model 一個事件
      const b = ev.bundle || {};
      addLog(`✓ 模型完成: ${b.name || b.type}`, 'success');
    } else if (ev.type === 'done') {
      setProgress(100);
      setStep('完成');
      addLog('✓ 訓練完成', 'success');
      if (ev.runId) setDoneRunId(ev.runId);
    } else if (ev.type === 'error') {
      addLog(`✗ ${ev.message}`, 'error');
    }
  }

  function cancel() {
    if (abortRef.current) abortRef.current.abort();
  }

  if (loadingDs) {
    return <div style={{ padding: 24, color: 'var(--fg-muted)' }}><span className="t-label">載入數據集...</span></div>;
  }
  if (datasets.length === 0) {
    return (
      <div style={{ padding: 24 }}>
        <Surface style={{ padding: 32, textAlign: 'center' }}>
          <Icon name="data" size={32} style={{ color: 'var(--primary)', marginBottom: 12 }} />
          <div className="t-title">還沒有數據集</div>
          <div className="t-label" style={{ marginTop: 6, marginBottom: 16 }}>先到「數據」頁上傳 CSV</div>
          <Button variant="ghost" onClick={onCancel}>返回</Button>
        </Surface>
      </div>
    );
  }

  return (
    <div style={{ padding: 24 }}>
      <Row style={{ justifyContent: 'space-between', marginBottom: 16 }}>
        <div>
          <h1 className="t-h1">新實驗</h1>
          <p className="t-label" style={{ marginTop: 4 }}>選 dataset、target、引擎 → 開始訓練 (即時 log)</p>
        </div>
        <Button variant="ghost" onClick={onCancel} disabled={training}>取消</Button>
      </Row>

      <Stack gap={12} style={{ maxWidth: 920 }}>
        {/* Essentials */}
        <Surface>
          <CardHeader title="基本設定" />
          <div style={{ padding: 16, display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
            <Field label="數據集">
              <select className="input" value={datasetId || ''} onChange={e => setDatasetId(e.target.value)} disabled={training}>
                {datasets.map(d => (
                  <option key={d.id} value={d.id}>{d.fileName} ({(d.rowCount || 0).toLocaleString()} 列)</option>
                ))}
              </select>
            </Field>
            <Field label="目標欄位 (target)" hint={headers.length ? `共 ${headers.length} 欄` : '載入中...'}>
              <select className="input" value={target || ''} onChange={e => setTarget(e.target.value)} disabled={training}>
                {headers.map(h => <option key={h} value={h}>{h}</option>)}
              </select>
            </Field>
            <Field label="任務類型">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6 }}>
                {[{ id: 'auto', label: '自動' }, { id: 'classification', label: '分類' }, { id: 'regression', label: '回歸' }].map(opt => (
                  <button key={opt.id} onClick={() => setTaskType(opt.id)} disabled={training}
                          className={taskType === opt.id ? 'btn btn-primary btn-sm' : 'btn btn-ghost btn-sm'}>
                    {opt.label}
                  </button>
                ))}
              </div>
            </Field>
            <Field label="測試集比例" hint={`${100 - testSize}% 訓練 / ${testSize}% 測試`}>
              <Range value={testSize} onChange={setTestSize} min={10} max={50} step={5} format={v => v + '%'} />
            </Field>
          </div>
        </Surface>

        {/* Engine */}
        <Surface>
          <CardHeader title="訓練引擎" />
          <div style={{ padding: 16, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            {[
              { id: 'sklearn', name: 'sklearn (預設)', body: '多演算法 · 秒~分鐘級', recommended: true },
              { id: 'pipeline', name: 'pipeline (深度)', body: 'HPO + NAS + Stacking · 數分鐘起', warning: '僅分類 · CPU 慢' },
            ].map(opt => (
              <button key={opt.id} onClick={() => !training && setEngine(opt.id)} disabled={training}
                      style={{
                        textAlign: 'left', padding: 14, borderRadius: 8, cursor: training ? 'not-allowed' : 'pointer',
                        background: engine === opt.id ? 'var(--primary-soft)' : 'var(--bg-sunken)',
                        border: '1px solid ' + (engine === opt.id ? 'var(--primary-line)' : 'var(--bd-subtle)'),
                      }}>
                <Row style={{ justifyContent: 'space-between', marginBottom: 6 }}>
                  <span className="t-body-lg fg-1" style={{ fontWeight: 500 }}>{opt.name}</span>
                  {opt.recommended && <Chip tone="primary" className="mono">建議</Chip>}
                  {opt.warning && <Chip tone="warn" className="mono">注意</Chip>}
                </Row>
                <div className="t-label">{opt.body}</div>
              </button>
            ))}
          </div>
        </Surface>

        {/* sklearn algorithms */}
        {engine === 'sklearn' && (
          <Surface>
            <CardHeader title="演算法" subtitle={`已選 ${algos.length}`}
                        right={<Row gap={6}>
                          <Button variant="bare" size="sm" onClick={() => setAlgos(visibleAlgos.map(a => a.id))}>全選</Button>
                          <Button variant="bare" size="sm" onClick={() => setAlgos(SKLEARN_DEFAULT)}>建議</Button>
                          <Button variant="bare" size="sm" onClick={() => setAlgos([])}>清除</Button>
                        </Row>} />
            <div style={{ padding: 14, display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
              {visibleAlgos.map(a => (
                <label key={a.id} style={{
                  padding: '8px 10px',
                  background: algos.includes(a.id) ? 'var(--primary-soft)' : 'var(--bg-sunken)',
                  border: '1px solid ' + (algos.includes(a.id) ? 'var(--primary-line)' : 'var(--bd-subtle)'),
                  borderRadius: 6, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 8,
                }}>
                  <input type="checkbox" checked={algos.includes(a.id)} disabled={training}
                         onChange={e => setAlgos(prev => e.target.checked ? [...prev, a.id] : prev.filter(x => x !== a.id))}
                         style={{ accentColor: 'var(--primary)' }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="t-body fg-1">{a.name}</div>
                    <div className="t-label mono" style={{ fontSize: 10 }}>{a.kind}</div>
                  </div>
                </label>
              ))}
            </div>
          </Surface>
        )}

        {/* pipeline options */}
        {engine === 'pipeline' && (
          <Surface style={{ borderColor: 'var(--warn)', borderLeftWidth: 2 }}>
            <CardHeader title="Pipeline 進階設定" right={<Chip tone="warn">僅分類</Chip>} />
            <div style={{ padding: 16, display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
              <ToggleCard label="快速模式" hint="縮減 trials" checked={pFast} onChange={setPFast} />
              <ToggleCard label="跳過 DL" hint="只跑傳統 ML" checked={pSkipDL} onChange={setPSkipDL} />
              <ToggleCard label="跳過 NAS" hint="用預設架構" checked={pSkipNAS} onChange={setPSkipNAS} />
              <ToggleCard label="時序模式" hint="TCN/PatchTST" checked={pTimeSeries} onChange={setPTimeSeries} />
            </div>
            <div style={{ padding: 16, paddingTop: 0, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <Field label="最佳化指標">
                <select className="input" value={pMetric} onChange={e => setPMetric(e.target.value)}>
                  <option value="f1">F1 (macro)</option>
                  <option value="accuracy">Accuracy</option>
                </select>
              </Field>
              <Field label="時間上限 (秒)" hint="0 = 不限">
                <input type="number" className="input mono" value={pTimeLimit} min={0} step={60}
                       onChange={e => setPTimeLimit(parseInt(e.target.value) || 0)} />
              </Field>
            </div>
          </Surface>
        )}

        {/* Footer / start */}
        {!training && !doneRunId && (
          <Row style={{ justifyContent: 'flex-end', paddingTop: 4 }}>
            <Button variant="primary" iconRight="play" onClick={startTraining}>開始訓練</Button>
          </Row>
        )}

        {/* Live training log */}
        {(training || log.length > 0) && (
          <Surface>
            <CardHeader title="訓練進度"
                        subtitle={step}
                        right={training
                          ? <button className="btn btn-danger-cancel btn-sm" onClick={cancel}>
                              <Icon name="close" size={12} /> 取消訓練
                            </button>
                          : doneRunId
                            ? <Button variant="primary" size="sm" iconRight="arrowRight" onClick={() => onDone(doneRunId)}>看結果</Button>
                            : null} />
            <div style={{ padding: 16 }}>
              <div style={{ height: 4, background: 'var(--bg-sunken)', borderRadius: 2, overflow: 'hidden', marginBottom: 12 }}>
                <div style={{ height: '100%', width: `${progress}%`, background: 'var(--primary)', transition: 'width .3s' }} />
              </div>
              <div ref={logBoxRef} style={{
                maxHeight: 320, overflow: 'auto', background: 'var(--bg-sunken)',
                borderRadius: 7, padding: 12, fontFamily: 'JetBrains Mono, monospace', fontSize: 11, lineHeight: 1.6,
              }}>
                {log.map((l, i) => {
                  const color = l.level === 'error' ? 'var(--bad)'
                              : l.level === 'warning' ? 'var(--warn)'
                              : l.level === 'success' ? 'var(--good)'
                              : l.level === 'muted' ? 'var(--fg-faint)'
                              : 'var(--fg-default)';
                  return <div key={i} style={{ color, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>[{l.ts}] {l.msg}</div>;
                })}
              </div>
            </div>
          </Surface>
        )}
      </Stack>
    </div>
  );
}

function ExpListItem({ exp, active, onClick }) {
  const tone = exp.status === 'completed' ? 'good' : exp.status === 'failed' ? 'bad' : exp.status === 'running' ? 'active' : 'muted';
  return (
    <button onClick={onClick} style={{
      width: '100%', padding: '12px 16px', display: 'block',
      background: active ? 'var(--primary-soft)' : 'transparent',
      borderBottom: '1px solid var(--bd-subtle)',
      borderLeft: active ? '2px solid var(--primary)' : '2px solid transparent',
      textAlign: 'left', cursor: 'pointer',
    }}
      onMouseOver={e => { if (!active) e.currentTarget.style.background = 'var(--bg-hover)'; }}
      onMouseOut={e => { if (!active) e.currentTarget.style.background = 'transparent'; }}>
      <Row gap={8} style={{ marginBottom: 4 }}>
        <Dot tone={tone} />
        <span className="t-body-lg fg-1" style={{
          fontWeight: 500, flex: 1, minWidth: 0,
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>{exp.name}</span>
        <span className="t-label mono">{exp.startedAt}</span>
      </Row>
      <Row gap={8} style={{ marginLeft: 14 }}>
        <span className="t-label mono">{exp.task === 'classification' ? 'cls' : 'reg'}</span>
        <span className="fg-4">·</span>
        {exp.status === 'completed' && exp.best && (
          <span className="t-label mono fg-1">{exp.best.metric} {exp.best.value}</span>
        )}
        {exp.status === 'running' && (
          <span className="t-label mono" style={{ color: 'var(--primary)' }}>
            {(exp.progress * 100).toFixed(0)}% · {exp.duration}
          </span>
        )}
        {exp.status === 'failed' && (
          <span className="t-label mono" style={{ color: 'var(--bad)' }}>失敗</span>
        )}
      </Row>
    </button>
  );
}

// =====================
// Experiment Detail
// =====================
function ExperimentDetail({ exp, onNavigate }) {
  if (!exp) return null;
  const [tab, setTab] = React.useState('summary');

  return (
    <div>
      {/* Sub-header */}
      <div style={{ padding: '20px 24px 0' }}>
        <Row style={{ justifyContent: 'space-between', marginBottom: 14, alignItems: 'flex-start', flexWrap: 'wrap', gap: 12 }}>
          <div style={{ minWidth: 0, flex: '1 1 400px' }}>
            <Row gap={10}>
              <h1 className="t-h1" style={{ whiteSpace: 'nowrap' }}>{exp.name}</h1>
              {exp.status === 'completed' && <Chip tone="good" icon="checkCircle">完成</Chip>}
              {exp.status === 'running' && <Chip tone="primary" icon="play">執行中</Chip>}
              {exp.status === 'failed' && <Chip tone="bad" icon="warning">失敗</Chip>}
            </Row>
            <Row gap={8} style={{ marginTop: 6, flexWrap: 'wrap' }}>
              <span className="t-label mono fg-3">{exp.id}</span>
              <span className="fg-4">·</span>
              <span className="t-label">dataset: <span className="mono fg-1">{exp.dataset}</span></span>
              <span className="fg-4">·</span>
              <span className="t-label">{exp.task === 'classification' ? '分類' : '回歸'}</span>
              <span className="fg-4">·</span>
              <span className="t-label">耗時 <span className="mono fg-1">{exp.duration}</span></span>
            </Row>
          </div>
          <Row gap={6} style={{ flexShrink: 0 }}>
            <Button variant="ghost" size="sm" icon="copy" title="複製設定" />
            <Button variant="ghost" size="sm" icon="download" title="匯出" />
            {exp.status === 'completed' && (
              <Button variant="primary" iconRight="arrowRight" onClick={() => onNavigate('insights')}>看洞察</Button>
            )}
          </Row>
        </Row>

        <PageTabs
          items={[
            { value: 'summary', label: '摘要' },
            { value: 'models',  label: '模型', count: exp.models },
            { value: 'logs',    label: '訓練 log' },
            { value: 'config',  label: '設定' },
          ]}
          value={tab}
          onChange={setTab}
        />
      </div>

      <div style={{ padding: 20 }}>
        {tab === 'summary' && <ExpSummary exp={exp} onNavigate={onNavigate} />}
        {tab === 'models'  && <ExpModelsList exp={exp} onNavigate={onNavigate} />}
        {tab === 'logs'    && <ExpLogs exp={exp} />}
        {tab === 'config'  && <ExpConfig exp={exp} />}
      </div>
    </div>
  );
}

function ExpSummary({ exp, onNavigate }) {
  if (exp.status === 'failed') {
    return (
      <Surface>
        <div style={{ padding: 24, textAlign: 'center' }}>
          <div style={{ width: 48, height: 48, borderRadius: 12, background: 'var(--bad-soft)', color: 'var(--bad)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', marginBottom: 14 }}>
            <Icon name="warning" size={20} />
          </div>
          <div className="t-title" style={{ marginBottom: 8 }}>訓練失敗</div>
          <div className="t-body fg-2" style={{ marginBottom: 16 }}>{exp.error}</div>
          <Button variant="ghost" icon="copy">複製設定到新實驗</Button>
        </div>
      </Surface>
    );
  }

  if (exp.status === 'running') {
    return <ExpRunningView exp={exp} />;
  }

  // exp.best 可能是 null (run 沒模型),要 defensive
  const best = exp.best || { metric: '—', value: '—', algo: '—' };
  // 從 myModels 排 top 5 (依 testScore desc)
  const top5 = React.useMemo(() => {
    return (exp.myModels || []).map(m => {
      const b = m.bundle || {}, mt = b.metrics || {};
      return {
        id: m.id,
        algo: (b.name || b.type || '?').replace(/^\[(原始|預處理)\]\s*/, ''),
        source: b.dataSource === 'preprocessed' ? '預處理' : '原始',
        f1: typeof mt.f1 === 'number' ? mt.f1 : null,
        auc: typeof mt.auc === 'number' ? mt.auc : null,
        acc: typeof mt.testAccuracy === 'number' ? mt.testAccuracy : null,
        testScore: mt.testScore,
        trainTime: (b.trainTime || 0) / 1000,
      };
    }).sort((a, b) => (b.testScore || 0) - (a.testScore || 0)).slice(0, 5);
  }, [exp.myModels]);

  return (
    <div>
      {/* Top metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
        <Surface style={{ padding: 16, borderColor: 'var(--primary-line)', background: 'linear-gradient(180deg, var(--primary-soft) 0%, var(--bg-surface) 60%)' }}>
          <div className="t-label">最佳 {best.metric}</div>
          <div className="t-metric" style={{ marginTop: 6 }}>{best.value}</div>
          <div className="t-label mono" style={{ marginTop: 6, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{best.algo}</div>
        </Surface>
        <Surface style={{ padding: 16 }}>
          <div className="t-label">已訓練模型</div>
          <div className="t-metric" style={{ marginTop: 6 }}>{exp.models}</div>
          <div className="t-label" style={{ marginTop: 6 }}>engine: {exp.engine || '?'}</div>
        </Surface>
        <Surface style={{ padding: 16 }}>
          <div className="t-label">總耗時</div>
          <div className="t-metric" style={{ marginTop: 6 }}>{exp.duration}</div>
          <div className="t-label" style={{ marginTop: 6 }}>實際訓練時長</div>
        </Surface>
        <Surface style={{ padding: 16 }}>
          <div className="t-label">啟動時間</div>
          <div className="t-label mono" style={{ marginTop: 6, fontSize: 18, fontWeight: 600, color: 'var(--fg-strong)' }}>
            {exp.startedAtSec ? new Date(exp.startedAtSec * 1000).toLocaleString('zh-TW', { hour12: false }) : '?'}
          </div>
          <div className="t-label" style={{ marginTop: 6 }}>{exp.startedAt}</div>
        </Surface>
      </div>

      {/* Top 5 models table */}
      <Surface>
        <CardHeader title="Top 5 模型" subtitle={`按 ${top5[0]?.testScore != null ? '分數' : 'F1'} 排名`}
                    right={<Button variant="bare" size="sm" iconRight="arrowRight" onClick={() => onNavigate('models')}>看全部排行榜</Button>} />
        {top5.length === 0
          ? <div style={{ padding: 24, textAlign: 'center', color: 'var(--fg-muted)' }}>
              <span className="t-label">此實驗沒有模型 (失敗 or 仍在運行)</span>
            </div>
          : <table className="tbl">
              <thead>
                <tr>
                  <th>#</th>
                  <th>演算法</th>
                  <th>來源</th>
                  <th style={{ textAlign: 'right' }}>F1</th>
                  <th style={{ textAlign: 'right' }}>AUC</th>
                  <th style={{ textAlign: 'right' }}>Accuracy</th>
                  <th style={{ textAlign: 'right' }}>訓練 (s)</th>
                </tr>
              </thead>
              <tbody>
                {top5.map((m, i) => (
                  <tr key={m.id}>
                    <td className="mono fg-3">{i + 1}</td>
                    <td className="fg-1">{m.algo}</td>
                    <td><Chip className="mono">{m.source}</Chip></td>
                    <td className="mono fg-1" style={{ textAlign: 'right' }}>{m.f1 != null ? m.f1.toFixed(3) : '—'}</td>
                    <td className="mono" style={{ textAlign: 'right' }}>{m.auc != null ? m.auc.toFixed(3) : '—'}</td>
                    <td className="mono" style={{ textAlign: 'right' }}>{m.acc != null ? m.acc.toFixed(3) : '—'}</td>
                    <td className="mono fg-3" style={{ textAlign: 'right' }}>{m.trainTime > 0 ? m.trainTime.toFixed(1) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
        }
      </Surface>
    </div>
  );
}

// 真實的即時訓練面板 — 每 3 秒 poll /api/training-runs/{id}/progress
// 後端 SSE 旁路把 progress / log / last_seen_at 寫進 DB,這裡只負責顯示
function ExpRunningView({ exp }) {
  const [progress, setProgress] = React.useState({
    progressPct: 0, currentStep: '等待後端回報...', latestLog: [],
    lastSeenAt: null, status: 'running', errorMsg: null,
  });
  const [pollError, setPollError] = React.useState(null);

  React.useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const data = await NewUI.api.get(`/api/training-runs/${encodeURIComponent(exp.id)}/progress`);
        if (!alive) return;
        setProgress(data);
        setPollError(null);
        // 跑完或失敗就停 polling
        if (data.status !== 'running') {
          clearInterval(timer);
        }
      } catch (e) {
        if (!alive) return;
        setPollError(e.message || 'poll failed');
      }
    };
    poll();   // 立刻第一次
    const timer = setInterval(poll, 3000);
    return () => { alive = false; clearInterval(timer); };
  }, [exp.id]);

  // 推算 staleness — 60 秒沒新訊號 = 可疑,5 分鐘 = 大概死了
  const staleness = React.useMemo(() => {
    if (!progress.lastSeenAt) return null;
    const ageSec = Math.floor(Date.now() / 1000 - progress.lastSeenAt);
    if (ageSec > 300) return { level: 'bad', ageSec, msg: '超過 5 分鐘沒收到後端進度,可能已死掉' };
    if (ageSec > 60)  return { level: 'warn', ageSec, msg: `${ageSec} 秒沒收到後端進度,可能卡住或暫停` };
    return { level: 'good', ageSec, msg: `${ageSec} 秒前更新` };
  }, [progress.lastSeenAt]);

  const log = progress.latestLog || [];
  const pct = progress.progressPct || 0;

  return (
    <div>
      {/* Top status bar */}
      <Surface style={{ marginBottom: 12 }}>
        <div style={{ padding: 16 }}>
          <Row style={{ justifyContent: 'space-between', marginBottom: 12 }}>
            <Row gap={10}>
              <Dot tone={progress.status === 'completed' ? 'good' : progress.status === 'failed' ? 'bad' : 'active'} />
              <span className="t-title">
                {progress.status === 'running' ? '執行中' : progress.status === 'completed' ? '已完成' : progress.status === 'failed' ? '失敗' : progress.status}
              </span>
              <span className="t-label mono">{progress.currentStep || '?'}</span>
            </Row>
            <Row gap={8}>
              {staleness && (
                <Chip tone={staleness.level}>{staleness.msg}</Chip>
              )}
              {pollError && (
                <Chip tone="warn">poll 失敗: {pollError}</Chip>
              )}
            </Row>
          </Row>
          <div style={{ height: 4, background: 'var(--bg-sunken)', borderRadius: 2, overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${pct}%`, background: 'var(--primary)', transition: 'width .4s' }} />
          </div>
          <Row style={{ marginTop: 6, justifyContent: 'space-between' }}>
            <span className="t-label mono">{pct}% 完成</span>
            <span className="t-label mono">
              {progress.startedAt && `啟動 ${_fmtTimeAgo(progress.startedAt)} 前`}
            </span>
          </Row>
        </div>
        {progress.errorMsg && (
          <div className="divider-t" style={{ padding: '12px 16px', background: 'color-mix(in srgb, var(--bad) 8%, transparent)' }}>
            <Row gap={8}>
              <Icon name="warning" size={14} style={{ color: 'var(--bad)' }} />
              <span className="t-label" style={{ color: 'var(--bad)' }}>{progress.errorMsg}</span>
            </Row>
          </div>
        )}
      </Surface>

      {/* Live log */}
      <Surface>
        <CardHeader title="即時 log"
                    subtitle={`最近 ${log.length} 行 (後端旁路寫入,每 ~2s 更新一次)`}
                    right={<span className="t-label mono">poll 3s</span>} />
        <div style={{ padding: 12, maxHeight: 380, minHeight: 200, overflow: 'auto',
                       background: 'var(--bg-sunken)', fontFamily: 'JetBrains Mono, monospace',
                       fontSize: 11, lineHeight: 1.6 }}>
          {log.length === 0 ? (
            <div style={{ padding: 16, color: 'var(--fg-faint)', textAlign: 'center' }}>
              還沒有 log — 訓練可能剛啟動或這個分頁不是訓練啟動的分頁
            </div>
          ) : log.map((line, i) => {
            const level = line.level || 'info';
            const color = level === 'error' ? 'var(--bad)'
                        : level === 'warning' ? 'var(--warn)'
                        : level === 'success' ? 'var(--good)'
                        : level === 'muted' ? 'var(--fg-faint)'
                        : 'var(--fg-default)';
            return (
              <div key={i} style={{ color, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                {line.msg}
              </div>
            );
          })}
        </div>
      </Surface>

      <div style={{ marginTop: 12, padding: 12, textAlign: 'center', color: 'var(--fg-muted)' }}>
        <span className="t-label">
          提示:即時 log 從 DB 拉,所以 1) 跨分頁/裝置都看得到 2) 後端 throttle 每 2 秒寫一次,有延遲
        </span>
      </div>
    </div>
  );
}

function _fmtTimeAgo(sec) {
  if (!sec) return '?';
  const diff = Math.floor(Date.now() / 1000 - sec);
  if (diff < 60)   return `${diff}s`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h`;
  return `${Math.floor(diff / 86400)}d`;
}

function ExpModelsList({ exp, onNavigate }) {
  // exp.myModels 是從 run 篩出來的 raw model 物件,正規化成顯示用 shape 再排
  const list = React.useMemo(() => {
    return (exp.myModels || []).map(m => {
      const b = m.bundle || {}, mt = b.metrics || {};
      return {
        id: m.id,
        algo: (b.name || b.type || '?').replace(/^\[(原始|預處理)\]\s*/, ''),
        source: b.dataSource === 'preprocessed' ? '預處理' : '原始',
        f1: typeof mt.f1 === 'number' ? mt.f1 : null,
        auc: typeof mt.auc === 'number' ? mt.auc : null,
        acc: typeof mt.testAccuracy === 'number' ? mt.testAccuracy : null,
        testScore: mt.testScore,
        trainTime: (b.trainTime || 0) / 1000,
        inferLatency: b.inferLatency || 0,
      };
    }).sort((a, b) => (b.testScore || 0) - (a.testScore || 0));
  }, [exp.myModels]);

  function analyze(modelId) {
    try { sessionStorage.setItem('newui_focus_model_id', modelId); } catch (e) {}
    onNavigate('insights');
  }

  return (
    <Surface>
      <CardHeader title="所有模型" subtitle={`${list.length} 個模型`}
                  right={<Button variant="primary" size="sm" iconRight="arrowRight" onClick={() => onNavigate('models')}>排行榜檢視</Button>} />
      {list.length === 0 ? (
        <div style={{ padding: 24, textAlign: 'center', color: 'var(--fg-muted)' }}>
          <span className="t-label">此實驗沒有可顯示的模型</span>
        </div>
      ) : (
        <div style={{ overflow: 'auto' }}>
          <table className="tbl">
            <thead>
              <tr>
                <th>#</th>
                <th>演算法</th>
                <th>來源</th>
                <th style={{ textAlign: 'right' }}>F1</th>
                <th style={{ textAlign: 'right' }}>AUC</th>
                <th style={{ textAlign: 'right' }}>Accuracy</th>
                <th style={{ textAlign: 'right' }}>訓練 (s)</th>
                <th style={{ textAlign: 'right' }}>推論 (ms)</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {list.map((m, i) => (
                <tr key={m.id}>
                  <td className="mono fg-3">{i + 1}</td>
                  <td className="fg-1">{m.algo}</td>
                  <td><Chip className="mono">{m.source}</Chip></td>
                  <td className="mono fg-1" style={{ textAlign: 'right' }}>{m.f1 != null ? m.f1.toFixed(3) : '—'}</td>
                  <td className="mono" style={{ textAlign: 'right' }}>{m.auc != null ? m.auc.toFixed(3) : '—'}</td>
                  <td className="mono" style={{ textAlign: 'right' }}>{m.acc != null ? m.acc.toFixed(3) : '—'}</td>
                  <td className="mono fg-3" style={{ textAlign: 'right' }}>{m.trainTime > 0 ? m.trainTime.toFixed(1) : '—'}</td>
                  <td className="mono fg-3" style={{ textAlign: 'right' }}>{m.inferLatency > 0 ? m.inferLatency.toFixed(2) : '—'}</td>
                  <td>
                    <button className="btn btn-ghost btn-sm" onClick={() => analyze(m.id)}>分析</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Surface>
  );
}

function ExpLogs({ exp }) {
  return (
    <Surface>
      <CardHeader title="訓練 log" subtitle={`${exp.id}`} right={<Button variant="ghost" size="sm" icon="download">下載完整 log</Button>} />
      <div style={{ padding: 12, fontFamily: 'JetBrains Mono, monospace', fontSize: 11, lineHeight: 1.7, background: 'var(--bg-sunken)' }}>
        <div className="fg-3">[14:02:01] preprocess: split 80/20 (random_state=42)</div>
        <div className="fg-3">[14:02:02] preprocess: 9,960 train · 2,490 test</div>
        <div className="fg-3">[14:02:03] feature engineering: 14 → 28 columns (onehot)</div>
        <div className="fg-3">[14:02:05] feature scaling: StandardScaler fit on train</div>
        <div className="fg-2">[14:02:10] HPO: starting 100 trials (Optuna TPE)</div>
        <div className="fg-2">[14:02:11] HPO: pool = [xgb, lgbm, cat, rf, lr]</div>
        <div className="fg-2">[14:02:14] trial #1 · XGBoost · mean F1 = 0.917</div>
        <div className="fg-2">[14:02:17] trial #2 · LightGBM · mean F1 = 0.921</div>
        <div className="fg-2">[14:02:20] trial #3 · CatBoost · mean F1 = 0.918</div>
        <div style={{ color: 'var(--good)' }}>[14:02:34] ★ new best at trial #12 · LightGBM · F1 = 0.935</div>
        <div className="fg-3">...</div>
        <div style={{ color: 'var(--good)' }}>[14:09:52] ★ new best at trial #87 · XGBoost · F1 = 0.942</div>
        <div className="fg-2">[14:10:11] HPO done. picked: XGBoost</div>
        <div className="fg-2">[14:10:11] final fit on full train</div>
        <div className="fg-2">[14:10:13] final eval on test set</div>
        <div className="fg-1">[14:10:13] final F1 = 0.942 · AUC = 0.961 · Acc = 0.928</div>
        <div style={{ color: 'var(--good)' }}>[14:10:13] ✓ done</div>
      </div>
    </Surface>
  );
}

function ExpConfig({ exp }) {
  const opts = exp.options || {};
  const sections = [
    { title: '資料', rows: [
      ['Dataset', exp.dataset || '?'],
      ['Target', exp.target || '?'],
      ['任務類型', exp.task === 'regression' ? '回歸' : '分類'],
      ['Test size', opts.testSize != null ? `${(opts.testSize * 100).toFixed(0)}%` : '預設 20%'],
      ['Random state', opts.randomState ?? '預設 42'],
      ['資料來源', (exp.sources || []).join(' + ') || '?'],
    ]},
    { title: '訓練', rows: [
      ['Engine', exp.engine || 'sklearn'],
      ['Task auto-detect', opts.taskType === 'auto' || !opts.taskType ? '是' : '否'],
      ['時序模式', opts.timeSeries ? '是' : '否'],
      ['Metric', opts.metric || 'F1 (macro)'],
      ['HPO trials', opts.hpoTrials ?? '預設'],
      ['Fast mode', opts.fast ? '是' : '否'],
      ['跳過 DL', opts.skipDl ? '是' : '否'],
      ['跳過 NAS', opts.noNas ? '是' : '否'],
    ]},
    { title: '狀態', rows: [
      ['Run ID', exp.id],
      ['狀態', exp.status],
      ['開始', exp.startedAt],
      ['耗時', exp.duration],
    ]},
  ];
  return (
    <Stack gap={12}>
      {sections.map(s => (
        <Surface key={s.title}>
          <CardHeader title={s.title} />
          <div style={{ padding: '4px 0' }}>
            {s.rows.map(([k, v], i) => (
              <Row key={k} style={{ padding: '10px 16px', borderBottom: i < s.rows.length - 1 ? '1px solid var(--bd-subtle)' : 'none' }}>
                <span className="t-label" style={{ width: 160 }}>{k}</span>
                <span className="t-body-lg mono fg-1" style={{ flex: 1 }}>{v}</span>
              </Row>
            ))}
          </div>
        </Surface>
      ))}
    </Stack>
  );
}

// =====================
// New Experiment (cleaned-up form)
// =====================
function NewExperiment({ onCancel, onCreated }) {
  const [dataset, setDataset] = React.useState('churn');
  const [target, setTarget] = React.useState('churn');
  const [taskType, setTaskType] = React.useState('auto');
  const [engine, setEngine] = React.useState('sklearn');
  const [openAdvanced, setOpenAdvanced] = React.useState(false);
  const [algos, setAlgos] = React.useState(MOCK.algorithms.filter(a => a.enabled).map(a => a.id));
  const [testSize, setTestSize] = React.useState(20);
  const [hpoTrials, setHpoTrials] = React.useState(100);
  const [name, setName] = React.useState('客戶流失預測 v4');
  const [featureModalOpen, setFeatureModalOpen] = React.useState(false);
  const ds = MOCK.datasets.find(d => d.id === dataset);
  const featureCols = MOCK.columns.filter(c => c.role !== 'id' && c.name !== target);
  const [selectedFeatures, setSelectedFeatures] = React.useState(() => featureCols.map(c => c.name));

  // ----- Pipeline (Daniel) engine state -----
  const [pipelineFast, setPipelineFast] = React.useState(true);
  const [pipelineSkipDL, setPipelineSkipDL] = React.useState(false);
  const [pipelineSkipNAS, setPipelineSkipNAS] = React.useState(false);
  const [pipelineTimeSeries, setPipelineTimeSeries] = React.useState(false);
  const [pipelineMetric, setPipelineMetric] = React.useState('f1');
  const [pipelineTimeLimit, setPipelineTimeLimit] = React.useState(0);
  const [pipelinePredictFile, setPipelinePredictFile] = React.useState(null);

  function submit() { onCreated('exp-13'); }

  return (
    <div style={{ padding: 24 }}>
      <Row style={{ justifyContent: 'space-between', marginBottom: 16 }}>
        <div>
          <h1 className="t-h1">新實驗</h1>
          <p className="t-label" style={{ marginTop: 4 }}>填三個必要欄位就能開跑,進階設定都有合理預設</p>
        </div>
        <Button variant="ghost" onClick={onCancel}>取消</Button>
      </Row>

      <Stack gap={12} style={{ maxWidth: 880 }}>
        {/* ----- Essentials ----- */}
        <Surface>
          <CardHeader title="基本設定" subtitle="必填" />
          <div style={{ padding: 16, display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
            <Field label="實驗名稱">
              <input className="input" value={name} onChange={e => setName(e.target.value)} />
            </Field>
            <Field label="數據集">
              <select className="input" value={dataset} onChange={e => setDataset(e.target.value)}>
                {MOCK.datasets.map(d => (
                  <option key={d.id} value={d.id}>{d.name} ({d.rows.toLocaleString()} 列)</option>
                ))}
              </select>
            </Field>
            <Field label="目標欄位 (target)" hint={`偵測為「${ds.task === 'classification' ? '分類' : '回歸'}」任務`}>
              <select className="input" value={target} onChange={e => setTarget(e.target.value)}>
                {MOCK.columns.map(c => (
                  <option key={c.name} value={c.name}>{c.name} ({c.type})</option>
                ))}
              </select>
            </Field>
            <Field label="任務類型">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6 }}>
                {[
                  { id: 'auto', label: '自動' },
                  { id: 'classification', label: '分類' },
                  { id: 'regression', label: '回歸' },
                ].map(opt => (
                  <button key={opt.id} onClick={() => setTaskType(opt.id)}
                          className={taskType === opt.id ? 'btn btn-primary btn-sm' : 'btn btn-ghost btn-sm'}>
                    {opt.label}
                  </button>
                ))}
              </div>
            </Field>
          </div>
        </Surface>

        {/* ----- Engine selector — segmented control ----- */}
        <Surface>
          <CardHeader title="訓練引擎" subtitle="決定要跑哪一條 pipeline" />
          <div style={{ padding: 16, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            {[
              { id: 'sklearn', name: 'sklearn (預設)', body: '同時跑多個演算法 · HPO · 秒~分鐘級', recommended: true },
              { id: 'pipeline', name: 'pipeline (深度)', body: 'HPO + NAS + 5-Fold CV + Stacking · 數分鐘起', warning: '僅分類 · 需 torch+optuna' },
            ].map(opt => (
              <button key={opt.id} onClick={() => setEngine(opt.id)}
                      style={{
                        textAlign: 'left', padding: 14, borderRadius: 8, cursor: 'pointer',
                        background: engine === opt.id ? 'var(--primary-soft)' : 'var(--bg-sunken)',
                        border: '1px solid ' + (engine === opt.id ? 'var(--primary-line)' : 'var(--bd-subtle)'),
                      }}>
                <Row style={{ justifyContent: 'space-between', marginBottom: 6 }}>
                  <Row gap={6}>
                    <span style={{
                      width: 14, height: 14, borderRadius: '50%',
                      border: '2px solid ' + (engine === opt.id ? 'var(--primary)' : 'var(--bd-default)'),
                      background: engine === opt.id ? 'var(--primary)' : 'transparent',
                      boxShadow: engine === opt.id ? 'inset 0 0 0 2px var(--bg-canvas)' : 'none',
                    }} />
                    <span className="t-body-lg fg-1" style={{ fontWeight: 500 }}>{opt.name}</span>
                  </Row>
                  {opt.recommended && <Chip tone="primary" className="mono">建議</Chip>}
                  {opt.warning && <Chip tone="warn" className="mono">注意</Chip>}
                </Row>
                <div className="t-label" style={{ marginLeft: 20 }}>{opt.body}</div>
                {opt.warning && <div className="t-label" style={{ marginLeft: 20, marginTop: 4, color: 'var(--warn)' }}>{opt.warning}</div>}
              </button>
            ))}
          </div>
        </Surface>

        {/* ----- Pipeline engine options (only when engine === 'pipeline') ----- */}
        {engine === 'pipeline' && (
          <Surface style={{ borderColor: 'var(--warn)', borderLeftWidth: 2 }}>
            <CardHeader
              title="Pipeline 進階設定"
              subtitle="HPO + NAS + 5-fold CV + Stacking · 預設 fast 模式縮減 trials"
              right={<Chip tone="warn">僅分類 · 需 torch+optuna</Chip>}
            />
            <div style={{ padding: 16, display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
              <ToggleCard label="快速模式"   hint="縮減 HPO trials"   checked={pipelineFast}      onChange={setPipelineFast} />
              <ToggleCard label="跳過 DL"   hint="只跑傳統 ML"      checked={pipelineSkipDL}    onChange={setPipelineSkipDL} />
              <ToggleCard label="跳過 NAS"  hint="用預設架構"        checked={pipelineSkipNAS}   onChange={setPipelineSkipNAS} />
              <ToggleCard label="時序模式"  hint="TCN / PatchTST"   checked={pipelineTimeSeries} onChange={setPipelineTimeSeries} />
            </div>
            <div style={{ padding: 16, paddingTop: 0, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <Field label="最佳化指標" hint="CV 內部評分">
                <select className="input" value={pipelineMetric} onChange={e => setPipelineMetric(e.target.value)}>
                  <option value="f1">F1 (macro)</option>
                  <option value="accuracy">Accuracy</option>
                </select>
              </Field>
              <Field label="時間上限 (秒)" hint="0 = 不限制 · Render free tier 建議 600 內">
                <input type="number" className="input mono" value={pipelineTimeLimit} min={0} step={60}
                       onChange={e => setPipelineTimeLimit(parseInt(e.target.value) || 0)} />
              </Field>
            </div>
            {/* Predict CSV uploader */}
            <div className="divider-t" style={{ padding: 16 }}>
              <Row style={{ justifyContent: 'space-between', gap: 16 }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="t-body-lg fg-1">📄 預測 CSV (可選)</div>
                  <div className="t-label" style={{ marginTop: 3 }}>
                    訓練完成後將用 pipeline 對此檔案做預測,結果可直接下載成 <span className="mono">submission.csv</span>
                  </div>
                </div>
                <Row gap={6} style={{ flexShrink: 0 }}>
                  {pipelinePredictFile ? (
                    <Chip className="mono">{pipelinePredictFile} <button onClick={() => setPipelinePredictFile(null)} style={{ background: 'transparent', border: 'none', color: 'var(--fg-muted)', cursor: 'pointer' }}>✕</button></Chip>
                  ) : (
                    <Button variant="ghost" size="sm" icon="upload" onClick={() => setPipelinePredictFile('test.csv')}>選擇檔案</Button>
                  )}
                </Row>
              </Row>
            </div>
          </Surface>
        )}

        {/* ----- Algorithms (sklearn only) ----- */}
        {engine === 'sklearn' && (
        <Surface>
          <CardHeader title="演算法" subtitle={`勾選要跑的模型 · 已選 ${algos.length} / ${MOCK.algorithms.length}`}
                      right={
                        <Row gap={6}>
                          <Button variant="bare" size="sm" onClick={() => setAlgos(MOCK.algorithms.map(a => a.id))}>全選</Button>
                          <Button variant="bare" size="sm" onClick={() => setAlgos(['xgb', 'lgbm'])}>建議</Button>
                          <Button variant="bare" size="sm" onClick={() => setAlgos([])}>清除</Button>
                        </Row>
                      } />
          <div style={{ padding: 14, display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
            {MOCK.algorithms.map(a => (
              <label key={a.id} style={{
                padding: '8px 10px',
                background: algos.includes(a.id) ? 'var(--primary-soft)' : 'var(--bg-sunken)',
                border: '1px solid ' + (algos.includes(a.id) ? 'var(--primary-line)' : 'var(--bd-subtle)'),
                borderRadius: 6, cursor: 'pointer',
                display: 'flex', alignItems: 'center', gap: 8,
              }}>
                <input type="checkbox" checked={algos.includes(a.id)}
                       onChange={e => setAlgos(prev => e.target.checked ? [...prev, a.id] : prev.filter(x => x !== a.id))}
                       style={{ accentColor: 'var(--primary)' }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="t-body fg-1">{a.name}</div>
                  <div className="t-label mono" style={{ fontSize: 10 }}>{a.kind}</div>
                </div>
              </label>
            ))}
          </div>
        </Surface>
        )}

        {/* ----- Advanced (collapsible) ----- */}
        <Surface>
          <button onClick={() => setOpenAdvanced(o => !o)} style={{
            width: '100%', padding: '14px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--fg-default)',
          }}>
            <div style={{ textAlign: 'left' }}>
              <div className="t-title">進階設定</div>
              <div className="t-label" style={{ marginTop: 2 }}>切分比例 · HPO trials · CV · 特徵欄位</div>
            </div>
            <Icon name={openAdvanced ? 'chevronUp' : 'chevronDown'} size={16} />
          </button>
          {openAdvanced && (
            <div className="divider-t" style={{ padding: 16, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <Field label="測試集比例" hint={`${100 - testSize}% / ${testSize}% · random_state=42`}>
                <Range value={testSize} onChange={setTestSize} min={10} max={50} step={5} format={v => v + '%'} />
              </Field>
              <Field label="HPO trials" hint="Optuna TPE">
                <Range value={hpoTrials} onChange={setHpoTrials} min={20} max={300} step={20} />
              </Field>
              <Field label="CV folds">
                <select className="input"><option>5 (預設)</option><option>3</option><option>10</option></select>
              </Field>
              <Field label="最佳化指標">
                <select className="input"><option>F1 (macro)</option><option>Accuracy</option><option>AUC</option></select>
              </Field>
              <Field label={`特徵欄位 (${selectedFeatures.length} / ${featureCols.length})`} hint="點擊以選擇/取消特徵">
                <Button variant="ghost" size="sm" iconRight="chevronRight" onClick={() => setFeatureModalOpen(true)}
                        style={{ justifyContent: 'space-between', width: '100%' }}>
                  {selectedFeatures.length === featureCols.length ? `全部 ${featureCols.length} 個欄位` : `已選 ${selectedFeatures.length} / ${featureCols.length}`}
                </Button>
              </Field>
              <Field label="時間序列">
                <Row gap={10}>
                  <Toggle checked={false} onChange={() => {}} />
                  <span className="t-label">啟用後依時間順序切分</span>
                </Row>
              </Field>
            </div>
          )}
        </Surface>

        {/* ----- Footer actions ----- */}
        <Row style={{ justifyContent: 'space-between', alignItems: 'center', paddingTop: 4 }}>
          <span className="t-label">
            預估耗時 <span className="mono fg-1">{engine === 'pipeline' ? (pipelineFast ? '~12 分鐘' : '~45 分鐘') : '~6 分鐘'}</span>
            <span className="fg-4 mono"> · </span>
            {engine === 'pipeline'
              ? <>使用 {selectedFeatures.length} 個特徵 · pipeline 自動選</>
              : <>預估 {algos.length * 20} 個 trial</>
            }
          </span>
          <Row gap={8}>
            <Button variant="ghost" onClick={onCancel}>取消</Button>
            <Button variant="primary" iconRight="play" onClick={submit}>開始訓練</Button>
          </Row>
        </Row>
      </Stack>

      <FeatureColumnModal
        open={featureModalOpen}
        onClose={() => setFeatureModalOpen(false)}
        columns={featureCols}
        selected={selectedFeatures}
        onChange={setSelectedFeatures}
      />
    </div>
  );
}

// Toggle card used for the pipeline boolean options
function ToggleCard({ label, hint, checked, onChange }) {
  return (
    <label style={{
      padding: 10, borderRadius: 7, cursor: 'pointer',
      background: checked ? 'var(--primary-soft)' : 'var(--bg-sunken)',
      border: '1px solid ' + (checked ? 'var(--primary-line)' : 'var(--bd-subtle)'),
      display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8,
    }}>
      <div style={{ minWidth: 0 }}>
        <div className="t-body fg-1">{label}</div>
        <div className="t-label" style={{ marginTop: 2, fontSize: 10 }}>{hint}</div>
      </div>
      <Toggle checked={checked} onChange={onChange} />
    </label>
  );
}

// ------- Feature column multi-select modal -------
function FeatureColumnModal({ open, onClose, columns, selected, onChange }) {
  const [query, setQuery] = React.useState('');
  const filtered = columns.filter(c => c.name.toLowerCase().includes(query.toLowerCase()));

  function toggle(name) {
    onChange(selected.includes(name) ? selected.filter(x => x !== name) : [...selected, name]);
  }
  function selectAll() { onChange(columns.map(c => c.name)); }
  function clearAll() { onChange([]); }
  function selectByType(type) {
    if (type === 'numeric') onChange(columns.filter(c => c.type === 'int' || c.type === 'float').map(c => c.name));
    else if (type === 'categorical') onChange(columns.filter(c => c.type === 'cat').map(c => c.name));
    else if (type === 'date') onChange(columns.filter(c => c.type === 'date').map(c => c.name));
  }

  return (
    <Modal open={open} onClose={onClose} title="選擇特徵欄位" width={680}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>取消</Button>
          <Button variant="primary" onClick={onClose}>確定 ({selected.length} 個)</Button>
        </>
      }>
      <Row gap={8} style={{ marginBottom: 12 }}>
        <input className="input" placeholder="搜尋欄位..." value={query} onChange={e => setQuery(e.target.value)} style={{ flex: 1 }} />
      </Row>
      <Row gap={4} style={{ marginBottom: 14, flexWrap: 'wrap' }}>
        <Button variant="ghost" size="sm" onClick={selectAll}>全選</Button>
        <Button variant="ghost" size="sm" onClick={clearAll}>清除</Button>
        <span style={{ width: 1, background: 'var(--bd-subtle)', alignSelf: 'stretch' }} />
        <Button variant="ghost" size="sm" onClick={() => selectByType('numeric')}>僅數值</Button>
        <Button variant="ghost" size="sm" onClick={() => selectByType('categorical')}>僅類別</Button>
        <Button variant="ghost" size="sm" onClick={() => selectByType('date')}>僅日期</Button>
      </Row>

      <div style={{ background: 'var(--bg-sunken)', borderRadius: 7, border: '1px solid var(--bd-subtle)', maxHeight: 380, overflow: 'auto' }}>
        <table className="tbl" style={{ fontSize: 13 }}>
          <thead>
            <tr>
              <th style={{ width: 36 }}>
                <input type="checkbox"
                       checked={selected.length === columns.length}
                       ref={el => { if (el) el.indeterminate = selected.length > 0 && selected.length < columns.length; }}
                       onChange={e => e.target.checked ? selectAll() : clearAll()}
                       style={{ accentColor: 'var(--primary)' }} />
              </th>
              <th>欄位</th>
              <th>類型</th>
              <th style={{ textAlign: 'right' }}>缺失</th>
              <th style={{ textAlign: 'right' }}>唯一值</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(c => (
              <tr key={c.name} style={{ cursor: 'pointer' }} onClick={() => toggle(c.name)}
                  className={selected.includes(c.name) ? 'selected' : ''}>
                <td onClick={e => e.stopPropagation()}>
                  <input type="checkbox" checked={selected.includes(c.name)} onChange={() => toggle(c.name)}
                         style={{ accentColor: 'var(--primary)' }} />
                </td>
                <td className="mono fg-1">{c.name}</td>
                <td><Chip>{c.type}</Chip></td>
                <td className="mono" style={{ textAlign: 'right', color: c.missing > 0 ? 'var(--warn)' : 'var(--fg-default)' }}>
                  {c.missing > 0 ? c.missing : '0'}
                </td>
                <td className="mono fg-3" style={{ textAlign: 'right' }}>{c.unique.toLocaleString()}</td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr><td colSpan={5} style={{ padding: 20, textAlign: 'center', color: 'var(--fg-muted)' }}>沒有符合的欄位</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <Row style={{ marginTop: 12, justifyContent: 'space-between' }}>
        <span className="t-label">共 {columns.length} 個欄位 · 顯示 {filtered.length} 筆</span>
        <span className="t-label mono">已選 <span className="fg-1">{selected.length}</span></span>
      </Row>
    </Modal>
  );
}

function Field({ label, hint, children }) {
  return (
    <div>
      <label className="t-label" style={{ display: 'block', marginBottom: 6, color: 'var(--fg-default)' }}>{label}</label>
      {children}
      {hint && <div className="t-label" style={{ marginTop: 6, fontSize: 10 }}>{hint}</div>}
    </div>
  );
}

window.PageExperiments = PageExperiments;
