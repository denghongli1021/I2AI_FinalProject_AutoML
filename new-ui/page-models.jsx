// =============================================================================
// Page: Models — 接後端 /api/models + /api/training-runs
// =============================================================================

function PageModels({ onNavigate }) {
  const [sortBy, setSortBy] = React.useState({ key: 'f1', dir: 'desc' });
  const [selected, setSelected] = React.useState([]);
  const [filter, setFilter] = React.useState('');
  const [runFilter, setRunFilter] = React.useState('all');
  const [loading, setLoading] = React.useState(true);
  const [error, setError]     = React.useState(null);
  const [models, setModels]   = React.useState([]);
  const [runs, setRuns]       = React.useState([]);

  React.useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      NewUI.api.getCached('/api/models?limit=500').then(r => r.models || []).catch(e => { console.warn('models fail', e); return []; }),
      NewUI.api.getCached('/api/training-runs?limit=100').then(r => r.runs || []).catch(e => { console.warn('runs fail', e); return []; }),
    ]).then(([rModels, rRuns]) => {
      setModels(rModels);
      setRuns(rRuns);
      setLoading(false);
    }).catch(err => {
      setError(err.message || '載入失敗');
      setLoading(false);
    });
  }, []);

  // ---- 攤平 / 正規化 ----
  // models 來自後端 (m.bundle / m.preprocessorId / m.datasetId)
  // 我們把 bundle 內容拉到頂層,並依分數 desc 排 rank,讓 sort / display 簡單
  const normalized = React.useMemo(() => {
    return models.map(m => {
      const b = m.bundle || {};
      const metrics = b.metrics || {};
      return {
        id: m.id,
        algo: (b.name || b.type || '?').replace(/^\[(原始|預處理)\]\s*/, ''),
        rawName: b.name || b.type || '?',
        source: b.dataSource === 'preprocessed' ? '預處理'
              : b.dataSource === 'raw' ? '原始'
              : (b.dataSource || '?'),
        f1:   typeof metrics.f1 === 'number' ? metrics.f1 : null,
        auc:  typeof metrics.auc === 'number' ? metrics.auc : null,
        acc:  typeof metrics.testAccuracy === 'number' ? metrics.testAccuracy
            : typeof metrics.accuracy === 'number' ? metrics.accuracy : null,
        testScore: typeof metrics.testScore === 'number' ? metrics.testScore : null,
        testScoreLabel: metrics.testScoreLabel || (b.taskType === 'regression' ? 'R²' : 'Acc'),
        trainTime: (b.trainTime || 0) / 1000,  // ms → s
        inferLatency: b.inferLatency || 0,     // already ms per sample
        tag: metrics.tag || (b.taskType === 'regression' ? null : null),
        taskType: b.taskType || 'classification',
        dataSource: b.dataSource,
        trainingRunId: m.trainingRunId,
        datasetId: m.datasetId,
        bundle: b,
      };
    });
  }, [models]);

  // ---- 依 run filter + algo filter 篩選 ----
  const filtered = React.useMemo(() => {
    return normalized.filter(m => {
      if (runFilter !== 'all' && m.trainingRunId !== runFilter) return false;
      if (filter && !m.algo.toLowerCase().includes(filter.toLowerCase())) return false;
      return true;
    });
  }, [normalized, runFilter, filter]);

  // ---- 排序 ----
  const sorted = React.useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      const av = a[sortBy.key], bv = b[sortBy.key];
      const sign = sortBy.dir === 'asc' ? 1 : -1;
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === 'string') return av.localeCompare(bv) * sign;
      return (av - bv) * sign;
    });
    return arr;
  }, [filtered, sortBy]);

  function sortClick(k) {
    setSortBy(s => s.key === k ? { key: k, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key: k, dir: 'desc' });
  }

  function toggle(id) {
    setSelected(s => s.includes(id) ? s.filter(x => x !== id) : [...s, id]);
  }

  // 「分析」按鈕 → 把 modelId 寫進 sessionStorage,洞察頁讀
  function analyzeModel(modelId) {
    try { sessionStorage.setItem('newui_focus_model_id', modelId); } catch (e) {}
    onNavigate('insights');
  }

  function thSort(k, label, align) {
    const cls = ['tbl-sort', sortBy.key === k ? sortBy.dir : ''].join(' ');
    return (
      <th onClick={() => sortClick(k)} className={cls} style={{ textAlign: align || 'left', cursor: 'pointer' }}>
        {label}
        {sortBy.key === k && <span className="mono fg-3" style={{ marginLeft: 4 }}>{sortBy.dir === 'asc' ? '↑' : '↓'}</span>}
      </th>
    );
  }

  // ============ render ============

  if (loading) {
    return (
      <div style={{ padding: 24, textAlign: 'center', color: 'var(--fg-muted)', minHeight: 300 }}>
        <span className="t-label">載入模型中...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: 24 }}>
        <Surface style={{ padding: 24 }}>
          <Row gap={10}>
            <Icon name="warning" size={20} style={{ color: 'var(--bad)' }} />
            <div>
              <div className="t-title">無法載入</div>
              <div className="t-label" style={{ marginTop: 4 }}>{error}</div>
            </div>
          </Row>
        </Surface>
      </div>
    );
  }

  if (models.length === 0) {
    return (
      <div style={{ padding: 24 }}>
        <Row align="end" style={{ marginBottom: 16 }}>
          <div>
            <h1 className="t-h1">模型</h1>
            <p className="t-label" style={{ marginTop: 4 }}>還沒有任何訓練好的模型</p>
          </div>
        </Row>
        <Surface style={{ padding: 48, textAlign: 'center' }}>
          <Icon name="bars" size={40} style={{ color: 'var(--primary)', marginBottom: 12 }} />
          <div className="t-title">尚無模型</div>
          <div className="t-label" style={{ marginTop: 6, marginBottom: 20 }}>
            前往實驗室開始第一個訓練
          </div>
          <Button variant="primary" icon="plus" onClick={() => onNavigate('experiments')}>新實驗</Button>
        </Surface>
      </div>
    );
  }

  // 統計各 run 的 model 數,給下拉選項用
  const runOptions = runs.map(r => ({
    value: r.id,
    label: `${r.datasetName || r.id} · ${r.target} · ${_runTime(r.startedAt)} (${normalized.filter(m => m.trainingRunId === r.id).length})`,
  }));

  // compareModels 改從 normalized 拿
  const compareModels = normalized.filter(m => selected.includes(m.id));

  return (
    <div style={{ padding: 24, paddingBottom: selected.length ? 280 : 24 }} >
      <Row style={{ justifyContent: 'space-between', marginBottom: 16 }} align="end">
        <div>
          <h1 className="t-h1">模型</h1>
          <p className="t-label" style={{ marginTop: 4 }}>
            共 <span className="mono fg-1">{models.length}</span> 個模型 · 來自 <span className="mono fg-1">{runs.length}</span> 個訓練 run
            {filtered.length !== normalized.length && <span> · 篩選後 <span className="mono fg-1">{filtered.length}</span></span>}
          </p>
        </div>
        <Row gap={8}>
          <select className="input" value={runFilter} onChange={e => setRunFilter(e.target.value)} style={{ width: 280 }}>
            <option value="all">全部 run ({models.length})</option>
            {runOptions.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
          <input className="input" placeholder="篩選演算法名稱..." value={filter}
                 onChange={e => setFilter(e.target.value)} style={{ width: 180 }} />
        </Row>
      </Row>

      <Surface>
        <div style={{ overflow: 'auto' }}>
          <table className="tbl">
            <thead>
              <tr>
                <th style={{ width: 36 }}>
                  <input type="checkbox"
                         checked={selected.length > 0 && selected.length === sorted.length}
                         onChange={e => setSelected(e.target.checked ? sorted.map(m => m.id) : [])}
                         style={{ accentColor: 'var(--primary)' }} />
                </th>
                <th>#</th>
                <th>演算法</th>
                <th>來源</th>
                {thSort('f1',   'F1',       'right')}
                {thSort('auc',  'AUC',      'right')}
                {thSort('acc',  'Accuracy', 'right')}
                {thSort('trainTime', '訓練 (s)', 'right')}
                {thSort('inferLatency', '推論 (ms)', 'right')}
                <th></th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((m, i) => (
                <tr key={m.id} className={selected.includes(m.id) ? 'selected' : ''}>
                  <td>
                    <input type="checkbox" checked={selected.includes(m.id)} onChange={() => toggle(m.id)}
                           style={{ accentColor: 'var(--primary)' }} />
                  </td>
                  <td className="mono fg-3">{i + 1}</td>
                  <td className="fg-1">
                    <span title={m.rawName}>{m.algo}</span>
                  </td>
                  <td><Chip className="mono" tone={m.dataSource === 'preprocessed' ? 'primary' : undefined}>{m.source}</Chip></td>
                  <td className="mono fg-1" style={{ textAlign: 'right' }}>
                    {m.f1 != null ? <ScoreCell value={m.f1} max={1} /> : <span className="fg-4">—</span>}
                  </td>
                  <td className="mono" style={{ textAlign: 'right' }}>
                    {m.auc != null ? m.auc.toFixed(3) : <span className="fg-4">—</span>}
                  </td>
                  <td className="mono" style={{ textAlign: 'right' }}>
                    {m.acc != null ? m.acc.toFixed(3) : <span className="fg-4">—</span>}
                  </td>
                  <td className="mono fg-3" style={{ textAlign: 'right' }}>
                    {m.trainTime > 0 ? m.trainTime.toFixed(1) : <span className="fg-4">—</span>}
                  </td>
                  <td className="mono fg-3" style={{ textAlign: 'right' }}>
                    {m.inferLatency > 0 ? m.inferLatency.toFixed(2) : <span className="fg-4">—</span>}
                  </td>
                  <td>
                    <Row gap={4}>
                      <button
                        className="btn btn-ghost btn-sm"
                        onClick={() => analyzeModel(m.id)}
                        title="跳到洞察頁分析這個模型"
                      >
                        分析
                      </button>
                      <Button variant="bare" size="sm" icon="ellipsis" title="更多" />
                    </Row>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Surface>

      {/* Comparison drawer at the bottom */}
      {selected.length > 0 && <CompareDrawer models={compareModels} onClear={() => setSelected([])} onNavigate={onNavigate} />}
    </div>
  );
}

function _runTime(sec) {
  if (!sec) return '';
  const d = new Date(sec * 1000);
  return `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

function ScoreCell({ value, max }) {
  const pct = Math.min(Math.max(value / max, 0), 1);
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8, justifyContent: 'flex-end' }}>
      <span style={{ width: 36, height: 4, background: 'var(--bg-sunken)', borderRadius: 2, overflow: 'hidden' }}>
        <span style={{ display: 'block', height: '100%', width: `${pct * 100}%`, background: 'var(--primary)' }} />
      </span>
      <span style={{ minWidth: 44, textAlign: 'right' }}>{value.toFixed(3)}</span>
    </span>
  );
}

function CompareDrawer({ models, onClear, onNavigate }) {
  if (models.length === 0) return null;
  return (
    <div style={{
      position: 'fixed', left: 220, right: 0, bottom: 0,
      background: 'var(--bg-surface)',
      borderTop: '1px solid var(--bd-default)',
      boxShadow: '0 -12px 40px -10px rgba(0,0,0,0.4)',
      padding: 16,
      zIndex: 20,
      maxHeight: '40vh', overflow: 'auto',
    }} >
      <Row style={{ justifyContent: 'space-between', marginBottom: 12 }}>
        <Row gap={10}>
          <Chip tone="primary" className="mono">{models.length} 已選</Chip>
          <span className="t-body fg-2">
            比較 · {models.slice(0, 3).map(m => m.algo).join(' · ')}
            {models.length > 3 ? ` (+${models.length - 3})` : ''}
          </span>
        </Row>
        <Row gap={8}>
          <Button variant="ghost" size="sm" onClick={onClear}>清除</Button>
          <Button variant="primary" size="sm" iconRight="arrowRight" onClick={() => onNavigate('insights')}>深入分析</Button>
        </Row>
      </Row>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12 }}>
        {[
          { label: 'F1',       key: 'f1',    fmt: v => v == null ? '—' : v.toFixed(3) },
          { label: 'AUC',      key: 'auc',   fmt: v => v == null ? '—' : v.toFixed(3) },
          { label: 'Accuracy', key: 'acc',   fmt: v => v == null ? '—' : v.toFixed(3) },
          { label: '訓練 (s)',  key: 'trainTime', fmt: v => (v || 0).toFixed(1) },
          { label: '推論 (ms)', key: 'inferLatency', fmt: v => (v || 0).toFixed(2) },
        ].map(c => {
          const vals = models.map(m => m[c.key]).filter(v => v != null);
          const best = c.key === 'trainTime' || c.key === 'inferLatency'
            ? (vals.length ? Math.min(...vals) : null)
            : (vals.length ? Math.max(...vals) : null);
          return (
            <div key={c.key} style={{ padding: 12, background: 'var(--bg-sunken)', borderRadius: 7 }}>
              <div className="t-label" style={{ marginBottom: 8 }}>{c.label}</div>
              <Stack gap={6}>
                {models.map((m, i) => (
                  <Row key={m.id} gap={6}>
                    <div style={{ width: 8, height: 8, borderRadius: 2,
                                   background: ['var(--primary)', 'var(--good)', 'var(--warn)', 'var(--bad)', 'var(--fg-default)'][i % 5] }} />
                    <span className="t-label fg-2" style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m.algo}</span>
                    <span className={`mono ${m[c.key] === best ? 'fg-1' : 'fg-3'}`}
                          style={{ fontSize: 11, fontWeight: m[c.key] === best ? 600 : 400 }}>
                      {c.fmt(m[c.key])}
                    </span>
                  </Row>
                ))}
              </Stack>
            </div>
          );
        })}
      </div>
    </div>
  );
}

window.PageModels = PageModels;
