// =============================================================================
// Page: Dashboard — 從後端拉真實資料 (training runs / models / datasets)
// =============================================================================

function PageDashboard({ onNavigate }) {
  const [range, setRange] = React.useState('7d');
  const ranges = [
    { value: '24h', label: '24h' },
    { value: '7d',  label: '7d'  },
    { value: '30d', label: '30d' },
    { value: '90d', label: '90d' },
  ];

  // ---- 拉後端資料 ----
  const [loading, setLoading] = React.useState(true);
  const [error, setError]     = React.useState(null);
  const [runs, setRuns]       = React.useState([]);
  const [models, setModels]   = React.useState([]);
  const [datasets, setDatasets] = React.useState([]);
  const [refreshTick, setRefreshTick] = React.useState(0);

  React.useEffect(() => {
    setLoading(true);
    setError(null);
    // 用 cached 版本,首次載入後 60 秒內切頁面回來不會重抓 (refreshTick > 0 時 force 重抓)
    const opts = refreshTick > 0 ? { force: true } : {};
    Promise.all([
      NewUI.api.getCached('/api/training-runs?limit=100', opts).then(r => r.runs || []).catch(e => { console.warn('runs fetch fail', e); return []; }),
      NewUI.api.getCached('/api/models?limit=500', opts).then(r => r.models || []).catch(e => { console.warn('models fetch fail', e); return []; }),
      NewUI.api.getCached('/api/dataset/list', opts).then(r => r.datasets || []).catch(e => { console.warn('datasets fetch fail', e); return []; }),
    ]).then(([rRuns, rModels, rDatasets]) => {
      setRuns(rRuns);
      setModels(rModels);
      setDatasets(rDatasets);
      setLoading(false);
    }).catch(err => {
      setError(err.message || '無法載入儀表板資料');
      setLoading(false);
    });
  }, [refreshTick]);

  // ---- 依時間範圍 filter runs ----
  const cutoffSec = React.useMemo(() => {
    const now = Date.now() / 1000;
    const map = { '24h': 86400, '7d': 7 * 86400, '30d': 30 * 86400, '90d': 90 * 86400 };
    return now - (map[range] || 7 * 86400);
  }, [range]);

  const runsInRange = React.useMemo(
    () => runs.filter(r => (r.startedAt || 0) >= cutoffSec),
    [runs, cutoffSec],
  );
  const completedRuns = React.useMemo(
    () => runsInRange.filter(r => r.status === 'completed'),
    [runsInRange],
  );

  // ---- 衍生 metric ----
  const metrics = React.useMemo(() => {
    // 最佳分類 F1 (從 models bundle 拿)
    let bestF1 = null, bestF1Model = null;
    models.forEach(m => {
      const b = m.bundle || {};
      const mt = b.metrics || {};
      if ((b.taskType === 'classification' || mt.taskType === 'classification') && typeof mt.f1 === 'number') {
        if (bestF1 == null || mt.f1 > bestF1) {
          bestF1 = mt.f1;
          bestF1Model = b;
        }
      }
    });
    // 平均訓練時長 (秒)
    const elapsed = completedRuns.map(r => r.elapsedSec).filter(x => typeof x === 'number' && x > 0);
    const avgElapsed = elapsed.length ? elapsed.reduce((a, b) => a + b, 0) / elapsed.length : null;
    return {
      datasetCount: datasets.length,
      completedCount: completedRuns.length,
      bestF1, bestF1Model,
      avgElapsed,
    };
  }, [datasets, completedRuns, models]);

  // ---- 趨勢圖:過去 N 天每天的最佳 F1 (分類) / RMSE 相關 (回歸) ----
  const trend = React.useMemo(() => {
    const days = range === '24h' ? 1 : range === '7d' ? 7 : range === '30d' ? 30 : 90;
    const buckets = new Array(days).fill(null).map(() => ({ cls: null, reg: null }));
    const nowDay = Math.floor(Date.now() / 86400000);
    runsInRange.forEach(r => {
      if (r.status !== 'completed' || !r.resultsSummary) return;
      const rDay = Math.floor((r.startedAt || 0) / 86400);
      const idx = days - 1 - (nowDay - rDay);
      if (idx < 0 || idx >= days) return;
      // 找 best score:從 resultsSummary 拿
      const summary = r.resultsSummary;
      const score = summary.topModels?.[0]?.score
        ?? summary.perSource?.[0]?.bestScore
        ?? summary.perSource?.[0]?.f1
        ?? null;
      if (typeof score !== 'number') return;
      const bucket = buckets[idx];
      if (r.taskType === 'classification' || (r.taskType !== 'regression' && score <= 1)) {
        if (bucket.cls == null || score > bucket.cls) bucket.cls = score;
      } else {
        if (bucket.reg == null || score < bucket.reg) bucket.reg = score;  // 回歸:小越好
      }
    });
    return buckets;
  }, [runsInRange, range]);

  // ---- 任務類型分布 ----
  const taskMix = React.useMemo(() => {
    const mix = { classification: 0, regression: 0, timeseries: 0 };
    runsInRange.forEach(r => {
      const isTS = r.options?.timeSeries;
      if (isTS) mix.timeseries++;
      else if (r.taskType === 'regression') mix.regression++;
      else mix.classification++;
    });
    return mix;
  }, [runsInRange]);

  // ---- 最近實驗 (4 個,新→舊) ----
  const recentRuns = React.useMemo(() => runs.slice(0, 4), [runs]);

  // ---- 通知:從 runs 衍生 (failed runs / running runs) ----
  const notifications = React.useMemo(() => {
    const out = [];
    runs.slice(0, 10).forEach(r => {
      if (r.status === 'failed') {
        out.push({
          kind: 'warn',
          title: `訓練失敗 — ${r.datasetName || 'Dataset'}`,
          body: r.errorMsg || '未知錯誤',
          time: _fmtTime(r.startedAt),
        });
      } else if (r.status === 'running') {
        out.push({
          kind: 'info',
          title: `訓練進行中 — ${r.datasetName || 'Dataset'}`,
          body: `target=${r.target} · engine=${r.engine}`,
          time: _fmtTime(r.startedAt),
        });
      }
    });
    return out.slice(0, 5);
  }, [runs]);

  // ---- 第一次載入 / 完全沒資料的 empty state ----
  const isEmpty = !loading && runs.length === 0 && datasets.length === 0 && models.length === 0;

  // ============ render ============

  if (loading) {
    return (
      <div style={{ padding: 24, display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 400 }}>
        <Row gap={10}>
          <div style={{
            width: 16, height: 16, border: '2px solid var(--primary)',
            borderTopColor: 'transparent', borderRadius: '50%',
            animation: 'spin 0.8s linear infinite',
          }} />
          <span className="t-label">載入儀表板...</span>
        </Row>
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

  if (isEmpty) {
    return (
      <div style={{ padding: 24 }}>
        <Row gap={16} align="end" style={{ justifyContent: 'space-between', marginBottom: 20 }}>
          <div>
            <h1 className="t-h1">總覽</h1>
            <p className="t-label" style={{ marginTop: 4 }}>歡迎!還沒有資料,先上傳一個資料集開始</p>
          </div>
        </Row>
        <Surface style={{ padding: 48, textAlign: 'center' }}>
          <Icon name="data" size={40} style={{ color: 'var(--primary)', marginBottom: 12 }} />
          <div className="t-title">尚無資料</div>
          <div className="t-label" style={{ marginTop: 6, marginBottom: 20 }}>
            上傳一個 CSV 開始你的第一個 AutoML 實驗
          </div>
          <Button variant="primary" icon="upload" onClick={() => onNavigate('data')}>前往上傳數據</Button>
        </Surface>
      </div>
    );
  }

  // ---- 正常 dashboard ----
  return (
    <div style={{ padding: 24 }} >
      {/* Page header */}
      <Row gap={16} align="end" style={{ justifyContent: 'space-between', marginBottom: 20 }}>
        <div>
          <h1 className="t-h1">總覽</h1>
          <p className="t-label" style={{ marginTop: 4 }}>
            本{_rangeLabel(range)} <span className="mono fg-1">{runsInRange.length}</span> 個實驗 · <span className="mono fg-1">{models.length}</span> 個模型 · <span className="mono">最後更新 {_now()}</span>
          </p>
        </div>
        <Row gap={8}>
          <Tabs items={ranges} value={range} onChange={setRange} />
          <Button variant="ghost" size="sm" icon="refresh" onClick={() => setRefreshTick(t => t + 1)}>重新整理</Button>
          <Button variant="primary" icon="plus" onClick={() => onNavigate('experiments')}>新實驗</Button>
        </Row>
      </Row>

      {/* Metric cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
        <MetricCard
          label="活躍數據集"
          value={metrics.datasetCount}
          hint={`${datasets.length === 0 ? '尚未上傳' : datasets[0]?.fileName || ''}`}
        />
        <MetricCard
          label={`本${_rangeLabel(range)}已完成實驗`}
          value={metrics.completedCount}
          hint={`共 ${runsInRange.length} 次嘗試`}
        />
        <MetricCard
          label="最佳分類 F1"
          value={metrics.bestF1 != null ? metrics.bestF1.toFixed(3) : '—'}
          hint={metrics.bestF1Model ? (metrics.bestF1Model.name || '').replace(/^\[(原始|預處理)\]\s*/, '') : '尚無分類模型'}
          highlight={metrics.bestF1 != null}
        />
        <MetricCard
          label="平均訓練時長"
          value={metrics.avgElapsed != null ? _fmtElapsed(metrics.avgElapsed) : '—'}
          hint={`${completedRuns.length} 次完成訓練的平均`}
        />
      </div>

      {/* Charts row */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 12, marginBottom: 16 }}>
        <Surface>
          <CardHeader
            title="模型效能趨勢"
            subtitle={`分類 F1 / 回歸 score · 過去 ${_rangeLabel(range)}`}
            right={
              <Row gap={14}>
                <Row gap={6}><Dot tone="active" /><span className="t-label">分類</span></Row>
                <Row gap={6}><span className="dot dot-muted" /><span className="t-label">回歸</span></Row>
              </Row>
            }
          />
          <div style={{ padding: 16, height: 200 }}>
            <TrendChart buckets={trend} />
          </div>
        </Surface>

        <Surface>
          <CardHeader title="任務類型" subtitle={`總計 ${runsInRange.length} 個`} />
          <div style={{ padding: 16 }}>
            <Stack gap={12}>
              <TaskBar label="分類"     count={taskMix.classification} total={Math.max(runsInRange.length, 1)} primary />
              <TaskBar label="回歸"     count={taskMix.regression}     total={Math.max(runsInRange.length, 1)} opacity={0.6} />
              <TaskBar label="時間序列" count={taskMix.timeseries}     total={Math.max(runsInRange.length, 1)} opacity={0.35} />
            </Stack>
          </div>
        </Surface>
      </div>

      {/* Activity + Notifications row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <Surface>
          <CardHeader
            title="最近實驗"
            right={
              <button className="btn btn-bare btn-sm" onClick={() => onNavigate('experiments')}>
                查看全部 <Icon name="arrowRight" size={11} />
              </button>
            }
          />
          <div>
            {recentRuns.length === 0
              ? <div style={{ padding: 24, textAlign: 'center' }}><span className="t-label">尚無實驗</span></div>
              : recentRuns.map(r => (
                  <ExperimentRowReal key={r.id} run={r} onClick={() => onNavigate('experiments')} />
                ))}
          </div>
        </Surface>

        <Surface>
          <CardHeader title="需要處理" right={<span className="t-label mono">{notifications.length} unread</span>} />
          <div>
            {notifications.length === 0
              ? <div style={{ padding: 24, textAlign: 'center' }}><span className="t-label">沒有需要處理的事項</span></div>
              : notifications.map((n, i) => (
                <div key={i} style={{
                  padding: '12px 16px',
                  borderBottom: i < notifications.length - 1 ? '1px solid var(--bd-subtle)' : 'none',
                  display: 'flex', gap: 10,
                }}>
                  <Chip tone={n.kind === 'warn' ? 'warn' : n.kind === 'good' ? 'good' : undefined} className="mono">
                    {n.kind === 'warn' ? '警告' : n.kind === 'good' ? '完成' : 'info'}
                  </Chip>
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div className="t-body-lg fg-1" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{n.title}</div>
                    <div className="t-label" style={{ marginTop: 4, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{n.body}</div>
                  </div>
                  <span className="t-label mono fg-4">{n.time}</span>
                </div>
              ))}
          </div>
        </Surface>
      </div>
    </div>
  );
}

// ---- 小工具 ----
function _rangeLabel(r) { return { '24h': '日', '7d': '週', '30d': '月', '90d': '季' }[r] || '週'; }
function _now() {
  const d = new Date();
  return `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
}
function _fmtTime(sec) {
  if (!sec) return '';
  const diff = Math.floor(Date.now() / 1000 - sec);
  if (diff < 60)    return `${diff}s`;
  if (diff < 3600)  return `${Math.floor(diff / 60)}m`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h`;
  return `${Math.floor(diff / 86400)}d`;
}
function _fmtElapsed(sec) {
  if (sec < 60) return `${sec.toFixed(0)}s`;
  const m = Math.floor(sec / 60);
  const s = Math.round(sec - m * 60);
  return `${m}m ${s.toString().padStart(2, '0')}s`;
}

// ---- Helpers ----
function MetricCard({ label, value, delta, deltaTone = 'flat', hint, highlight }) {
  const deltaColor = { good: 'var(--good)', bad: 'var(--bad)', flat: 'var(--fg-muted)' }[deltaTone];
  return (
    <Surface
      className={highlight ? 'surface-strong' : ''}
      style={{
        padding: 16,
        ...(highlight ? { borderColor: 'var(--primary-line)', background: 'linear-gradient(180deg, var(--primary-soft) 0%, var(--bg-surface) 60%)' } : {}),
      }}
    >
      <div className="t-label">{label}</div>
      <Row gap={8} style={{ marginTop: 8, alignItems: 'baseline' }}>
        <span className="t-metric">{value}</span>
        {delta && <span className="mono" style={{ color: deltaColor, fontSize: 11 }}>{delta}</span>}
      </Row>
      {hint && <div className="t-label" style={{ marginTop: 6, fontSize: 10, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{hint}</div>}
    </Surface>
  );
}

function TaskBar({ label, count, total, primary, opacity = 1 }) {
  const pct = Math.round((count / total) * 100);
  return (
    <div>
      <Row style={{ justifyContent: 'space-between', marginBottom: 4 }}>
        <span className="t-label fg-2">{label}</span>
        <span className="mono fg-1" style={{ fontSize: 11 }}>{count} <span className="fg-4">·</span> {pct}%</span>
      </Row>
      <div style={{ height: 6, borderRadius: 3, background: 'var(--bg-sunken)', overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: 'var(--primary)', opacity }} />
      </div>
    </div>
  );
}

// 真實 run 用的 row (取代 mock 用的 ExperimentRow)
function ExperimentRowReal({ run, onClick }) {
  const tone = run.status === 'completed' ? 'good' : run.status === 'failed' ? 'bad' : run.status === 'running' ? 'active' : 'muted';
  const summary = run.resultsSummary || {};
  const best = summary.topModels?.[0] || summary.perSource?.[0] || null;
  return (
    <button onClick={onClick} style={{
      width: '100%', padding: '11px 16px', display: 'flex', alignItems: 'center', gap: 12,
      background: 'transparent', border: 'none', borderBottom: '1px solid var(--bd-subtle)',
      textAlign: 'left', cursor: 'pointer',
    }}
      onMouseOver={e => e.currentTarget.style.background = 'var(--bg-hover)'}
      onMouseOut={e => e.currentTarget.style.background = 'transparent'}>
      <Dot tone={tone} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div className="t-body-lg fg-1" style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
          {run.datasetName || 'Dataset'} <span className="fg-4">·</span> <span className="mono fg-3">{run.target}</span>
        </div>
        <div className="t-label mono" style={{ marginTop: 2 }}>
          {run.taskType === 'regression' ? 'reg' : 'cls'} · {run.engine}
          {best?.name && ` · ${(best.name || '').replace(/^\[(原始|預處理)\]\s*/, '').slice(0, 30)}`}
        </div>
      </div>
      <div style={{ textAlign: 'right' }}>
        {run.status === 'completed' && best && typeof best.score === 'number' && (
          <div className="mono fg-1" style={{ fontSize: 12 }}>{best.score.toFixed(3)}</div>
        )}
        {run.status === 'running' && (
          <div className="mono" style={{ fontSize: 12, color: 'var(--primary)' }}>running</div>
        )}
        {run.status === 'failed' && <Chip tone="bad">失敗</Chip>}
        <div className="t-label" style={{ marginTop: 2, fontSize: 10 }}>{_fmtTime(run.startedAt)}</div>
      </div>
    </button>
  );
}

// Trend chart — 接 buckets [{cls, reg}, ...]
function TrendChart({ buckets }) {
  const W = 100, H = 100;
  const validCls = buckets.map(b => b.cls).filter(v => v != null);
  const validReg = buckets.map(b => b.reg).filter(v => v != null);
  if (validCls.length === 0 && validReg.length === 0) {
    return (
      <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--fg-faint)' }}>
        <span className="t-label">此期間沒有完成的訓練</span>
      </div>
    );
  }

  const allVals = [...validCls, ...validReg];
  const min = Math.min(...allVals), max = Math.max(...allVals);
  const range = max - min || 1;
  const yScale = v => H - ((v - min) / range) * (H - 16) - 8;
  const xs = i => (i / Math.max(buckets.length - 1, 1)) * W;

  const pathFor = (key) => {
    const segments = [];
    let started = false;
    buckets.forEach((b, i) => {
      const v = b[key];
      if (v == null) {
        started = false;
        return;
      }
      segments.push(`${started ? 'L' : 'M'}${xs(i).toFixed(2)},${yScale(v).toFixed(2)}`);
      started = true;
    });
    return segments.join(' ');
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ width: '100%', height: '100%' }}>
      <g stroke="var(--bd-subtle)" strokeWidth="0.3">
        <line x1="0" y1="20" x2={W} y2="20" />
        <line x1="0" y1="50" x2={W} y2="50" />
        <line x1="0" y1="80" x2={W} y2="80" />
      </g>
      <path d={pathFor('reg')} fill="none" stroke="var(--fg-faint)" strokeWidth="0.6" strokeDasharray="1.5 1.5" />
      <path d={pathFor('cls')} fill="none" stroke="var(--primary)" strokeWidth="0.9" vectorEffect="non-scaling-stroke" />
      {buckets.map((b, i) => b.cls != null && (
        <circle key={i} cx={xs(i)} cy={yScale(b.cls)} r="0.8" fill="var(--primary)" />
      ))}
    </svg>
  );
}

window.PageDashboard = PageDashboard;
