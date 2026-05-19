// =============================================================================
// Page: Dashboard
// =============================================================================

function PageDashboard({ onNavigate }) {
  const [range, setRange] = React.useState('7d');
  const ranges = [
    { value: '24h', label: '24h' },
    { value: '7d',  label: '7d'  },
    { value: '30d', label: '30d' },
    { value: '90d', label: '90d' },
  ];

  return (
    <div style={{ padding: 24 }} >
      {/* Page header */}
      <Row gap={16} align="end" style={{ justifyContent: 'space-between', marginBottom: 20 }}>
        <div>
          <h1 className="t-h1">總覽</h1>
          <p className="t-label" style={{ marginTop: 4 }}>
            本週 {MOCK.counts.experiments} 個實驗 · {MOCK.counts.models} 個模型 · <span className="mono">最後更新 14:32</span>
          </p>
        </div>
        <Row gap={8}>
          <Tabs items={ranges} value={range} onChange={setRange} />
          <Button variant="primary" icon="plus" onClick={() => onNavigate('experiments')}>新實驗</Button>
        </Row>
      </Row>

      {/* Metric cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
        <MetricCard label="活躍數據集" value="3" delta="+1" deltaTone="good" hint="vs 上週 2 個" />
        <MetricCard label="已完成實驗" value="12" delta="+4" deltaTone="good" hint="vs 上週 8 次" />
        <MetricCard label="最佳模型 F1" value="0.942" delta="+0.012" deltaTone="good" hint="客戶流失預測 v3 · XGBoost" highlight />
        <MetricCard label="平均訓練時長" value="6m 21s" delta="−42s" deltaTone="good" hint="HPO 縮短" />
      </div>

      {/* Charts row */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 12, marginBottom: 16 }}>
        <Surface>
          <CardHeader
            title="模型效能趨勢"
            subtitle="F1-score · 過去 7 天"
            right={
              <Row gap={14}>
                <Row gap={6}><Dot tone="active" /><span className="t-label">分類 (主)</span></Row>
                <Row gap={6}><span className="dot dot-muted" /><span className="t-label">回歸</span></Row>
              </Row>
            }
          />
          <div style={{ padding: 16, height: 200 }}>
            <TrendChart classification={MOCK.trendF1} regression={MOCK.trendRMSE.map(x => x / 3500)} />
          </div>
          <div className="divider-t" style={{ padding: '8px 16px', display: 'flex', justifyContent: 'space-between' }}>
            <Row gap={16}>
              {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'].map(d => (
                <span key={d} className="t-label mono" style={{ fontSize: 10 }}>{d}</span>
              ))}
            </Row>
            <span className="t-label mono">peak 0.951 · Fri</span>
          </div>
        </Surface>

        <Surface>
          <CardHeader title="任務類型" subtitle={`總計 ${MOCK.counts.experiments} 個`} />
          <div style={{ padding: 16 }}>
            <Stack gap={12}>
              <TaskBar label="分類"     count={MOCK.taskMix.classification} total={47} primary />
              <TaskBar label="回歸"     count={MOCK.taskMix.regression}     total={47} opacity={0.6} />
              <TaskBar label="時間序列" count={MOCK.taskMix.timeseries}     total={47} opacity={0.35} />
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
            {MOCK.experiments.slice(0, 4).map(exp => (
              <ExperimentRow key={exp.id} exp={exp} onClick={() => onNavigate('experiments')} />
            ))}
          </div>
        </Surface>

        <Surface>
          <CardHeader title="需要處理" right={<span className="t-label mono">3 unread</span>} />
          <div>
            {MOCK.notifications.map((n, i) => (
              <div key={i} style={{
                padding: '12px 16px',
                borderBottom: i < MOCK.notifications.length - 1 ? '1px solid var(--bd-subtle)' : 'none',
                display: 'flex', gap: 10,
              }}>
                <Chip tone={n.kind === 'warn' ? 'warn' : n.kind === 'good' ? 'good' : undefined} className="mono">
                  {n.kind === 'warn' ? '警告' : n.kind === 'good' ? '完成' : 'info'}
                </Chip>
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div className="t-body-lg fg-1">{n.title}</div>
                  <div className="t-label" style={{ marginTop: 4 }}>{n.body}</div>
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
      {hint && <div className="t-label" style={{ marginTop: 6, fontSize: 10 }}>{hint}</div>}
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

function ExperimentRow({ exp, onClick }) {
  const tone = exp.status === 'completed' ? 'good' : exp.status === 'failed' ? 'bad' : exp.status === 'running' ? 'active' : 'muted';
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
          {exp.name}
        </div>
        <div className="t-label mono" style={{ marginTop: 2 }}>
          {exp.task} · {exp.best ? exp.best.algo : exp.error || '—'}
        </div>
      </div>
      <div style={{ textAlign: 'right' }}>
        {exp.status === 'completed' && exp.best && (
          <div className="mono fg-1" style={{ fontSize: 12 }}>{exp.best.metric} {exp.best.value}</div>
        )}
        {exp.status === 'running' && (
          <div className="mono" style={{ fontSize: 12, color: 'var(--primary)' }}>trial 47/100</div>
        )}
        {exp.status === 'failed' && <Chip tone="bad">失敗</Chip>}
        <div className="t-label" style={{ marginTop: 2, fontSize: 10 }}>{exp.startedAt}</div>
      </div>
    </button>
  );
}

// Custom 2-series chart (we want full control, no chart lib)
function TrendChart({ classification, regression }) {
  const W = 100, H = 100;
  const yScale = vals => {
    const min = Math.min(...vals), max = Math.max(...vals);
    const r = max - min || 1;
    return v => H - ((v - min) / r) * (H - 16) - 8;
  };
  const cls = classification, reg = regression;
  const clsY = yScale(cls), regY = yScale(reg);
  const xs = i => (i / (cls.length - 1)) * W;
  const pathCls = cls.map((v, i) => `${i === 0 ? 'M' : 'L'}${xs(i).toFixed(2)},${clsY(v).toFixed(2)}`).join(' ');
  const pathReg = reg.map((v, i) => `${i === 0 ? 'M' : 'L'}${xs(i).toFixed(2)},${regY(v).toFixed(2)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ width: '100%', height: '100%' }}>
      {/* gridlines */}
      <g stroke="var(--bd-subtle)" strokeWidth="0.3">
        <line x1="0" y1="20" x2={W} y2="20" />
        <line x1="0" y1="50" x2={W} y2="50" />
        <line x1="0" y1="80" x2={W} y2="80" />
      </g>
      {/* regression dashed */}
      <path d={pathReg} fill="none" stroke="var(--fg-faint)" strokeWidth="0.6" strokeDasharray="1.5 1.5" />
      {/* classification main */}
      <path d={pathCls} fill="none" stroke="var(--primary)" strokeWidth="0.9" vectorEffect="non-scaling-stroke" />
      {/* peak marker */}
      <circle cx={xs(cls.length - 1)} cy={clsY(cls[cls.length - 1])} r="1.2" fill="var(--primary)" />
      <circle cx={xs(cls.length - 1)} cy={clsY(cls[cls.length - 1])} r="3" fill="var(--primary)" opacity="0.2" />
    </svg>
  );
}

window.PageDashboard = PageDashboard;
