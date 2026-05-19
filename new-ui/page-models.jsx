// =============================================================================
// Page: Models — leaderboard with sortable cols, selection, compare drawer
// =============================================================================

function PageModels({ onNavigate }) {
  const [sortBy, setSortBy] = React.useState({ key: 'f1', dir: 'desc' });
  const [selected, setSelected] = React.useState([]);
  const [filter, setFilter] = React.useState('');

  function sortClick(k) {
    setSortBy(s => s.key === k ? { key: k, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key: k, dir: 'desc' });
  }

  const sorted = [...MOCK.models]
    .filter(m => !filter || m.algo.toLowerCase().includes(filter.toLowerCase()))
    .sort((a, b) => {
      const av = a[sortBy.key], bv = b[sortBy.key];
      const sign = sortBy.dir === 'asc' ? 1 : -1;
      if (typeof av === 'string') return av.localeCompare(bv) * sign;
      return (av - bv) * sign;
    });

  const compareModels = MOCK.models.filter(m => selected.includes(m.id));

  function toggle(id) {
    setSelected(s => s.includes(id) ? s.filter(x => x !== id) : [...s, id]);
  }

  function thSort(k, label, align) {
    const cls = ['tbl-sort', sortBy.key === k ? sortBy.dir : ''].join(' ');
    return (
      <th onClick={() => sortClick(k)} className={cls} style={{ textAlign: align || 'left' }}>
        {label}
      </th>
    );
  }

  return (
    <div style={{ padding: 24, paddingBottom: selected.length ? 220 : 24 }} >
      <Row style={{ justifyContent: 'space-between', marginBottom: 16 }} align="end">
        <div>
          <h1 className="t-h1">模型</h1>
          <p className="t-label" style={{ marginTop: 4 }}>
            來自 <span className="mono fg-1">客戶流失預測 v3</span> · {MOCK.models.length} 個模型
          </p>
        </div>
        <Row gap={8}>
          <select className="input" style={{ width: 200 }}>
            <option>客戶流失預測 v3 (12)</option>
            <option>銷售額預測 Q2 (5)</option>
            <option>產品推薦引擎 (9)</option>
          </select>
          <input className="input" placeholder="篩選演算法..." value={filter} onChange={e => setFilter(e.target.value)} style={{ width: 180 }} />
        </Row>
      </Row>

      <Surface>
        <table className="tbl">
          <thead>
            <tr>
              <th style={{ width: 36 }}>
                <input type="checkbox"
                       checked={selected.length === MOCK.models.length}
                       onChange={e => setSelected(e.target.checked ? MOCK.models.map(m => m.id) : [])}
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
              <th>標籤</th>
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
                <td className="mono fg-3">{m.rank}</td>
                <td className="fg-1">
                  <Row gap={8}>
                    <span>{m.algo}</span>
                  </Row>
                </td>
                <td><Chip className="mono">{m.source}</Chip></td>
                <td className="mono fg-1" style={{ textAlign: 'right' }}>
                  <ScoreCell value={m.f1} max={0.95} />
                </td>
                <td className="mono" style={{ textAlign: 'right' }}>{m.auc.toFixed(3)}</td>
                <td className="mono" style={{ textAlign: 'right' }}>{m.acc.toFixed(3)}</td>
                <td className="mono fg-3" style={{ textAlign: 'right' }}>{m.trainTime}</td>
                <td className="mono fg-3" style={{ textAlign: 'right' }}>{m.inferLatency.toFixed(1)}</td>
                <td>{m.tag ? <Chip tone={m.tag === 'best' ? 'primary' : 'good'} className="mono">{m.tag}</Chip> : null}</td>
                <td>
                  <Row gap={4}>
                    <button
                      className="btn btn-ghost btn-sm"
                      onClick={() => onNavigate('insights')}
                      title="跳到洞察頁分析這個模型"
                    >
                      分析
                    </button>
                    <Button variant="bare" size="sm" icon="download" title="下載模型" />
                    <Button variant="bare" size="sm" icon="ellipsis" title="更多" />
                  </Row>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Surface>

      {/* Comparison drawer at the bottom */}
      {selected.length > 0 && <CompareDrawer models={compareModels} onClear={() => setSelected([])} onNavigate={onNavigate} />}
    </div>
  );
}

function ScoreCell({ value, max }) {
  const pct = Math.min(value / max, 1);
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
  return (
    <div style={{
      position: 'fixed', left: 220, right: 0, bottom: 0,
      background: 'var(--bg-surface)',
      borderTop: '1px solid var(--bd-default)',
      boxShadow: '0 -12px 40px -10px rgba(0,0,0,0.4)',
      padding: 16,
      zIndex: 20,
    }} >
      <Row style={{ justifyContent: 'space-between', marginBottom: 12 }}>
        <Row gap={10}>
          <Chip tone="primary" className="mono">{models.length} 已選</Chip>
          <span className="t-body fg-2">比較 · {models.map(m => m.algo).join(' · ')}</span>
        </Row>
        <Row gap={8}>
          <Button variant="ghost" size="sm" onClick={onClear}>清除</Button>
          <Button variant="ghost" size="sm" icon="download">匯出比較</Button>
          <Button variant="primary" size="sm" iconRight="arrowRight" onClick={() => onNavigate('insights')}>深入分析</Button>
        </Row>
      </Row>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12 }}>
        {[
          { label: 'F1',       key: 'f1',    fmt: v => v.toFixed(3) },
          { label: 'AUC',      key: 'auc',   fmt: v => v.toFixed(3) },
          { label: 'Accuracy', key: 'acc',   fmt: v => v.toFixed(3) },
          { label: '訓練 (s)',  key: 'trainTime', fmt: v => v + 's' },
          { label: '推論 (ms)', key: 'inferLatency', fmt: v => v + 'ms' },
        ].map(c => {
          const vals = models.map(m => m[c.key]);
          const best = Math.max(...vals);
          return (
            <div key={c.key} style={{ padding: 12, background: 'var(--bg-sunken)', borderRadius: 7 }}>
              <div className="t-label" style={{ marginBottom: 8 }}>{c.label}</div>
              <Stack gap={6}>
                {models.map((m, i) => (
                  <Row key={m.id} gap={6}>
                    <div style={{ width: 8, height: 8, borderRadius: 2, background: ['var(--primary)', 'var(--good)', 'var(--warn)', 'var(--bad)', 'var(--fg-default)'][i % 5] }} />
                    <span className="t-label fg-2" style={{ flex: 1 }}>{m.algo}</span>
                    <span className={`mono ${m[c.key] === best ? 'fg-1' : 'fg-3'}`} style={{ fontSize: 11, fontWeight: m[c.key] === best ? 600 : 400 }}>
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
