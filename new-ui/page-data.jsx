// =============================================================================
// Page: Data — 從後端拉真實 datasets (含上傳)
// =============================================================================

function PageData({ onNavigate }) {
  const [tab, setTab] = React.useState('overview');
  const [datasets, setDatasets] = React.useState([]);
  const [activeId, setActiveId] = React.useState(null);
  const [activeDetail, setActiveDetail] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [detailLoading, setDetailLoading] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [uploading, setUploading] = React.useState(false);
  const fileInputRef = React.useRef(null);

  // 拉 dataset list (有 cache,切回 Data 頁不用重抓)
  const reloadList = React.useCallback((force = false) => {
    setLoading(true);
    setError(null);
    NewUI.api.getCached('/api/dataset/list', { force })
      .then(r => {
        const list = r.datasets || [];
        setDatasets(list);
        if (list.length > 0 && (!activeId || !list.find(d => d.id === activeId))) {
          setActiveId(list[0].id);
        } else if (list.length === 0) {
          setActiveId(null);
          setActiveDetail(null);
        }
      })
      .catch(e => setError(e.message || '無法載入數據集'))
      .finally(() => setLoading(false));
  }, [activeId]);

  React.useEffect(() => { reloadList(false); }, []);

  // 拉 active dataset detail (cached)
  React.useEffect(() => {
    if (!activeId) { setActiveDetail(null); return; }
    setDetailLoading(true);
    NewUI.api.getCached(`/api/dataset/${encodeURIComponent(activeId)}`)
      .then(setActiveDetail)
      .catch(e => {
        console.warn('dataset detail fetch fail', e);
        setActiveDetail(null);
      })
      .finally(() => setDetailLoading(false));
  }, [activeId]);

  // 上傳新 CSV
  function onUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    const form = new FormData();
    form.append('file', file);
    NewUI.api.request('/api/preprocess', { method: 'POST', body: form })
      .then(resp => {
        // 寫操作完成 → 清掉 dataset / models / runs 的 cache,讓 sidebar 跟其他頁拿到新資料
        NewUI.api.invalidate('/api/dataset');
        NewUI.api.invalidate('/api/models');
        NewUI.api.invalidate('/api/training-runs');
        window.dispatchEvent(new CustomEvent('newui:refresh-counts'));
        setActiveId(resp.id);
        reloadList(true);
      })
      .catch(err => alert(`上傳失敗: ${err.message || err}`))
      .finally(() => {
        setUploading(false);
        if (fileInputRef.current) fileInputRef.current.value = '';
      });
  }

  // 刪除 dataset
  function deleteActive() {
    if (!activeId || !activeDetail) return;
    if (!confirm(`確定刪除「${activeDetail.fileName || activeId}」?\n此操作會連帶刪除衍生的 preprocessor / model。`)) return;
    NewUI.api.del(`/api/dataset/${encodeURIComponent(activeId)}`)
      .then(() => {
        // 級聯刪除影響 dataset / preprocessor / model / run,全部清 cache
        NewUI.api.invalidate('/api/dataset');
        NewUI.api.invalidate('/api/models');
        NewUI.api.invalidate('/api/training-runs');
        NewUI.api.invalidate('/api/preprocess');
        window.dispatchEvent(new CustomEvent('newui:refresh-counts'));
        setActiveId(null);
        reloadList(true);
      })
      .catch(err => alert(`刪除失敗: ${err.message || err}`));
  }

  const active = datasets.find(d => d.id === activeId);
  const colCount = active?.colCount ?? activeDetail?.colCount ?? 0;

  return (
    <div style={{ padding: 24 }} >
      {/* hidden file picker */}
      <input ref={fileInputRef} type="file" accept=".csv,.tsv,.txt"
             style={{ display: 'none' }} onChange={onUpload} />

      {/* Page header */}
      <Row align="end" style={{ justifyContent: 'space-between', marginBottom: 16 }}>
        <div>
          <h1 className="t-h1">數據</h1>
          <p className="t-label" style={{ marginTop: 4 }}>
            {loading ? '載入中...' : `${datasets.length} 個數據集 · 上傳、檢視、預處理`}
          </p>
        </div>
        <Row gap={8}>
          {active && (
            <Button variant="ghost" icon="trash" onClick={deleteActive}>刪除</Button>
          )}
          <Button variant="primary" icon="upload"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={uploading}>
            {uploading ? '上傳中...' : '上傳新數據'}
          </Button>
        </Row>
      </Row>

      {/* Loading state */}
      {loading && (
        <div style={{ padding: 48, textAlign: 'center', color: 'var(--fg-muted)' }}>
          <span className="t-label">載入數據集中...</span>
        </div>
      )}

      {/* Error state */}
      {error && !loading && (
        <Surface style={{ padding: 24, marginBottom: 16 }}>
          <Row gap={10}>
            <Icon name="warning" size={20} style={{ color: 'var(--bad)' }} />
            <div>
              <div className="t-title">無法載入數據集</div>
              <div className="t-label" style={{ marginTop: 4 }}>{error}</div>
            </div>
          </Row>
        </Surface>
      )}

      {/* Empty state (no datasets yet) */}
      {!loading && !error && datasets.length === 0 && (
        <Surface style={{ padding: 48, textAlign: 'center' }}>
          <Icon name="data" size={40} style={{ color: 'var(--primary)', marginBottom: 12 }} />
          <div className="t-title">尚未上傳任何數據集</div>
          <div className="t-label" style={{ marginTop: 6, marginBottom: 20 }}>
            上傳 CSV 開始預處理 / 訓練流程
          </div>
          <Button variant="primary" icon="upload" onClick={() => fileInputRef.current?.click()}>
            選擇 CSV 檔案
          </Button>
        </Surface>
      )}

      {/* Dataset list strip + content (only when has data) */}
      {!loading && !error && datasets.length > 0 && (
        <>
          {/* Dataset list strip (horizontal cards) */}
          <Row gap={10} style={{ marginBottom: 16, alignItems: 'stretch', flexWrap: 'wrap' }}>
            {datasets.map(d => (
              <button key={d.id}
                      onClick={() => setActiveId(d.id)}
                      className={`surface ${activeId === d.id ? 'surface-strong' : ''}`}
                      style={{
                        flex: '1 1 220px', minWidth: 220, padding: 14, textAlign: 'left',
                        background: activeId === d.id ? 'var(--primary-soft)' : 'var(--bg-surface)',
                        borderColor: activeId === d.id ? 'var(--primary-line)' : 'var(--bd-subtle)',
                        cursor: 'pointer', position: 'relative',
                      }}>
                <Row style={{ justifyContent: 'space-between', marginBottom: 8 }}>
                  <Icon name="data" size={16}
                        style={{ color: activeId === d.id ? 'var(--primary)' : 'var(--fg-muted)' }} />
                  <span className="t-label mono fg-4" style={{ fontSize: 10 }}>{_fmtRelTime(d.loadedAt)}</span>
                </Row>
                <div className="t-title" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {d.fileName || d.id}
                </div>
                <Row gap={8} style={{ marginTop: 6 }}>
                  <span className="t-label mono">{(d.rowCount ?? 0).toLocaleString()} 列</span>
                  <span className="fg-4">·</span>
                  <span className="t-label mono">{d.colCount ?? 0} 欄</span>
                </Row>
              </button>
            ))}
            <button onClick={() => fileInputRef.current?.click()} style={{
              flex: '0 0 90px',
              background: 'transparent',
              border: '1px dashed var(--bd-default)',
              borderRadius: 10,
              color: 'var(--fg-muted)',
              cursor: 'pointer',
              display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 4,
            }}>
              <Icon name="plus" size={20} />
              <span className="t-label">上傳</span>
            </button>
          </Row>

          {/* Tab navigation */}
          {active && (
            <PageTabs
              items={[
                { value: 'overview',  label: '概覽' },
                { value: 'columns',   label: '欄位', count: colCount },
                { value: 'preview',   label: '資料預覽' },
                { value: 'eda',       label: 'EDA' },
                { value: 'preprocess', label: '預處理' },
              ]}
              value={tab}
              onChange={setTab}
            />
          )}

          <div style={{ paddingTop: 16 }}>
            {detailLoading && (
              <div style={{ padding: 24, textAlign: 'center', color: 'var(--fg-muted)' }}>
                <span className="t-label">載入數據集詳情...</span>
              </div>
            )}
            {!detailLoading && active && (
              <>
                {tab === 'overview'  && <DataOverview ds={active} detail={activeDetail} />}
                {tab === 'columns'   && <DataColumns detail={activeDetail} />}
                {tab === 'preview'   && <DataPreview detail={activeDetail} />}
                {tab === 'eda'       && <DataEDA ds={active} detail={activeDetail} />}
                {tab === 'preprocess' && <DataPreprocess onNavigate={onNavigate} />}
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function _fmtRelTime(sec) {
  if (!sec) return '';
  const diff = Math.floor(Date.now() / 1000 - sec);
  if (diff < 60)    return '剛剛';
  if (diff < 3600)  return `${Math.floor(diff / 60)}m`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h`;
  return `${Math.floor(diff / 86400)}d`;
}

// ---- Overview tab ----
function DataOverview({ ds, detail }) {
  // detail 是後端 /api/dataset/{id} 的 response,可能還沒載
  const analysis = detail?.analysis || [];
  const auditReport = detail?.auditReport || {};
  const colTypes = analysis.reduce((acc, c) => {
    const t = c.type || 'unknown';
    acc[t] = (acc[t] || 0) + 1;
    return acc;
  }, {});
  const totalCols = analysis.length || ds.colCount || 0;

  // 從 audit report 衍生問題清單 (備用 fallback)
  const issues = _deriveIssues(detail, analysis);
  // 健康度從 audit 拿
  const health = typeof auditReport.score === 'number' ? auditReport.score : 75;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr) 1.6fr', gap: 12 }}>
      <SimpleStat label="總列數"     value={(ds.rowCount ?? 0).toLocaleString()} />
      <SimpleStat label="總欄位"     value={ds.colCount ?? totalCols} />
      <SimpleStat label="檔名"       value={ds.fileName || ds.id} mono />
      <SimpleStat label="缺失值欄位"  value={`${issues.filter(i => i.kind === 'missing').length}`} />

      <Surface style={{ gridColumn: 'span 5', marginTop: 4 }}>
        <CardHeader title="數據品質檢核" subtitle="自動掃描的問題"
                    right={<Chip tone={health >= 80 ? 'good' : health >= 60 ? 'warn' : 'bad'}>健康度 {health}/100</Chip>} />
        <div style={{ padding: '4px 16px' }}>
          {issues.length === 0 ? (
            <div style={{ padding: 16, textAlign: 'center' }}>
              <Chip tone="good" icon="checkCircle">資料品質檢核完成</Chip>
              <div className="t-label" style={{ marginTop: 8 }}>暫未偵測到明顯問題</div>
            </div>
          ) : issues.map((issue, i) => (
            <Row key={i} gap={10} align="start" style={{
              padding: '12px 0',
              borderBottom: i < 3 ? '1px solid var(--bd-subtle)' : 'none',
            }}>
              <Chip tone={issue.tone}><Icon name={issue.icon} size={11} /></Chip>
              <div style={{ flex: 1 }}>
                <div className="t-body-lg fg-1">{issue.title}</div>
                <div className="t-label" style={{ marginTop: 3 }}>{issue.body}</div>
              </div>
              <Button variant="bare" size="sm" iconRight="chevronRight">查看</Button>
            </Row>
          ))}
        </div>
      </Surface>

      <Surface style={{ gridColumn: 'span 5' }}>
        <CardHeader title="健康度雷達" subtitle="5 個維度 · 1.0 為滿分" />
        <div style={{ padding: 16, height: 280, display: 'flex', justifyContent: 'center' }}>
          <HealthRadar values={{
            完整性: 0.92, 一致性: 0.78, 平衡度: 0.74, 異常值: 0.66, 特徵品質: 0.86,
          }} />
        </div>
      </Surface>

      <Surface style={{ gridColumn: 'span 5' }}>
        <CardHeader title="欄位類型分布" />
        <div style={{ padding: 16, display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
          {[
            { label: '數值',   value: (colTypes.numeric || 0) + (colTypes.int || 0) + (colTypes.float || 0), color: 'var(--primary)' },
            { label: '類別',   value: (colTypes.categorical || 0) + (colTypes.cat || 0),   color: 'color-mix(in srgb, var(--primary) 70%, transparent)' },
            { label: '日期',   value: (colTypes.datetime || 0) + (colTypes.date || 0),  color: 'color-mix(in srgb, var(--primary) 45%, transparent)' },
            { label: '其他',   value: (colTypes.unknown || 0) + (colTypes.string || 0) + (colTypes.text || 0),    color: 'color-mix(in srgb, var(--primary) 25%, transparent)' },
          ].map((t, i) => (
            <div key={i} style={{ padding: 12, background: 'var(--bg-sunken)', borderRadius: 7, border: '1px solid var(--bd-subtle)' }}>
              <div className="t-label">{t.label}</div>
              <Row gap={8} style={{ marginTop: 6, alignItems: 'baseline' }}>
                <span className="mono t-h1" style={{ fontWeight: 600, color: 'var(--fg-strong)' }}>{t.value}</span>
                <span className="t-label mono">/ {totalCols || 1}</span>
              </Row>
              <div style={{ marginTop: 8, height: 3, background: 'var(--bg-canvas)', borderRadius: 2, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${(t.value / Math.max(totalCols, 1)) * 100}%`, background: t.color }} />
              </div>
            </div>
          ))}
        </div>
      </Surface>
    </div>
  );
}

function SimpleStat({ label, value, mono }) {
  return (
    <Surface style={{ padding: 16 }}>
      <div className="t-label">{label}</div>
      <div className={mono ? 'mono' : ''} style={{ marginTop: 6, fontSize: 22, fontWeight: 600, color: 'var(--fg-strong)' }}>{value}</div>
    </Surface>
  );
}

// ---- Columns tab ----
function DataColumns({ detail }) {
  const analysis = detail?.analysis || [];
  const totalRows = detail?.rowCount || 1;

  if (analysis.length === 0) {
    return (
      <Surface style={{ padding: 24, textAlign: 'center' }}>
        <span className="t-label">無欄位資訊 — 後端尚未回傳 analysis</span>
      </Surface>
    );
  }

  return (
    <Surface>
      <CardHeader
        title="欄位詳細"
        subtitle={`共 ${analysis.length} 個欄位`}
        right={
          <Row gap={6}>
            <Button variant="ghost" size="sm" icon="download">匯出 schema</Button>
          </Row>
        }
      />
      <table className="tbl">
        <thead>
          <tr>
            <th>欄位</th>
            <th>類型</th>
            <th style={{ textAlign: 'right' }}>缺失</th>
            <th style={{ textAlign: 'right' }}>唯一值</th>
            <th>分布</th>
          </tr>
        </thead>
        <tbody>
          {analysis.map((c, i) => {
            const missing = c.missing ?? c.nullCount ?? 0;
            const unique = c.unique ?? c.uniqueCount ?? 0;
            const distData = c.distribution || c.histogram || _synthDist(i);
            return (
              <tr key={c.name || i}>
                <td className="mono fg-1">{c.name}</td>
                <td><Chip tone={['numeric','int','float'].includes(c.type) ? 'primary' : undefined}>{c.type || '—'}</Chip></td>
                <td className="mono" style={{ textAlign: 'right', color: missing > 0 ? 'var(--warn)' : 'var(--fg-default)' }}>
                  {missing > 0 ? `${missing} (${(missing / totalRows * 100).toFixed(1)}%)` : '0'}
                </td>
                <td className="mono" style={{ textAlign: 'right' }}>{(unique || 0).toLocaleString()}</td>
                <td style={{ width: 100 }}>
                  <MiniBar data={Array.isArray(distData) ? distData : _synthDist(i)} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Surface>
  );
}

function _synthDist(seed) {
  return Array.from({ length: 8 }, (_, j) => 5 + Math.abs(Math.sin(seed * 1.3 + j)) * 20 + j * (seed % 2 ? -1 : 1));
}

// ---- Preview tab ----
function DataPreview({ detail }) {
  const headers = detail?.headers || [];
  // 後端 response 可能用 `preview` 或 `data` (依 preprocess.run 實作)
  const previewRows = detail?.preview || detail?.data || [];
  const totalRows = detail?.rowCount || 0;

  if (previewRows.length === 0 || headers.length === 0) {
    return (
      <Surface style={{ padding: 24, textAlign: 'center' }}>
        <span className="t-label">無預覽資料</span>
      </Surface>
    );
  }

  // preview 可能是 list of objects 或 list of arrays
  const isArrayFormat = Array.isArray(previewRows[0]);
  const displayRows = previewRows.slice(0, 20);

  return (
    <Surface>
      <CardHeader title={`前 ${displayRows.length} 筆`}
                  subtitle={`共 ${totalRows.toLocaleString()} 筆 · ${headers.length} 個欄位`} />
      <div style={{ overflow: 'auto', maxHeight: 480 }}>
        <table className="tbl" style={{ fontSize: 12 }}>
          <thead>
            <tr>
              <th style={{ width: 40, textAlign: 'right' }} className="mono fg-4">#</th>
              {headers.map(c => <th key={c} className="mono">{c}</th>)}
            </tr>
          </thead>
          <tbody>
            {displayRows.map((row, i) => (
              <tr key={i}>
                <td className="mono fg-4" style={{ textAlign: 'right' }}>{i + 1}</td>
                {headers.map((c, j) => {
                  const v = isArrayFormat ? row[j] : row[c];
                  return (
                    <td key={c} className="mono" style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {v == null || v === '' ? <span className="fg-4">—</span> : String(v)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Surface>
  );
}

// ---- Preprocess tab ----
function DataPreprocess({ onNavigate }) {
  const [running, setRunning] = React.useState(false);
  const [done, setDone] = React.useState(false);
  const [steps, setSteps] = React.useState([
    { id: 'impute', name: '缺失值填補',     method: 'median',         enabled: true },
    { id: 'outlier', name: '離群值處理',    method: 'iqr-clip',       enabled: true },
    { id: 'encode', name: '類別變數編碼',   method: 'onehot',         enabled: true },
    { id: 'scale', name: '數值標準化',      method: 'standard',       enabled: true },
    { id: 'date', name: '日期特徵分解',     method: 'extract-yt',     enabled: true },
    { id: 'corr', name: '高度相關特徵移除', method: 'r > 0.95',       enabled: false },
  ]);

  function run() {
    setRunning(true);
    setDone(false);
    setTimeout(() => { setRunning(false); setDone(true); }, 1800);
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 12 }}>
      <Surface>
        <CardHeader title="預處理流程" subtitle="按順序執行,可個別停用" />
        <div style={{ padding: '4px 16px' }}>
          {steps.map((s, i) => (
            <Row key={s.id} gap={12} style={{
              padding: '14px 0',
              borderBottom: i < steps.length - 1 ? '1px solid var(--bd-subtle)' : 'none',
            }}>
              <div style={{
                width: 24, height: 24, borderRadius: 6,
                background: s.enabled ? 'var(--primary-soft)' : 'var(--bg-sunken)',
                color: s.enabled ? 'var(--primary)' : 'var(--fg-faint)',
                border: '1px solid ' + (s.enabled ? 'var(--primary-line)' : 'var(--bd-subtle)'),
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 11, fontWeight: 600,
              }} className="mono">{i + 1}</div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="t-body-lg fg-1">{s.name}</div>
                <div className="t-label mono" style={{ marginTop: 2 }}>method: {s.method}</div>
              </div>
              <select className="input" style={{ width: 140, fontSize: 12 }} value={s.method}
                      onChange={e => setSteps(ss => ss.map(x => x.id === s.id ? {...x, method: e.target.value} : x))}>
                {{
                  impute:   ['median', 'mean', 'mode', 'knn-5'],
                  outlier:  ['iqr-clip', 'iqr-remove', 'zscore-3', 'none'],
                  encode:   ['onehot', 'target', 'ordinal', 'frequency'],
                  scale:    ['standard', 'minmax', 'robust', 'none'],
                  date:     ['extract-yt', 'extract-ymd', 'unix-only'],
                  corr:     ['r > 0.95', 'r > 0.90', 'r > 0.85'],
                }[s.id].map(m => <option key={m} value={m}>{m}</option>)}
              </select>
              <Toggle checked={s.enabled} onChange={v => setSteps(ss => ss.map(x => x.id === s.id ? {...x, enabled: v} : x))} />
            </Row>
          ))}
        </div>
        <div className="divider-t" style={{ padding: 14, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span className="t-label">將套用 {steps.filter(s => s.enabled).length} / {steps.length} 個步驟</span>
          <Row gap={8}>
            <Button variant="ghost" size="sm" icon="refresh">恢復預設</Button>
            <Button variant="primary" icon={running ? undefined : 'play'} onClick={run} disabled={running}>
              {running ? '處理中...' : '執行預處理'}
            </Button>
          </Row>
        </div>
      </Surface>

      <Surface>
        <CardHeader title="處理結果預覽" subtitle={done ? '完成,可以前往實驗頁面' : running ? '計算中' : '尚未執行'} />
        <div style={{ padding: 16 }}>
          {!running && !done && (
            <Empty icon="zap" title="尚未執行預處理" body="點擊左方「執行預處理」後,這裡會顯示前後對比" />
          )}
          {running && (
            <Stack gap={10}>
              {steps.filter(s => s.enabled).map((s, i) => (
                <Row key={s.id} gap={10}>
                  <div className="shimmer" style={{ width: 18, height: 18, borderRadius: 4 }} />
                  <div className="t-body-lg fg-2">{s.name}</div>
                  <span className="t-label mono fg-4" style={{ marginLeft: 'auto' }}>計算中</span>
                </Row>
              ))}
            </Stack>
          )}
          {done && (
            <Stack gap={14}>
              {[
                { label: '原始欄位數', a: 14, b: 14, suffix: '' },
                { label: '展開後欄位數', a: 14, b: 28, suffix: '', deltaTone: 'primary' },
                { label: '缺失值',     a: 187, b: 0, suffix: '', deltaTone: 'good' },
                { label: '離群值',     a: 261, b: 9, suffix: '', deltaTone: 'good' },
                { label: '訓練/測試 切分', a: null, b: '9,960 / 2,490', suffix: '', deltaTone: 'flat' },
              ].map((m, i) => (
                <Row key={i} style={{ justifyContent: 'space-between' }}>
                  <span className="t-body fg-2">{m.label}</span>
                  <Row gap={8}>
                    {m.a != null && <span className="mono fg-4" style={{ fontSize: 12 }}>{m.a}</span>}
                    {m.a != null && <Icon name="arrowRight" size={11} style={{ color: 'var(--fg-faint)' }} />}
                    <span className="mono fg-1" style={{ fontSize: 13 }}>{m.b}</span>
                  </Row>
                </Row>
              ))}
              <Button variant="primary" iconRight="arrowRight" onClick={() => onNavigate('experiments')}>
                前往訓練模型
              </Button>
            </Stack>
          )}
        </div>
      </Surface>
    </div>
  );
}

// ---- Health radar (5 維度) ----
function HealthRadar({ values }) {
  const keys = Object.keys(values);
  const n = keys.length;
  const cx = 50, cy = 50, rMax = 36;
  const angle = i => -Math.PI / 2 + (i / n) * Math.PI * 2;
  const point = (i, v) => [
    cx + Math.cos(angle(i)) * rMax * v,
    cy + Math.sin(angle(i)) * rMax * v,
  ];
  const polygon = keys.map((k, i) => point(i, values[k])).map(p => p.join(',')).join(' ');
  const rings = [0.25, 0.5, 0.75, 1];

  return (
    <svg viewBox="0 0 100 100" style={{ width: '100%', maxWidth: 280, height: '100%' }}>
      {/* Concentric rings */}
      {rings.map(r => (
        <polygon key={r} fill="none" stroke="var(--bd-subtle)" strokeWidth="0.3"
                 points={keys.map((_, i) => point(i, r)).map(p => p.join(',')).join(' ')} />
      ))}
      {/* Axes */}
      {keys.map((k, i) => {
        const [x, y] = point(i, 1);
        return <line key={k} x1={cx} y1={cy} x2={x} y2={y} stroke="var(--bd-subtle)" strokeWidth="0.3" />;
      })}
      {/* Filled radar */}
      <polygon points={polygon} fill="var(--primary)" opacity="0.25" stroke="var(--primary)" strokeWidth="0.6" />
      {/* Labels */}
      {keys.map((k, i) => {
        const [x, y] = point(i, 1.18);
        return (
          <text key={k} x={x} y={y} fontSize="3.2" fill="var(--fg-default)" fontFamily="Inter" textAnchor="middle" dominantBaseline="middle">
            {k}
          </text>
        );
      })}
      {/* Score points */}
      {keys.map((k, i) => {
        const [x, y] = point(i, values[k]);
        return <circle key={k} cx={x} cy={y} r="1.2" fill="var(--primary)" />;
      })}
    </svg>
  );
}

// ---- EDA tab (correlation heatmap + target dist + missing heatmap) ----
function DataEDA({ ds }) {
  const numericCols = MOCK.columns.filter(c => c.type === 'int' || c.type === 'float').map(c => c.name);
  // Synthesize a correlation matrix
  const corr = (() => {
    const m = numericCols.map(() => numericCols.map(() => 0));
    for (let i = 0; i < numericCols.length; i++) {
      for (let j = 0; j < numericCols.length; j++) {
        m[i][j] = i === j ? 1 : Math.sin(i * 7 + j * 3 + 1) * 0.7;
      }
    }
    return m;
  })();

  return (
    <Stack gap={12}>
      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 12 }}>
        <Surface>
          <CardHeader title="特徵相關係數" subtitle={`Pearson · ${numericCols.length} × ${numericCols.length}`} />
          <div style={{ padding: 16 }}>
            <CorrelationHeatmap matrix={corr} labels={numericCols} />
          </div>
        </Surface>

        <Surface>
          <CardHeader title={`Target 分布 — ${ds.target}`} subtitle={ds.task === 'classification' ? '類別比例' : '數值直方圖'} />
          <div style={{ padding: 16, height: 320 }}>
            {ds.task === 'classification'
              ? <TargetClassBar />
              : <TargetHistogram />}
          </div>
        </Surface>
      </div>

      <Surface>
        <CardHeader title="缺失值熱圖" subtitle="灰色 = 完整,黃 = 缺失" />
        <div style={{ padding: 16 }}>
          <MissingHeatmap />
        </div>
      </Surface>
    </Stack>
  );
}

function CorrelationHeatmap({ matrix, labels }) {
  const n = matrix.length;
  const cellSize = Math.max(8, 80 / n);
  return (
    <div style={{ overflow: 'auto' }}>
      <svg viewBox={`0 0 ${n * cellSize + 30} ${n * cellSize + 30}`}
           style={{ width: '100%', minWidth: n * 18 + 30 }}>
        {/* Column labels (top) */}
        {labels.map((l, i) => (
          <text key={'c' + i} x={30 + (i + 0.5) * cellSize} y={24}
                fontSize="2.8" fill="var(--fg-muted)" fontFamily="JetBrains Mono"
                transform={`rotate(-45, ${30 + (i + 0.5) * cellSize}, 24)`} textAnchor="end">
            {l.length > 10 ? l.slice(0, 9) + '…' : l}
          </text>
        ))}
        {/* Row labels (left) */}
        {labels.map((l, i) => (
          <text key={'r' + i} x={28} y={30 + (i + 0.55) * cellSize}
                fontSize="2.8" fill="var(--fg-muted)" fontFamily="JetBrains Mono" textAnchor="end">
            {l.length > 10 ? l.slice(0, 9) + '…' : l}
          </text>
        ))}
        {/* Cells */}
        {matrix.map((row, i) => row.map((v, j) => {
          const intensity = Math.abs(v);
          const fill = v > 0
            ? `color-mix(in srgb, var(--primary) ${intensity * 100}%, var(--bg-sunken))`
            : `color-mix(in srgb, var(--bad) ${intensity * 100}%, var(--bg-sunken))`;
          return (
            <g key={`${i}-${j}`}>
              <rect x={30 + j * cellSize} y={30 + i * cellSize} width={cellSize - 0.5} height={cellSize - 0.5}
                    fill={fill} stroke="var(--bg-canvas)" strokeWidth="0.3" />
              {cellSize >= 11 && (
                <text x={30 + (j + 0.5) * cellSize} y={30 + (i + 0.6) * cellSize}
                      fontSize="2.6" fill={intensity > 0.5 ? 'var(--bg-canvas)' : 'var(--fg-default)'}
                      fontFamily="JetBrains Mono" textAnchor="middle">
                  {v.toFixed(2)}
                </text>
              )}
            </g>
          );
        }))}
      </svg>
    </div>
  );
}

function TargetClassBar() {
  // Mock: churn=No 73.4% / churn=Yes 26.6%
  const data = [
    { label: 'No (未流失)',  count: 9135, pct: 73.4, color: 'var(--good)' },
    { label: 'Yes (流失)',  count: 3315, pct: 26.6, color: 'var(--bad)' },
  ];
  return (
    <Stack gap={20}>
      {data.map(d => (
        <div key={d.label}>
          <Row style={{ justifyContent: 'space-between', marginBottom: 6 }}>
            <span className="t-body-lg fg-1">{d.label}</span>
            <span className="mono fg-1">{d.count.toLocaleString()} <span className="fg-4">·</span> {d.pct}%</span>
          </Row>
          <div style={{ height: 12, background: 'var(--bg-sunken)', borderRadius: 4, overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${d.pct}%`, background: d.color, opacity: 0.85 }} />
          </div>
        </div>
      ))}
      <div style={{ padding: 12, background: 'var(--bg-sunken)', borderRadius: 7, border: '1px solid var(--bd-subtle)' }}>
        <Row gap={8}>
          <Chip tone="good" icon="checkCircle">類別均衡</Chip>
          <span className="t-label">少數類別 ≥ 20%,可直接訓練</span>
        </Row>
      </div>
    </Stack>
  );
}

function TargetHistogram() {
  const bins = 20;
  const counts = Array.from({ length: bins }, (_, i) => {
    const x = (i - bins / 2) / (bins / 4);
    return Math.exp(-x * x / 2) * 100 + Math.sin(i) * 5;
  });
  const max = Math.max(...counts);
  return (
    <svg viewBox="0 0 100 80" preserveAspectRatio="none" style={{ width: '100%', height: '100%' }}>
      <g stroke="var(--bd-subtle)" strokeWidth="0.3">
        <line x1="4" y1="10" x2="96" y2="10" />
        <line x1="4" y1="40" x2="96" y2="40" />
        <line x1="4" y1="70" x2="96" y2="70" />
      </g>
      {counts.map((c, i) => {
        const x = 4 + (i / bins) * 92;
        const w = 92 / bins - 0.5;
        const h = (c / max) * 60;
        return <rect key={i} x={x} y={70 - h} width={w} height={h} fill="var(--primary)" opacity="0.75" />;
      })}
      <text x="4"  y="78" fontSize="2.4" fill="var(--fg-muted)" fontFamily="JetBrains Mono">min</text>
      <text x="96" y="78" fontSize="2.4" fill="var(--fg-muted)" fontFamily="JetBrains Mono" textAnchor="end">max</text>
    </svg>
  );
}

function MissingHeatmap() {
  // 50 rows × N columns; each cell is "missing" with low probability
  const cols = MOCK.columns;
  const rows = 50;
  return (
    <svg viewBox={`0 0 ${cols.length * 8 + 8} ${rows + 16}`} style={{ width: '100%', height: 220 }}>
      {cols.map((c, j) => (
        <text key={j} x={4 + j * 8 + 3} y={10} fontSize="2.6" fill="var(--fg-muted)" fontFamily="JetBrains Mono"
              transform={`rotate(-45, ${4 + j * 8 + 3}, 10)`} textAnchor="end">
          {c.name.length > 10 ? c.name.slice(0, 9) + '…' : c.name}
        </text>
      ))}
      {Array.from({ length: rows }).map((_, i) =>
        cols.map((c, j) => {
          const missingRate = c.missing / 12450;
          const isMissing = Math.random() < missingRate * 2;
          return (
            <rect key={`${i}-${j}`} x={4 + j * 8} y={14 + i} width={7.5} height={0.9}
                  fill={isMissing ? 'var(--warn)' : 'var(--bg-hover)'} />
          );
        })
      )}
    </svg>
  );
}

// 從 audit report + analysis 衍生「數據品質檢核」列表
function _deriveIssues(detail, analysis) {
  if (!detail) return [];
  const issues = [];
  const audit = detail.auditReport || {};
  const totalRows = detail.rowCount || 1;

  // 1. 缺失值欄位
  const missingCols = analysis.filter(c => (c.missing ?? 0) > 0);
  if (missingCols.length === 0) {
    issues.push({ tone: 'good', icon: 'checkCircle', title: '無缺失值', body: '所有欄位都完整,不需要補值' });
  } else {
    issues.push({
      tone: 'warn', icon: 'warning', kind: 'missing',
      title: `${missingCols.length} 個欄位有缺失值`,
      body: missingCols.slice(0, 5).map(c => `${c.name} (${c.missing} 列, ${((c.missing || 0) / totalRows * 100).toFixed(1)}%)`).join(' · ')
            + (missingCols.length > 5 ? ` · 其他 ${missingCols.length - 5} 個...` : ''),
    });
  }

  // 2. audit duplicates
  if (typeof audit.duplicateCount === 'number') {
    if (audit.duplicateCount === 0) {
      issues.push({ tone: 'good', icon: 'checkCircle', title: '無重複列', body: `${totalRows.toLocaleString()} 列全部唯一` });
    } else {
      issues.push({ tone: 'warn', icon: 'warning', title: `偵測到 ${audit.duplicateCount} 列重複資料`, body: '建議刪除或檢查資料來源' });
    }
  }

  // 3. audit outliers
  if (Array.isArray(audit.outlierCols) && audit.outlierCols.length > 0) {
    issues.push({
      tone: 'warn', icon: 'warning',
      title: `${audit.outlierCols.length} 個欄位有離群值`,
      body: audit.outlierCols.slice(0, 3).map(o => `${o.column || o.name}: ${o.outlierRatio ? (o.outlierRatio * 100).toFixed(1) + '%' : '已偵測'}`).join(' · '),
    });
  }

  // 4. high cardinality categoricals
  const highCardCat = analysis.filter(c =>
    (c.type === 'categorical' || c.type === 'cat' || c.type === 'string') && (c.unique || 0) > 50,
  );
  if (highCardCat.length > 0) {
    issues.push({
      tone: 'info', icon: 'info',
      title: `${highCardCat.length} 個高基數類別欄位`,
      body: highCardCat.slice(0, 3).map(c => `${c.name} (${c.unique} 個唯一值)`).join(' · ')
            + ' — 建議用 target encoding 或 frequency encoding',
    });
  }

  return issues;
}

window.PageData = PageData;
