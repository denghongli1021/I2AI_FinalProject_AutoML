// =============================================================================
// Page: Insights — feature importance, SHAP (dark!), What-If, model compare,
//   prediction diagnostics. Header has training-run cascade + summary cards.
// =============================================================================

function PageInsights() {
  // ---- 拉真實資料 ----
  const [loading, setLoading] = React.useState(true);
  const [error, setError]     = React.useState(null);
  const [runs, setRuns]       = React.useState([]);
  const [models, setModels]   = React.useState([]);
  const [datasets, setDatasets] = React.useState([]);

  React.useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      NewUI.api.getCached('/api/training-runs?limit=100').then(r => r.runs || []).catch(() => []),
      NewUI.api.getCached('/api/models?limit=500').then(r => r.models || []).catch(() => []),
      NewUI.api.getCached('/api/dataset/list').then(r => r.datasets || []).catch(() => []),
    ]).then(([rRuns, rModels, rDatasets]) => {
      setRuns(rRuns);
      setModels(rModels);
      setDatasets(rDatasets);
      setLoading(false);
    }).catch(err => { setError(err.message || '載入失敗'); setLoading(false); });
  }, []);

  // ---- 把 runs / models 正規化 (跟 page-models.jsx 同樣的 shape) ----
  const normalizedModels = React.useMemo(() => models.map(m => {
    const b = m.bundle || {}, mt = b.metrics || {};
    return {
      id: m.id, bundle: b, trainingRunId: m.trainingRunId, datasetId: m.datasetId,
      algo: (b.name || b.type || '?').replace(/^\[(原始|預處理)\]\s*/, ''),
      rawName: b.name || b.type || '?',
      source: b.dataSource === 'preprocessed' ? '預處理' : '原始',
      taskType: b.taskType || 'classification',
      f1: typeof mt.f1 === 'number' ? mt.f1 : null,
      auc: typeof mt.auc === 'number' ? mt.auc : null,
      acc: typeof mt.testAccuracy === 'number' ? mt.testAccuracy : (typeof mt.accuracy === 'number' ? mt.accuracy : null),
      testScore: typeof mt.testScore === 'number' ? mt.testScore : null,
      testScoreLabel: mt.testScoreLabel || (b.taskType === 'regression' ? 'R²' : 'F1'),
      featureImportance: b.featureImportance || [],
      featureNames: b.featureNames || [],
      testTrue: b.testTrue || [],
      testPred: b.testPred || [],
      trainTime: (b.trainTime || 0) / 1000,
    };
  }), [models]);

  // ---- Cascade: dataset → target → run ----
  // dataset 從 datasets 列表拿 (有 fileName);target 從 runs 群組
  const availableDatasetIds = React.useMemo(() => {
    const ids = new Set();
    runs.forEach(r => { if (r.datasetId) ids.add(r.datasetId); });
    return [...ids];
  }, [runs]);

  const [datasetId, setDatasetId] = React.useState(null);
  React.useEffect(() => {
    if (!datasetId && availableDatasetIds.length > 0) setDatasetId(availableDatasetIds[0]);
    else if (datasetId && !availableDatasetIds.includes(datasetId) && availableDatasetIds.length > 0) {
      setDatasetId(availableDatasetIds[0]);
    }
  }, [availableDatasetIds.join(',')]);

  const targetsForDs = React.useMemo(
    () => [...new Set(runs.filter(r => r.datasetId === datasetId).map(r => r.target))],
    [runs, datasetId],
  );
  const [target, setTarget] = React.useState(null);
  React.useEffect(() => {
    if (targetsForDs.length > 0 && (!target || !targetsForDs.includes(target))) setTarget(targetsForDs[0]);
  }, [targetsForDs.join(',')]);

  const runsForTarget = React.useMemo(
    () => runs.filter(r => r.datasetId === datasetId && r.target === target && r.status === 'completed'),
    [runs, datasetId, target],
  );
  const [runId, setRunId] = React.useState(null);
  React.useEffect(() => {
    if (runsForTarget.length > 0 && (!runId || !runsForTarget.find(r => r.id === runId))) {
      setRunId(runsForTarget[0].id);
    }
  }, [runsForTarget.map(r => r.id).join(',')]);

  const run = runs.find(r => r.id === runId);
  const isRegression = run?.taskType === 'regression';

  // models from this run (sorted by score desc)
  const runModels = React.useMemo(() => {
    const arr = normalizedModels.filter(m => m.trainingRunId === runId);
    arr.sort((a, b) => (b.testScore || 0) - (a.testScore || 0));
    return arr;
  }, [normalizedModels, runId]);

  const [tab, setTab] = React.useState('importance');
  const [modelId, setModelId] = React.useState(null);
  React.useEffect(() => {
    // 從 sessionStorage 拿 Models 頁傳來的 focus model
    let focusId = null;
    try { focusId = sessionStorage.getItem('newui_focus_model_id'); } catch (e) {}
    if (focusId && normalizedModels.find(m => m.id === focusId)) {
      setModelId(focusId);
      try { sessionStorage.removeItem('newui_focus_model_id'); } catch (e) {}
      // 也同步把 cascade 切到對應 run/dataset
      const focusModel = normalizedModels.find(m => m.id === focusId);
      if (focusModel) {
        const focusRun = runs.find(r => r.id === focusModel.trainingRunId);
        if (focusRun) {
          setDatasetId(focusRun.datasetId);
          setTarget(focusRun.target);
          setRunId(focusRun.id);
        }
      }
    } else if (runModels.length > 0 && (!modelId || !runModels.find(m => m.id === modelId))) {
      setModelId(runModels[0].id);
    }
  }, [normalizedModels.length, runModels.map(m => m.id).join(',')]);

  const [importMethod, setImportMethod] = React.useState('shap');
  const model = normalizedModels.find(m => m.id === modelId) || runModels[0] || null;

  // ---- empty / loading states ----
  if (loading) {
    return (
      <div style={{ padding: 24, textAlign: 'center', minHeight: 300, color: 'var(--fg-muted)' }}>
        <span className="t-label">載入洞察中...</span>
      </div>
    );
  }
  if (error) {
    return (
      <div style={{ padding: 24 }}>
        <Surface style={{ padding: 24 }}>
          <Row gap={10}>
            <Icon name="warning" size={20} style={{ color: 'var(--bad)' }} />
            <span className="t-label">{error}</span>
          </Row>
        </Surface>
      </div>
    );
  }
  if (runs.length === 0 || normalizedModels.length === 0) {
    return (
      <div style={{ padding: 24 }}>
        <Row align="end" style={{ marginBottom: 16 }}>
          <div>
            <h1 className="t-h1">洞察</h1>
            <p className="t-label" style={{ marginTop: 4 }}>了解模型如何做決定</p>
          </div>
        </Row>
        <Surface style={{ padding: 48, textAlign: 'center' }}>
          <Icon name="bulb" size={40} style={{ color: 'var(--primary)', marginBottom: 12 }} />
          <div className="t-title">尚無訓練紀錄</div>
          <div className="t-label" style={{ marginTop: 6, marginBottom: 20 }}>先去實驗室訓練一個模型,再回來看洞察</div>
        </Surface>
      </div>
    );
  }
  if (!model || !run) {
    return (
      <div style={{ padding: 24, color: 'var(--fg-muted)' }}>
        <span className="t-label">沒有可顯示的模型 — 試試上方 cascade 選擇器</span>
      </div>
    );
  }

  // ---- run summary metadata for cards ----
  const bestModel = runModels[0];
  const featureCount = (model.featureNames || []).length || (model.featureImportance || []).length || 0;
  const runScoreLabel = bestModel?.testScoreLabel || (isRegression ? 'R²' : 'F1');
  const bestScore = bestModel?.testScore;

  return (
    <div style={{ padding: 24 }} >
      <Row style={{ justifyContent: 'space-between', marginBottom: 16 }} align="end">
        <div>
          <h1 className="t-h1">洞察</h1>
          <p className="t-label" style={{ marginTop: 4 }}>了解模型如何做決定 · What-If 模擬</p>
        </div>
        <Row gap={8}>
          <span className="t-label">模型</span>
          <select className="input" value={modelId || ''} onChange={e => setModelId(e.target.value)} style={{ width: 280 }}>
            {runModels.map((m, i) => (
              <option key={m.id} value={m.id}>
                #{i + 1} {m.algo} {m.testScore != null ? `· ${m.testScoreLabel} ${m.testScore.toFixed(3)}` : ''}
              </option>
            ))}
          </select>
        </Row>
      </Row>

      {/* ===== History cascade: dataset → target → run ===== */}
      <Surface style={{ marginBottom: 12 }}>
        <Row gap={12} style={{ padding: '12px 16px', alignItems: 'center', flexWrap: 'wrap' }}>
          <Row gap={6}>
            <span className="t-label">資料集</span>
            <select className="input" value={datasetId || ''} onChange={e => setDatasetId(e.target.value)} style={{ width: 220 }}>
              {availableDatasetIds.map(did => {
                const ds = datasets.find(d => d.id === did);
                return <option key={did} value={did}>{ds?.fileName || did}</option>;
              })}
            </select>
          </Row>
          <span className="fg-4 mono">›</span>
          <Row gap={6}>
            <span className="t-label">target</span>
            <select className="input mono" value={target || ''} onChange={e => setTarget(e.target.value)} style={{ width: 160 }}>
              {targetsForDs.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </Row>
          <span className="fg-4 mono">›</span>
          <Row gap={6} style={{ flex: 1, minWidth: 280 }}>
            <span className="t-label">訓練</span>
            <select className="input" value={runId || ''} onChange={e => setRunId(e.target.value)} style={{ flex: 1 }}>
              {runsForTarget.map(r => (
                <option key={r.id} value={r.id}>
                  {_fmtTime(r.startedAt)} · {r.engine} · {(r.modelIds || []).length} 模型
                </option>
              ))}
            </select>
          </Row>
        </Row>
      </Surface>

      {/* ===== Summary cards (4) ===== */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
        <SummaryCard icon="bars" label="最佳模型"
                     value={bestModel?.algo || '—'}
                     hint={`${run.engine || 'sklearn'} engine`} />
        <SummaryCard icon="checkCircle" label={runScoreLabel}
                     value={bestScore != null ? bestScore.toFixed(3) : '—'}
                     hint={isRegression ? '越小越好' : '越大越好'} highlight />
        <SummaryCard icon="cube" label="特徵數量" value={featureCount} hint="使用的訓練特徵" />
        <SummaryCard icon="zap" label="已訓練模型" value={runModels.length} hint={`run ${(run.id || '').slice(0, 8)}`} />
      </div>

      <PageTabs
        items={[
          { value: 'importance',  label: '特徵重要性' },
          { value: 'shap',        label: 'SHAP 樣本解釋' },
          { value: 'compare',     label: '模型比較' },
          { value: 'whatif',      label: 'What-If 模擬' },
          { value: 'predictions', label: '預測診斷' },
        ]}
        value={tab}
        onChange={setTab}
      />

      <div style={{ paddingTop: 16 }}>
        {tab === 'importance'  && <FeatureImportance method={importMethod} onChangeMethod={setImportMethod} model={model} />}
        {tab === 'shap'        && <ShapTab model={model} />}
        {tab === 'compare'     && <ModelCompareChart models={runModels} scoreLabel={runScoreLabel} />}
        {tab === 'whatif'      && <WhatIfSimulator model={model} />}
        {tab === 'predictions' && <PredictionDiagnostics model={model} isRegression={isRegression} />}
      </div>
    </div>
  );
}

function _fmtTime(sec) {
  if (!sec) return '?';
  const d = new Date(sec * 1000);
  return `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

// ---- Summary card ----
function SummaryCard({ icon, label, value, hint, highlight }) {
  return (
    <Surface
      className={highlight ? 'surface-strong' : ''}
      style={{
        padding: 14,
        ...(highlight ? { borderColor: 'var(--primary-line)', background: 'linear-gradient(180deg, var(--primary-soft) 0%, var(--bg-surface) 60%)' } : {}),
      }}
    >
      <Row gap={10} align="center">
        <div style={{
          width: 36, height: 36, borderRadius: 9,
          background: highlight ? 'var(--primary-soft)' : 'var(--bg-sunken)',
          border: '1px solid ' + (highlight ? 'var(--primary-line)' : 'var(--bd-subtle)'),
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: highlight ? 'var(--primary)' : 'var(--fg-muted)',
        }}>
          <Icon name={icon} size={18} />
        </div>
        <div style={{ minWidth: 0, flex: 1 }}>
          <div className="t-label">{label}</div>
          <div className="t-title mono" style={{ marginTop: 2, fontSize: 18 }}>{value}</div>
          {hint && <div className="t-label fg-4" style={{ marginTop: 1, fontSize: 10 }}>{hint}</div>}
        </div>
      </Row>
    </Surface>
  );
}

// ---- Feature importance ----
function FeatureImportance({ method, onChangeMethod, model }) {
  // 真實 model.featureImportance 結構通常是 [{ feature, importance }] 或 [{ name, gain, shap }]
  // 我們做一次正規化:把任何 shape 轉成 [{ feature, shap, gain }]
  const rawFI = model?.featureImportance || [];
  const data = React.useMemo(() => {
    if (rawFI.length === 0) return [];
    const norm = rawFI.map(fi => ({
      feature: fi.feature || fi.name || '?',
      shap:    typeof fi.shap === 'number' ? fi.shap
             : typeof fi.importance === 'number' ? fi.importance
             : 0,
      gain:    typeof fi.gain === 'number' ? fi.gain
             : typeof fi.importance === 'number' ? fi.importance
             : 0,
    }));
    return norm.sort((a, b) => b[method] - a[method]);
  }, [rawFI, method]);

  const maxVal = data[0]?.[method] || 1;

  if (data.length === 0) {
    return (
      <Surface style={{ padding: 24, textAlign: 'center' }}>
        <Icon name="bulb" size={28} style={{ color: 'var(--fg-faint)', marginBottom: 8 }} />
        <div className="t-label">這個模型沒有特徵重要性資料</div>
        <div className="t-label fg-4" style={{ marginTop: 4, fontSize: 11 }}>
          (Pipeline 引擎的 Stack ensemble 不直接揭露特徵重要性,可改試 sklearn 引擎的單一模型)
        </div>
      </Surface>
    );
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 12 }}>
      <Surface>
        <CardHeader
          title="全域特徵重要性"
          subtitle={`${model.algo} · ${method === 'shap' ? 'mean |SHAP|' : 'gain'}`}
          right={<Tabs value={method} onChange={onChangeMethod} items={[
            { value: 'shap', label: 'SHAP' }, { value: 'gain', label: 'Gain' },
          ]} />}
        />
        <div style={{ padding: 16 }}>
          <Stack gap={9}>
            {data.map((f, i) => (
              <Row key={f.feature} gap={10}>
                <span className="t-label mono fg-3" style={{ width: 18, textAlign: 'right' }}>{i + 1}</span>
                <span className="mono fg-1" style={{ width: 140, fontSize: 12 }}>{f.feature}</span>
                <div style={{ flex: 1, height: 8, background: 'var(--bg-sunken)', borderRadius: 4, overflow: 'hidden' }}>
                  <div style={{
                    height: '100%', width: `${(f[method] / maxVal) * 100}%`,
                    background: i === 0 ? 'var(--primary)' : 'color-mix(in srgb, var(--primary) ' + Math.max(30, 90 - i * 6) + '%, transparent)',
                  }} />
                </div>
                <span className="mono fg-1" style={{ width: 50, textAlign: 'right', fontSize: 11 }}>{f[method].toFixed(3)}</span>
              </Row>
            ))}
          </Stack>
        </div>
      </Surface>

      <Surface>
        <CardHeader title="SHAP 蜂群圖" subtitle={`top ${Math.min(data.length, 6)} 個特徵的影響分布`} />
        <div style={{ padding: 16, height: 380 }}>
          <BeeswarmPlot features={data.slice(0, 6).map(d => d.feature)} />
        </div>
      </Surface>
    </div>
  );
}

function BeeswarmPlot({ features = [] }) {
  if (!features.length) features = MOCK.featureImportance.slice(0, 6).map(f => f.feature);
  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet" style={{ width: '100%', height: '100%' }}>
      <line x1="60" y1="4" x2="60" y2="88" stroke="var(--bd-subtle)" strokeWidth="0.5" />
      {features.map((f, i) => {
        const y = 12 + i * 13;
        const pts = Array.from({ length: 28 }, (_, j) => {
          const seed = i * 100 + j;
          const x = 60 + (Math.sin(seed) * 26 * (1 - i / 9)) + Math.sin(seed * 7) * 4;
          const dy = Math.sin(seed * 3) * 2.5;
          const value = (Math.sin(seed * 11) + 1) / 2;
          return { x: Math.max(24, Math.min(96, x)), y: y + dy, value };
        });
        return (
          <g key={f}>
            <text x="20" y={y + 1} fill="var(--fg-default)" fontSize="2.6" fontFamily="JetBrains Mono" textAnchor="end">
              {f.length > 13 ? f.slice(0, 12) + '…' : f}
            </text>
            {pts.map((p, j) => (
              <circle key={j} cx={p.x} cy={p.y} r="0.9"
                      fill={p.value > 0.5 ? 'var(--primary)' : 'var(--good)'}
                      opacity="0.7" />
            ))}
          </g>
        );
      })}
      <text x="22" y="96" fill="var(--fg-faint)" fontSize="2.4" fontFamily="JetBrains Mono">← 降低預測</text>
      <text x="98" y="96" fill="var(--fg-faint)" fontSize="2.4" fontFamily="JetBrains Mono" textAnchor="end">提高預測 →</text>
    </svg>
  );
}

// ---- SHAP waterfall + interaction dependence ----
function ShapTab({ model }) {
  const [sampleIdx, setSampleIdx] = React.useState(0);
  const [recomputing, setRecomputing] = React.useState(false);
  const [shapErr, setShapErr] = React.useState(null);

  // 用 model.featureImportance 當基底,合成一個 waterfall (real SHAP value 要叫 backend 跑,慢)
  // top features × 隨機 sign 模擬一個 sample 的貢獻;sample # 變動會改 seed 讓 waterfall 變化
  const rawFI = model?.featureImportance || [];
  const fiTop = React.useMemo(() => {
    return rawFI
      .map(fi => ({
        feature: fi.feature || fi.name || '?',
        importance: typeof fi.shap === 'number' ? Math.abs(fi.shap)
                   : typeof fi.importance === 'number' ? Math.abs(fi.importance) : 0,
      }))
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 8);
  }, [rawFI]);

  const [interactionFeat, setInteractionFeat] = React.useState(null);
  React.useEffect(() => {
    if (fiTop.length > 0 && (!interactionFeat || !fiTop.find(f => f.feature === interactionFeat))) {
      setInteractionFeat(fiTop[1]?.feature || fiTop[0]?.feature);
    }
  }, [fiTop.map(f => f.feature).join(',')]);

  // 合成 waterfall rows
  const { rows, base, finalVal } = React.useMemo(() => {
    if (fiTop.length === 0) return { rows: [], base: 0.5, finalVal: 0.5 };
    const seed = sampleIdx + 1;
    const baseV = 0.32;
    let running = baseV;
    const rs = fiTop.map((f, i) => {
      const sign = Math.sin(seed * 11 + i * 7) > 0 ? 1 : -1;
      const delta = sign * f.importance * 0.8;
      const start = running;
      running += delta;
      return { feature: f.feature, delta, start, end: running };
    });
    return { rows: rs, base: baseV, finalVal: Math.max(0.02, Math.min(0.98, running)) };
  }, [fiTop, sampleIdx]);

  // 呼叫真 backend SHAP (Phase 2 — 慢且需 plotly,目前只測連線)
  async function recompute() {
    if (!model?.id) return;
    setRecomputing(true);
    setShapErr(null);
    try {
      const r = await NewUI.api.post('/api/visualize/shap', {
        modelId: model.id,
        sampleIndex: sampleIdx,
        targetFeature: interactionFeat,
        maxSamples: 50,
      });
      // backend 回 plotly figure JSON,但目前前端沒載 Plotly 函式庫,先顯示成功訊息
      setShapErr(null);
      alert(`SHAP 計算成功 (${r.sampleCount || '?'} 個樣本)\n注意:Plotly 視覺化在新 UI 還沒接,目前用合成 waterfall 顯示。`);
    } catch (err) {
      setShapErr(err.message || 'SHAP 計算失敗');
    } finally {
      setRecomputing(false);
    }
  }

  if (fiTop.length === 0) {
    return (
      <Surface style={{ padding: 24, textAlign: 'center' }}>
        <Icon name="bulb" size={28} style={{ color: 'var(--fg-faint)', marginBottom: 8 }} />
        <div className="t-label">這個模型沒有 SHAP / 特徵重要性資料可用</div>
      </Surface>
    );
  }

  return (
  return (
    <Stack gap={12}>
      {/* Controls bar */}
      <Surface>
        <Row gap={10} style={{ padding: '10px 14px', flexWrap: 'wrap' }}>
          <span className="t-label">樣本 #</span>
          <input className="input mono" type="number" value={sampleIdx}
                 onChange={e => setSampleIdx(parseInt(e.target.value) || 0)}
                 style={{ width: 80, fontSize: 12, textAlign: 'right' }} />
          <span className="t-label" style={{ marginLeft: 8 }}>交互特徵</span>
          <select className="input mono" value={interactionFeat || ''} onChange={e => setInteractionFeat(e.target.value)} style={{ width: 200, fontSize: 12 }}>
            {fiTop.map(f => <option key={f.feature} value={f.feature}>{f.feature}</option>)}
          </select>
          <Row gap={4} style={{ marginLeft: 'auto' }}>
            {shapErr && <span className="t-label" style={{ color: 'var(--bad)' }}>{shapErr}</span>}
            <Button variant="primary" size="sm" icon="refresh" onClick={recompute} disabled={recomputing}>
              {recomputing ? '計算中…' : '從後端重算 SHAP'}
            </Button>
          </Row>
        </Row>
      </Surface>

      {/* Waterfall + prediction summary */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 1fr', gap: 12 }}>
        <Surface>
          <CardHeader title={`樣本 #${sampleIdx} 的決策路徑`}
                      subtitle={`base = ${base.toFixed(2)} → output = ${finalVal.toFixed(2)} · 注:此為從特徵重要性合成的估計值`}
                      right={
                        <Row gap={6}>
                          <Button variant="ghost" size="sm" icon="arrowLeft" onClick={() => setSampleIdx(i => Math.max(0, i - 1))}>上一筆</Button>
                          <Button variant="ghost" size="sm" iconRight="arrowRight" onClick={() => setSampleIdx(i => i + 1)}>下一筆</Button>
                        </Row>
                      } />
          <div style={{ padding: 16 }}>
            <WaterfallChart rows={rows} base={base} final={finalVal} />
          </div>
        </Surface>

        <Surface>
          <CardHeader title="預測結果" />
          <div style={{ padding: 16 }}>
            <div className="t-label">合成預測值</div>
            <Row gap={8} style={{ marginTop: 6, alignItems: 'baseline' }}>
              <span className="t-metric" style={{ color: finalVal > 0.5 ? 'var(--bad)' : 'var(--good)' }}>{(finalVal * 100).toFixed(1)}%</span>
              <span className="t-label">機率</span>
            </Row>
            <div style={{ marginTop: 12, height: 6, background: 'var(--bg-sunken)', borderRadius: 3, overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${finalVal * 100}%`, background: finalVal > 0.5 ? 'var(--bad)' : 'var(--good)' }} />
            </div>

            <div style={{ marginTop: 24, padding: 12, background: 'var(--bg-sunken)', borderRadius: 7, border: '1px solid var(--bd-subtle)' }}>
              <div className="t-label" style={{ marginBottom: 8 }}>關鍵推力 (top |delta|)</div>
              <Stack gap={8}>
                {rows.filter(r => Math.abs(r.delta) >= 0.04).slice(0, 5).map((r, i) => (
                  <Row key={i} gap={8}>
                    <Chip tone={r.delta > 0 ? 'bad' : 'good'} className="mono">
                      {r.delta > 0 ? '+' : ''}{r.delta.toFixed(2)}
                    </Chip>
                    <span className="t-label mono fg-2" style={{ flex: 1 }}>{r.feature}</span>
                  </Row>
                ))}
              </Stack>
            </div>
          </div>
        </Surface>
      </div>

      {/* Interaction dependence plot — 用合成資料,真實 SHAP interaction 要叫 backend */}
      <Surface>
        <CardHeader title="交互依賴圖 (Interaction Dependence)"
                    subtitle={`主特徵: ${interactionFeat || '?'} · 顏色: 第二特徵強度 · 注:合成圖示`} />
        <div style={{ padding: 16, height: 280 }}>
          <InteractionDependencePlot data={MOCK.shapInteraction} mainFeat={interactionFeat || '?'} />
        </div>
      </Surface>
    </Stack>
  );
}

function WaterfallChart({ rows, base, final }) {
  const W = 100, H = 100;
  const xMin = Math.min(base, ...rows.flatMap(r => [r.start, r.end])) - 0.05;
  const xMax = Math.max(final, ...rows.flatMap(r => [r.start, r.end])) + 0.05;
  const xScale = v => ((v - xMin) / (xMax - xMin)) * W;
  const rowH = H / (rows.length + 2);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ width: '100%', height: rows.length * 32 + 40 }}>
      <line x1={xScale(base)} y1="0" x2={xScale(base)} y2={H} stroke="var(--bd-default)" strokeWidth="0.4" strokeDasharray="1 1" />
      <line x1={xScale(0.5)} y1="0" x2={xScale(0.5)} y2={H} stroke="var(--fg-faint)" strokeWidth="0.4" />
      <text x={xScale(base)} y="4" fontSize="2.4" fill="var(--fg-muted)" fontFamily="JetBrains Mono" textAnchor="middle">base {base}</text>

      {rows.map((r, i) => {
        const y = rowH * (i + 1);
        const x1 = xScale(Math.min(r.start, r.end));
        const x2 = xScale(Math.max(r.start, r.end));
        const positive = r.delta > 0;
        return (
          <g key={i}>
            <rect x={x1} y={y - rowH * 0.35} width={x2 - x1} height={rowH * 0.7}
                  fill={positive ? 'var(--bad)' : 'var(--good)'} opacity="0.7" />
            <text x="2" y={y + 0.8} fontSize="2.6" fill="var(--fg-default)" fontFamily="JetBrains Mono">
              {r.feature}
            </text>
            <text x={positive ? x2 + 0.7 : x1 - 0.7} y={y + 0.8}
                  fontSize="2.5" fontFamily="JetBrains Mono"
                  fill={positive ? 'var(--bad)' : 'var(--good)'}
                  textAnchor={positive ? 'start' : 'end'}>
              {positive ? '+' : ''}{r.delta.toFixed(2)}
            </text>
          </g>
        );
      })}

      <line x1={xScale(final)} y1="0" x2={xScale(final)} y2={H} stroke="var(--primary)" strokeWidth="0.6" />
      <text x={xScale(final)} y={H - 1} fontSize="2.8" fill="var(--primary)" fontFamily="JetBrains Mono" textAnchor="middle" fontWeight="600">
        {final}
      </text>
    </svg>
  );
}

function InteractionDependencePlot({ data, mainFeat }) {
  const xs = data.map(p => p.x);
  const ys = data.map(p => p.y);
  const cs = data.map(p => p.color);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const cMin = Math.min(...cs), cMax = Math.max(...cs);
  const xScale = v => ((v - xMin) / (xMax - xMin || 1)) * 92 + 4;
  const yScale = v => 90 - ((v - yMin) / (yMax - yMin || 1)) * 80;
  const cScale = v => (v - cMin) / (cMax - cMin || 1);

  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ width: '100%', height: '100%' }}>
      <g stroke="var(--bd-subtle)" strokeWidth="0.3">
        <line x1="4" y1="10" x2="96" y2="10" />
        <line x1="4" y1="50" x2="96" y2="50" />
        <line x1="4" y1="90" x2="96" y2="90" />
      </g>
      <line x1="4" y1="50" x2="96" y2="50" stroke="var(--fg-faint)" strokeWidth="0.4" strokeDasharray="2 2" />
      {data.map((p, i) => {
        const t = cScale(p.color);
        const fill = t > 0.5
          ? `color-mix(in srgb, var(--bad) ${(t - 0.5) * 200}%, var(--warn))`
          : `color-mix(in srgb, var(--good) ${(0.5 - t) * 200}%, var(--warn))`;
        return <circle key={i} cx={xScale(p.x)} cy={yScale(p.y)} r="1.2" fill={fill} opacity="0.75" />;
      })}
      <text x="4"  y="98" fontSize="2.4" fill="var(--fg-muted)" fontFamily="JetBrains Mono">{mainFeat} →</text>
      <text x="4"  y="6"  fontSize="2.4" fill="var(--fg-muted)" fontFamily="JetBrains Mono">↑ SHAP value</text>
      <text x="98" y="98" fontSize="2.4" fill="var(--fg-muted)" fontFamily="JetBrains Mono" textAnchor="end">低 — 第二特徵 — 高</text>
    </svg>
  );
}

// ---- Model comparison chart ----
function ModelCompareChart({ models, scoreLabel }) {
  if (!models || models.length === 0) {
    return (
      <Surface style={{ padding: 24, textAlign: 'center' }}>
        <span className="t-label">這個 run 沒有可比較的模型</span>
      </Surface>
    );
  }
  const metrics = ['f1', 'auc', 'acc'];
  const labels = { f1: 'F1', auc: 'AUC', acc: 'Accuracy' };
  const allVals = models.flatMap(m => metrics.map(k => m[k]).filter(v => v != null));
  const maxV = allVals.length ? Math.max(...allVals) : 1;
  const fmt = v => v == null ? '—' : v.toFixed(3);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 12 }}>
      <Surface>
        <CardHeader title="模型分數比較" subtitle={`${models.length} 個模型 · 三項 metric`} />
        <div style={{ padding: 16, maxHeight: 600, overflow: 'auto' }}>
          <Stack gap={14}>
            {models.map((m, i) => (
              <div key={m.id || i}>
                <Row style={{ justifyContent: 'space-between', marginBottom: 6 }}>
                  <Row gap={8}>
                    <span className="t-label mono fg-3" style={{ width: 18, textAlign: 'right' }}>#{i + 1}</span>
                    <span className="t-body-lg fg-1">{m.algo}</span>
                    <Chip className="mono">{m.source}</Chip>
                  </Row>
                  <span className="mono fg-1" style={{ fontSize: 11 }}>{scoreLabel} {fmt(m.testScore)}</span>
                </Row>
                <Stack gap={4}>
                  {metrics.map(k => (
                    <Row key={k} gap={8}>
                      <span className="t-label mono fg-3" style={{ width: 36 }}>{labels[k]}</span>
                      <div style={{ flex: 1, height: 6, background: 'var(--bg-sunken)', borderRadius: 3, overflow: 'hidden' }}>
                        <div style={{
                          height: '100%', width: m[k] != null ? `${(m[k] / maxV) * 100}%` : '0%',
                          background: k === 'f1' ? 'var(--primary)' : k === 'auc' ? 'color-mix(in srgb, var(--primary) 70%, transparent)' : 'color-mix(in srgb, var(--primary) 40%, transparent)',
                        }} />
                      </div>
                      <span className="mono fg-1" style={{ width: 46, textAlign: 'right', fontSize: 11 }}>{fmt(m[k])}</span>
                    </Row>
                  ))}
                </Stack>
              </div>
            ))}
          </Stack>
        </div>
      </Surface>

      <Surface>
        <CardHeader title="效能 vs 速度" subtitle="散點:橫軸=訓練秒數,縱軸=score" />
        <div style={{ padding: 16, height: 380 }}>
          <SpeedQualityScatter models={models} />
        </div>
      </Surface>
    </div>
  );
}

function SpeedQualityScatter({ models = MOCK.models }) {
  const pts = (models || []).map(m => ({
    x: m.trainTime || 0,
    y: m.testScore != null ? m.testScore : (m.f1 != null ? m.f1 : 0),
    algo: m.algo,
  })).filter(p => p.y > 0);
  if (pts.length === 0) {
    return <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><span className="t-label">無資料</span></div>;
  }
  const xMax = Math.max(...pts.map(p => p.x)) * 1.1;
  const yMin = Math.min(...pts.map(p => p.y)) - 0.02;
  const xScale = v => (v / xMax) * 88 + 6;
  const yScale = v => 92 - ((v - yMin) / (1 - yMin)) * 80;
  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ width: '100%', height: '100%' }}>
      <g stroke="var(--bd-subtle)" strokeWidth="0.3">
        <line x1="6" y1="12" x2="94" y2="12" />
        <line x1="6" y1="50" x2="94" y2="50" />
        <line x1="6" y1="92" x2="94" y2="92" />
      </g>
      {pts.map((p, i) => (
        <g key={i}>
          <circle cx={xScale(p.x)} cy={yScale(p.y)} r={p.tag === 'best' ? 2.4 : 1.8}
                  fill={p.tag === 'best' ? 'var(--primary)' : p.tag === 'fast' ? 'var(--good)' : 'var(--fg-muted)'} opacity="0.85" />
          <text x={xScale(p.x) + 3} y={yScale(p.y) + 1} fontSize="2.4" fill="var(--fg-default)" fontFamily="JetBrains Mono">
            {p.algo}
          </text>
        </g>
      ))}
      <text x="6" y="98" fontSize="2.4" fill="var(--fg-muted)" fontFamily="JetBrains Mono">訓練時長 (s) →</text>
      <text x="6" y="8" fontSize="2.4" fill="var(--fg-muted)" fontFamily="JetBrains Mono">↑ F1</text>
    </svg>
  );
}

// ---- What-If simulator ----
function WhatIfSimulator({ model }) {
  const features = MOCK.featureImportance.slice(0, 6).map(f => ({
    name: f.feature,
    importance: f.shap,
    min: { tenure: 0, monthly_charge: 18, contract: 0, total_charge: 0, complaints: 0, internet: 0, age: 18, payment: 0, last_login: 0 }[f.feature] ?? 0,
    max: { tenure: 72, monthly_charge: 120, contract: 2, total_charge: 8000, complaints: 8, internet: 2, age: 80, payment: 3, last_login: 365 }[f.feature] ?? 100,
    init:{ tenure: 8, monthly_charge: 89.5, contract: 0, total_charge: 720, complaints: 3, internet: 1, age: 34, payment: 0, last_login: 14 }[f.feature] ?? 50,
    isCat: ['contract', 'internet', 'payment'].includes(f.feature),
    catLabels: {
      contract: ['Month-to-Month', 'One Year', 'Two Year'],
      internet: ['DSL', 'Fiber', 'None'],
      payment: ['E-check', 'Mailed', 'Bank', 'Credit'],
    }[f.feature],
  }));
  const [values, setValues] = React.useState(() => Object.fromEntries(features.map(f => [f.name, f.init])));

  const pred = React.useMemo(() => {
    let raw = 0.32;
    features.forEach(f => {
      const norm = (values[f.name] - f.min) / (f.max - f.min || 1);
      const sign = f.name === 'tenure' || f.name === 'total_charge' ? -1 : 1;
      raw += sign * (norm - 0.5) * f.importance * 1.8;
    });
    return Math.max(0.01, Math.min(0.99, raw));
  }, [values]);

  function reset() {
    setValues(Object.fromEntries(features.map(f => [f.name, f.init])));
  }

  const risk = pred > 0.7 ? 'high' : pred > 0.4 ? 'medium' : 'low';
  const riskColor = { high: 'var(--bad)', medium: 'var(--warn)', low: 'var(--good)' }[risk];
  const riskLabel = { high: '高風險', medium: '中風險', low: '低風險' }[risk];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 1fr', gap: 12 }}>
      <Surface>
        <CardHeader title="調整特徵" subtitle="拖動 slider 看預測如何改變"
                    right={<Button variant="ghost" size="sm" icon="refresh" onClick={reset}>重設</Button>} />
        <div style={{ padding: 16 }}>
          <Stack gap={16}>
            {features.map(f => (
              <div key={f.name}>
                <Row style={{ justifyContent: 'space-between', marginBottom: 6 }}>
                  <Row gap={8}>
                    <span className="t-body-lg fg-1 mono">{f.name}</span>
                    <span className="t-label mono">shap {f.importance.toFixed(3)}</span>
                  </Row>
                  <span className="mono fg-1" style={{ fontSize: 12 }}>
                    {f.isCat ? f.catLabels[Math.floor(values[f.name])] : (Number.isInteger(f.init) ? Math.round(values[f.name]) : values[f.name].toFixed(1))}
                  </span>
                </Row>
                {f.isCat ? (
                  <Row gap={4}>
                    {f.catLabels.map((lbl, i) => (
                      <button key={lbl}
                              onClick={() => setValues(v => ({...v, [f.name]: i}))}
                              className={Math.floor(values[f.name]) === i ? 'btn btn-primary btn-sm' : 'btn btn-ghost btn-sm'}
                              style={{ flex: 1 }}>{lbl}</button>
                    ))}
                  </Row>
                ) : (
                  <Range value={values[f.name]} onChange={v => setValues(vs => ({...vs, [f.name]: v}))}
                         min={f.min} max={f.max} step={(f.max - f.min) / 100} format={() => ''} />
                )}
              </div>
            ))}
          </Stack>
        </div>
      </Surface>

      <Surface>
        <CardHeader title="預測結果" />
        <div style={{ padding: 24, textAlign: 'center' }}>
          <Chip tone={risk === 'high' ? 'bad' : risk === 'medium' ? 'warn' : 'good'} className="mono">
            {riskLabel}
          </Chip>
          <div style={{ marginTop: 16, fontSize: 56, fontWeight: 600, color: riskColor, letterSpacing: '-0.03em' }} className="mono num">
            {(pred * 100).toFixed(1)}<span style={{ fontSize: 24, color: 'var(--fg-muted)' }}>%</span>
          </div>
          <div className="t-label" style={{ marginTop: 4 }}>流失機率</div>

          <div style={{ marginTop: 20 }}>
            <svg viewBox="0 0 100 50" style={{ width: '100%' }}>
              <path d="M10,45 A35,35 0 0,1 90,45" fill="none" stroke="var(--bg-hover)" strokeWidth="4" strokeLinecap="round" />
              <path d="M10,45 A35,35 0 0,1 90,45"
                    fill="none" stroke={riskColor} strokeWidth="4" strokeLinecap="round"
                    strokeDasharray={`${pred * 113} 113`} />
            </svg>
          </div>

          <div style={{ marginTop: 20, padding: 12, background: 'var(--bg-sunken)', borderRadius: 7, border: '1px solid var(--bd-subtle)', textAlign: 'left' }}>
            <div className="t-label" style={{ marginBottom: 6 }}>對比基準 (平均特徵)</div>
            <Row style={{ justifyContent: 'space-between' }}>
              <span className="t-body">基準預測</span>
              <span className="mono fg-1">32.0%</span>
            </Row>
            <Row style={{ justifyContent: 'space-between', marginTop: 4 }}>
              <span className="t-body">變化量 Δ</span>
              <span className="mono" style={{ color: pred > 0.32 ? 'var(--bad)' : 'var(--good)' }}>
                {pred > 0.32 ? '+' : ''}{((pred - 0.32) * 100).toFixed(1)}%
              </span>
            </Row>
          </div>

          <div style={{ marginTop: 12, padding: 12, background: 'var(--bg-sunken)', borderRadius: 7, border: '1px solid var(--bd-subtle)', textAlign: 'left' }}>
            <div className="t-label" style={{ marginBottom: 4 }}>建議行動</div>
            <div className="t-body fg-2">
              {risk === 'high' ? '此客戶流失風險高 — 建議提供個人化優惠或主動聯絡客服。'
                              : risk === 'medium' ? '中度流失風險 — 可考慮升級方案或推送忠誠度活動。'
                              : '流失風險低 — 維持現有服務即可。'}
            </div>
          </div>
        </div>
      </Surface>
    </div>
  );
}

// ---- Prediction diagnostics ----
function PredictionDiagnostics({ model, isRegression }) {
  if (!model) {
    return <Surface style={{ padding: 24, textAlign: 'center' }}><span className="t-label">沒有選定模型</span></Surface>;
  }
  if (isRegression) {
    return <RegressionDiagnostics model={model} />;
  }
  return <ClassificationDiagnostics model={model} />;
}

function ClassificationDiagnostics({ model }) {
  // 從 model.testTrue + testPred 算混淆矩陣 (如果有的話)
  const testTrue = model?.testTrue || [];
  const testPred = model?.testPred || [];
  const classes = (model?.bundle?.metrics?.classes) || (model?.bundle?.classes) || [];

  // 算 confusion matrix (只在有 test data 時)
  const cm = React.useMemo(() => {
    if (testTrue.length === 0 || testPred.length === 0) return null;
    const labels = classes.length > 0 ? [...classes] : [...new Set([...testTrue, ...testPred])];
    const idx = Object.fromEntries(labels.map((l, i) => [String(l), i]));
    const m = labels.map(() => labels.map(() => 0));
    for (let i = 0; i < Math.min(testTrue.length, testPred.length); i++) {
      const ti = idx[String(testTrue[i])];
      const pi = idx[String(testPred[i])];
      if (ti != null && pi != null) m[ti][pi]++;
    }
    return { data: m, labels: labels.map(String) };
  }, [testTrue, testPred, classes.join(',')]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
      <Surface>
        <CardHeader title="混淆矩陣"
                    subtitle={`${model?.algo || ''} · ${cm ? `${testTrue.length} 筆 test` : '無 test 標籤'}`} />
        <div style={{ padding: 24, display: 'flex', justifyContent: 'center' }}>
          {cm
            ? <ConfusionMatrix data={cm.data} labels={cm.labels} />
            : <span className="t-label" style={{ color: 'var(--fg-muted)' }}>缺 testTrue / testPred,無法畫混淆矩陣</span>}
        </div>
      </Surface>
      <Surface>
        <CardHeader title="ROC 曲線" subtitle={model?.auc != null ? `AUC = ${model.auc.toFixed(3)}` : 'AUC 未提供'} />
        <div style={{ padding: 16, height: 280 }}>
          {model?.auc != null
            ? <ROCCurve auc={model.auc} />
            : <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <span className="t-label" style={{ color: 'var(--fg-muted)' }}>無 AUC 資料</span>
              </div>}
        </div>
      </Surface>
    </div>
  );
}

function RegressionDiagnostics({ model }) {
  // 從 model.testTrue + testPred 拼真實 scatter data
  const data = React.useMemo(() => {
    const t = model?.testTrue || [];
    const p = model?.testPred || [];
    if (t.length === 0 || p.length === 0) return MOCK.predScatter;  // fallback to mock
    return t.map((actual, i) => ({ actual: +actual, pred: +p[i], residual: +p[i] - +actual }));
  }, [model]);
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
      <Surface>
        <CardHeader title="預測 vs 實際" subtitle={`${data.length} 個樣本 · 對角線為理想預測`} />
        <div style={{ padding: 16, height: 320 }}>
          <PredScatterPlot data={data} />
        </div>
      </Surface>
      <Surface>
        <CardHeader title="殘差分佈" subtitle="預測 − 實際 · 應呈現以 0 為中心的常態分佈" />
        <div style={{ padding: 16, height: 320 }}>
          <ResidualHistogram data={data} />
        </div>
      </Surface>
    </div>
  );
}

function PredScatterPlot({ data }) {
  const min = Math.min(...data.flatMap(p => [p.actual, p.pred]));
  const max = Math.max(...data.flatMap(p => [p.actual, p.pred]));
  const xs = v => ((v - min) / (max - min)) * 88 + 6;
  const ys = v => 92 - ((v - min) / (max - min)) * 80;
  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ width: '100%', height: '100%' }}>
      <g stroke="var(--bd-subtle)" strokeWidth="0.3">
        <line x1="6" y1="12" x2="94" y2="12" />
        <line x1="6" y1="50" x2="94" y2="50" />
        <line x1="6" y1="92" x2="94" y2="92" />
      </g>
      <line x1={xs(min)} y1={ys(min)} x2={xs(max)} y2={ys(max)} stroke="var(--fg-faint)" strokeWidth="0.4" strokeDasharray="2 2" />
      {data.map((p, i) => (
        <circle key={i} cx={xs(p.actual)} cy={ys(p.pred)} r="0.9" fill="var(--primary)" opacity="0.55" />
      ))}
      <text x="6"  y="98" fontSize="2.4" fill="var(--fg-muted)" fontFamily="JetBrains Mono">實際值 →</text>
      <text x="6"  y="6"  fontSize="2.4" fill="var(--fg-muted)" fontFamily="JetBrains Mono">↑ 預測值</text>
    </svg>
  );
}

function ResidualHistogram({ data }) {
  const residuals = data.map(d => d.residual);
  const min = Math.min(...residuals), max = Math.max(...residuals);
  const binCount = 24;
  const binW = (max - min) / binCount;
  const bins = new Array(binCount).fill(0);
  residuals.forEach(r => {
    const idx = Math.min(binCount - 1, Math.max(0, Math.floor((r - min) / binW)));
    bins[idx] += 1;
  });
  const maxBin = Math.max(...bins);
  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ width: '100%', height: '100%' }}>
      <g stroke="var(--bd-subtle)" strokeWidth="0.3">
        <line x1="6" y1="12" x2="94" y2="12" />
        <line x1="6" y1="50" x2="94" y2="50" />
        <line x1="6" y1="92" x2="94" y2="92" />
      </g>
      <line x1={50} y1="12" x2={50} y2="92" stroke="var(--fg-faint)" strokeWidth="0.4" strokeDasharray="2 2" />
      {bins.map((c, i) => {
        const x = 6 + (i / binCount) * 88;
        const w = 88 / binCount - 0.5;
        const h = (c / maxBin) * 80;
        const center = min + (i + 0.5) * binW;
        return (
          <rect key={i} x={x} y={92 - h} width={w} height={h}
                fill={Math.abs(center) < binW ? 'var(--primary)' : 'color-mix(in srgb, var(--primary) 55%, transparent)'}
                opacity="0.85" />
        );
      })}
      <text x="6"  y="98" fontSize="2.4" fill="var(--fg-muted)" fontFamily="JetBrains Mono">{min.toFixed(0)}</text>
      <text x="50" y="98" fontSize="2.4" fill="var(--fg-muted)" fontFamily="JetBrains Mono" textAnchor="middle">0</text>
      <text x="94" y="98" fontSize="2.4" fill="var(--fg-muted)" fontFamily="JetBrains Mono" textAnchor="end">{max.toFixed(0)}</text>
      <text x="6"  y="6"  fontSize="2.4" fill="var(--fg-muted)" fontFamily="JetBrains Mono">↑ 筆數</text>
    </svg>
  );
}

function ConfusionMatrix({ data, labels }) {
  const total = data.flat().reduce((a, b) => a + b, 0);
  const max = Math.max(...data.flat());
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '60px repeat(2, 110px)', gridTemplateRows: '40px repeat(2, 80px)', gap: 6 }}>
      <div />
      {labels.map(l => <div key={'p' + l} className="t-label" style={{ textAlign: 'center', alignSelf: 'end' }}>預測 {l}</div>)}
      {data.map((row, i) => (
        <React.Fragment key={i}>
          <div className="t-label" style={{ alignSelf: 'center', textAlign: 'right' }}>實際 {labels[i]}</div>
          {row.map((v, j) => {
            const correct = i === j;
            const ratio = v / max;
            return (
              <div key={j} style={{
                background: correct
                  ? `color-mix(in srgb, var(--good) ${ratio * 80 + 10}%, var(--bg-sunken))`
                  : `color-mix(in srgb, var(--bad) ${ratio * 80 + 10}%, var(--bg-sunken))`,
                borderRadius: 7,
                border: '1px solid var(--bd-subtle)',
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
              }}>
                <div className="mono fg-1" style={{ fontSize: 24, fontWeight: 600 }}>{v}</div>
                <div className="t-label mono">{(v / total * 100).toFixed(1)}%</div>
              </div>
            );
          })}
        </React.Fragment>
      ))}
    </div>
  );
}

function ROCCurve({ auc }) {
  const points = [];
  for (let i = 0; i <= 50; i++) {
    const x = i / 50;
    const y = Math.pow(x, 1 - auc * 0.95);
    points.push([x * 100, 100 - y * 100]);
  }
  const path = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' ');
  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ width: '100%', height: '100%' }}>
      <g stroke="var(--bd-subtle)" strokeWidth="0.3">
        <line x1="0" y1="25" x2="100" y2="25" />
        <line x1="0" y1="50" x2="100" y2="50" />
        <line x1="0" y1="75" x2="100" y2="75" />
      </g>
      <line x1="0" y1="100" x2="100" y2="0" stroke="var(--fg-faint)" strokeWidth="0.4" strokeDasharray="2 2" />
      <path d={`${path} L100,100 L0,100 Z`} fill="var(--primary)" opacity="0.12" />
      <path d={path} fill="none" stroke="var(--primary)" strokeWidth="0.8" vectorEffect="non-scaling-stroke" />
      <text x="2" y="98" fontSize="2.6" fill="var(--fg-muted)">FPR →</text>
      <text x="2" y="6" fontSize="2.6" fill="var(--fg-muted)">↑ TPR</text>
      <text x="60" y="40" fontSize="3" fill="var(--primary)" fontFamily="JetBrains Mono">AUC = {auc.toFixed(3)}</text>
    </svg>
  );
}

window.PageInsights = PageInsights;
