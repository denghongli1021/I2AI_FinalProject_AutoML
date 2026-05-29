// ===== ECHARTS CONFIGURATIONS =====

const CHART_THEME = {
  bg: 'transparent',
  textColor: '#94a3b8',
  axisLineColor: '#1e293b',
  splitLineColor: '#1e293b',
  primary: '#3b82f6',
  accent: '#06b6d4',
  success: '#10b981',
  warning: '#f59e0b',
  danger: '#ef4444',
  purple: '#8b5cf6',
};

// Store chart instances for resize
const chartInstances = {};

function initChart(domId) {
  const el = document.getElementById(domId);
  if (!el) return null;
  if (chartInstances[domId]) {
    chartInstances[domId].dispose();
  }
  const chart = echarts.init(el, null, { renderer: 'canvas' });
  chartInstances[domId] = chart;
  return chart;
}

// ===== DASHBOARD CHARTS =====
function renderPerformanceTrend() {
  const chart = initChart('chart-performance-trend');
  if (!chart) return;
  const d = MOCK.performanceTrend;
  chart.setOption({
    tooltip: { trigger: 'axis', backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0', fontSize: 12 } },
    legend: { data: ['XGBoost', 'LightGBM', 'Ensemble'], top: 5, right: 10, textStyle: { color: CHART_THEME.textColor, fontSize: 11 } },
    grid: { left: 50, right: 20, top: 40, bottom: 30 },
    xAxis: { type: 'category', data: d.dates, axisLine: { lineStyle: { color: CHART_THEME.axisLineColor } }, axisLabel: { color: CHART_THEME.textColor, fontSize: 10 } },
    yAxis: { type: 'value', min: 0.90, max: 0.96, axisLine: { show: false }, axisLabel: { color: CHART_THEME.textColor, fontSize: 10, formatter: v => v.toFixed(2) }, splitLine: { lineStyle: { color: CHART_THEME.splitLineColor } } },
    series: [
      { name: 'XGBoost', type: 'line', data: d.xgboost, smooth: true, symbol: 'circle', symbolSize: 4, lineStyle: { width: 2, color: CHART_THEME.primary }, itemStyle: { color: CHART_THEME.primary }, areaStyle: { color: new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:'rgba(59,130,246,0.15)'},{offset:1,color:'rgba(59,130,246,0)'}]) } },
      { name: 'LightGBM', type: 'line', data: d.lightgbm, smooth: true, symbol: 'circle', symbolSize: 4, lineStyle: { width: 2, color: CHART_THEME.accent }, itemStyle: { color: CHART_THEME.accent }, areaStyle: { color: new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:'rgba(6,182,212,0.15)'},{offset:1,color:'rgba(6,182,212,0)'}]) } },
      { name: 'Ensemble', type: 'line', data: d.ensemble, smooth: true, symbol: 'diamond', symbolSize: 5, lineStyle: { width: 2.5, color: CHART_THEME.purple }, itemStyle: { color: CHART_THEME.purple }, areaStyle: { color: new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:'rgba(139,92,246,0.12)'},{offset:1,color:'rgba(139,92,246,0)'}]) } },
    ]
  });
}

function renderTaskDistribution() {
  const chart = initChart('chart-task-distribution');
  if (!chart) return;
  chart.setOption({
    tooltip: { trigger: 'item', backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0' } },
    series: [{
      type: 'pie', radius: ['50%', '75%'], center: ['50%', '50%'],
      avoidLabelOverlap: true,
      itemStyle: { borderRadius: 6, borderColor: '#0f172a', borderWidth: 3 },
      label: { show: true, color: CHART_THEME.textColor, fontSize: 11, formatter: '{b}\n{d}%' },
      labelLine: { lineStyle: { color: '#334155' } },
      data: [
        { value: 28, name: '二元分類', itemStyle: { color: CHART_THEME.primary } },
        { value: 15, name: '回歸', itemStyle: { color: CHART_THEME.accent } },
        { value: 8, name: '多元分類', itemStyle: { color: CHART_THEME.purple } },
        { value: 5, name: '時間序列', itemStyle: { color: CHART_THEME.warning } },
      ]
    }]
  });
}

// ===== DATASET CHARTS =====
function renderHealthScore() {
  const chart = initChart('chart-health-score');
  if (!chart) return;
  chart.setOption({
    radar: {
      indicator: [
        { name: '完整性', max: 100 },
        { name: '一致性', max: 100 },
        { name: '異常值', max: 100 },
        { name: '平衡度', max: 100 },
        { name: '特徵品質', max: 100 },
      ],
      shape: 'polygon',
      splitNumber: 4,
      axisName: { color: CHART_THEME.textColor, fontSize: 11 },
      splitLine: { lineStyle: { color: CHART_THEME.splitLineColor } },
      splitArea: { areaStyle: { color: ['rgba(59,130,246,0.02)', 'rgba(59,130,246,0.04)'] } },
      axisLine: { lineStyle: { color: CHART_THEME.splitLineColor } },
    },
    series: [{
      type: 'radar',
      symbol: 'circle', symbolSize: 6,
      data: [{
        value: [92, 88, 85, 72, 90],
        name: '數據健康度',
        areaStyle: { color: 'rgba(59,130,246,0.15)' },
        lineStyle: { color: CHART_THEME.primary, width: 2 },
        itemStyle: { color: CHART_THEME.primary },
      }]
    }]
  });
}

// ===== EXPERIMENT CHARTS =====
function renderOptimizationHistory() {
  const chart = initChart('chart-optimization-history');
  if (!chart) return;
  const d = MOCK.optimizationHistory;
  chart.setOption({
    tooltip: { trigger: 'axis', backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0', fontSize: 11 } },
    legend: { data: ['Trial F1', '最佳 F1'], top: 0, right: 0, textStyle: { color: CHART_THEME.textColor, fontSize: 10 } },
    grid: { left: 45, right: 10, top: 30, bottom: 25 },
    xAxis: { type: 'value', name: 'Trial', nameTextStyle: { color: CHART_THEME.textColor, fontSize: 10 }, axisLine: { lineStyle: { color: CHART_THEME.axisLineColor } }, axisLabel: { color: CHART_THEME.textColor, fontSize: 9 } },
    yAxis: { type: 'value', min: 0.84, max: 0.96, axisLine: { show: false }, axisLabel: { color: CHART_THEME.textColor, fontSize: 9 }, splitLine: { lineStyle: { color: CHART_THEME.splitLineColor } } },
    series: [
      { name: 'Trial F1', type: 'scatter', data: d.map(p => [p.trial, p.f1]), symbolSize: 4, itemStyle: { color: 'rgba(59,130,246,0.4)' } },
      { name: '最佳 F1', type: 'line', data: d.map(p => [p.trial, p.best]), smooth: true, symbol: 'none', lineStyle: { width: 2.5, color: CHART_THEME.success }, areaStyle: { color: new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:'rgba(16,185,129,0.1)'},{offset:1,color:'rgba(16,185,129,0)'}]) } },
    ]
  });
}

// ===== LEADERBOARD CHARTS =====
function renderCompareRadar(models) {
  const chart = initChart('chart-compare-radar');
  if (!chart) return;
  const colors = [CHART_THEME.primary, CHART_THEME.accent, CHART_THEME.purple];
  chart.setOption({
    legend: { data: models.map(m => m.name), bottom: 0, textStyle: { color: CHART_THEME.textColor, fontSize: 10 } },
    radar: {
      indicator: [
        { name: 'F1-Score', max: 1 },
        { name: 'AUC-ROC', max: 1 },
        { name: '準確率', max: 1 },
        { name: '速度', max: 1 },
        { name: '延遲', max: 1 },
      ],
      shape: 'polygon',
      splitNumber: 4,
      axisName: { color: CHART_THEME.textColor, fontSize: 10 },
      splitLine: { lineStyle: { color: CHART_THEME.splitLineColor } },
      splitArea: { areaStyle: { color: ['rgba(59,130,246,0.01)', 'rgba(59,130,246,0.03)'] } },
      axisLine: { lineStyle: { color: CHART_THEME.splitLineColor } },
    },
    series: [{
      type: 'radar',
      data: models.map((m, i) => ({
        value: [m.f1, m.auc, m.accuracy, 1 - m.timeVal / 1000, 1 - m.latencyVal / 50],
        name: m.name,
        lineStyle: { color: colors[i], width: 2 },
        itemStyle: { color: colors[i] },
        areaStyle: { color: colors[i].replace(')', ',0.1)').replace('rgb', 'rgba') },
      }))
    }]
  });
}

function renderCompareConfusion(models) {
  const chart = initChart('chart-compare-confusion');
  if (!chart) return;
  const cm = models[0];
  const tp = Math.round(cm.accuracy * 1000 * 0.7);
  const tn = Math.round(cm.accuracy * 1000 * 0.3);
  const fp = Math.round((1 - cm.accuracy) * 1000 * 0.4);
  const fn = Math.round((1 - cm.accuracy) * 1000 * 0.6);
  chart.setOption({
    title: { text: models[0].name, left: 'center', top: 5, textStyle: { color: '#e2e8f0', fontSize: 13 } },
    tooltip: { backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0' } },
    xAxis: { type: 'category', data: ['Pred: 0', 'Pred: 1'], axisLine: { lineStyle: { color: CHART_THEME.axisLineColor } }, axisLabel: { color: CHART_THEME.textColor }, splitLine: { show: false } },
    yAxis: { type: 'category', data: ['Actual: 1', 'Actual: 0'], axisLine: { lineStyle: { color: CHART_THEME.axisLineColor } }, axisLabel: { color: CHART_THEME.textColor }, splitLine: { show: false } },
    grid: { left: 70, right: 20, top: 35, bottom: 40 },
    visualMap: { show: false, min: 0, max: Math.max(tp, tn, fp, fn), inRange: { color: ['#1e293b', '#1e40af', '#3b82f6'] } },
    series: [{
      type: 'heatmap',
      data: [[0, 0, tn], [1, 0, fp], [0, 1, fn], [1, 1, tp]],
      label: { show: true, color: '#fff', fontSize: 16, fontWeight: 'bold' },
      itemStyle: { borderColor: '#0f172a', borderWidth: 3, borderRadius: 4 },
    }]
  });
}

function renderCompareROC(models) {
  const chart = initChart('chart-compare-roc');
  if (!chart) return;
  const colors = [CHART_THEME.primary, CHART_THEME.accent, CHART_THEME.purple];
  const series = models.map((m, i) => {
    const points = [];
    for (let x = 0; x <= 1; x += 0.02) {
      const y = Math.min(1, Math.pow(x, 1 / (m.auc * 3)));
      points.push([parseFloat(x.toFixed(3)), parseFloat(y.toFixed(3))]);
    }
    return {
      name: `${m.name} (AUC=${m.auc})`,
      type: 'line', data: points, smooth: true, symbol: 'none',
      lineStyle: { width: 2, color: colors[i] },
      areaStyle: { color: new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:colors[i]+'26'},{offset:1,color:colors[i]+'00'}]) },
    };
  });
  series.push({
    name: 'Random', type: 'line', data: [[0,0],[1,1]], symbol: 'none',
    lineStyle: { width: 1, color: '#475569', type: 'dashed' },
  });
  chart.setOption({
    tooltip: { trigger: 'axis', backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0', fontSize: 11 } },
    legend: { bottom: 0, textStyle: { color: CHART_THEME.textColor, fontSize: 10 } },
    grid: { left: 50, right: 20, top: 15, bottom: 40 },
    xAxis: { type: 'value', name: 'FPR', nameTextStyle: { color: CHART_THEME.textColor }, axisLine: { lineStyle: { color: CHART_THEME.axisLineColor } }, axisLabel: { color: CHART_THEME.textColor, fontSize: 10 }, splitLine: { lineStyle: { color: CHART_THEME.splitLineColor } } },
    yAxis: { type: 'value', name: 'TPR', nameTextStyle: { color: CHART_THEME.textColor }, axisLine: { show: false }, axisLabel: { color: CHART_THEME.textColor, fontSize: 10 }, splitLine: { lineStyle: { color: CHART_THEME.splitLineColor } } },
    series,
  });
}

// ===== INSIGHTS CHARTS =====
function renderFeatureImportance() {
  const chart = initChart('chart-feature-importance');
  if (!chart) return;
  const d = MOCK.featureImportance.slice().reverse();
  chart.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0' } },
    grid: { left: 130, right: 30, top: 10, bottom: 20 },
    xAxis: { type: 'value', axisLine: { show: false }, axisLabel: { color: CHART_THEME.textColor, fontSize: 10 }, splitLine: { lineStyle: { color: CHART_THEME.splitLineColor } } },
    yAxis: { type: 'category', data: d.map(f => f.name), axisLine: { show: false }, axisLabel: { color: '#e2e8f0', fontSize: 11 }, axisTick: { show: false } },
    series: [{
      type: 'bar', data: d.map(f => ({
        value: f.importance,
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0,0,1,0,[
            {offset:0, color: CHART_THEME.primary},
            {offset:1, color: CHART_THEME.accent}
          ]),
          borderRadius: [0, 4, 4, 0],
        }
      })),
      barWidth: '55%',
      label: { show: true, position: 'right', color: CHART_THEME.textColor, fontSize: 10, formatter: p => p.value.toFixed(3) },
    }]
  });
}

function renderShapWaterfall(sampleIdx) {
  const chart = initChart('chart-shap-waterfall');
  if (!chart) return;
  const sample = MOCK.shapSamples[sampleIdx || 0];
  const features = sample.features;
  const names = features.map(f => f.name);
  const values = features.map(f => f.value);

  chart.setOption({
    tooltip: { backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0' }, formatter: p => `${p.name}<br/>SHAP: ${p.data >= 0 ? '+' : ''}${p.data.toFixed(3)}` },
    grid: { left: 140, right: 40, top: 20, bottom: 30 },
    xAxis: {
      type: 'value',
      axisLine: { lineStyle: { color: CHART_THEME.axisLineColor } },
      axisLabel: { color: CHART_THEME.textColor, fontSize: 10, formatter: v => v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2) },
      splitLine: { lineStyle: { color: CHART_THEME.splitLineColor } },
    },
    yAxis: {
      type: 'category', data: names.slice().reverse(),
      axisLine: { show: false }, axisTick: { show: false },
      axisLabel: { color: '#e2e8f0', fontSize: 10 },
    },
    series: [{
      type: 'bar',
      data: values.slice().reverse().map(v => ({
        value: v,
        itemStyle: {
          color: v >= 0 ? CHART_THEME.danger : CHART_THEME.success,
          borderRadius: v >= 0 ? [0,4,4,0] : [4,0,0,4],
        }
      })),
      barWidth: '60%',
      label: { show: true, position: 'right', color: CHART_THEME.textColor, fontSize: 9, formatter: p => p.data >= 0 ? `+${p.data.toFixed(3)}` : p.data.toFixed(3) },
    }],
    graphic: [
      { type: 'text', left: 'center', bottom: 5, style: { text: `基礎值: ${sample.baseValue} → 預測值: ${sample.prediction.toFixed(2)}`, fill: '#64748b', fontSize: 10 } }
    ],
  });
}

function renderShapBeeswarm() {
  const chart = initChart('chart-shap-beeswarm');
  if (!chart) return;
  const d = MOCK.shapBeeswarm;
  chart.setOption({
    tooltip: {
      backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0', fontSize: 11 },
      formatter: p => `${d.features[p.data[1]]}<br/>SHAP: ${p.data[0].toFixed(3)}<br/>特徵值: ${p.data[2].toFixed(2)}`
    },
    grid: { left: 130, right: 40, top: 15, bottom: 30 },
    xAxis: {
      type: 'value', name: 'SHAP Value',
      nameTextStyle: { color: CHART_THEME.textColor, fontSize: 10 },
      axisLine: { lineStyle: { color: CHART_THEME.axisLineColor } },
      axisLabel: { color: CHART_THEME.textColor, fontSize: 10 },
      splitLine: { lineStyle: { color: CHART_THEME.splitLineColor } },
    },
    yAxis: {
      type: 'category', data: d.features.slice().reverse(),
      axisLine: { show: false }, axisTick: { show: false },
      axisLabel: { color: '#e2e8f0', fontSize: 10 },
    },
    visualMap: {
      show: true, dimension: 2, min: 0, max: 1,
      inRange: { color: [CHART_THEME.primary, CHART_THEME.danger] },
      text: ['高', '低'], textStyle: { color: CHART_THEME.textColor, fontSize: 10 },
      right: 0, top: 'center', orient: 'vertical', itemWidth: 10, itemHeight: 80,
    },
    series: [{
      type: 'scatter',
      data: d.data.map(p => [p[0], d.features.length - 1 - p[1], p[2]]),
      symbolSize: 5,
      itemStyle: { opacity: 0.7 },
    }]
  });
}

function renderWhatIfGauge(churnProb) {
  const chart = initChart('chart-whatif-gauge');
  if (!chart) return;
  let color;
  if (churnProb < 0.3) color = CHART_THEME.success;
  else if (churnProb < 0.6) color = CHART_THEME.warning;
  else color = CHART_THEME.danger;

  chart.setOption({
    series: [{
      type: 'gauge',
      startAngle: 200, endAngle: -20,
      min: 0, max: 100,
      splitNumber: 10,
      radius: '90%',
      center: ['50%', '60%'],
      axisLine: {
        lineStyle: {
          width: 18,
          color: [[0.3, CHART_THEME.success], [0.6, CHART_THEME.warning], [1, CHART_THEME.danger]]
        }
      },
      pointer: { length: '60%', width: 5, itemStyle: { color } },
      axisTick: { distance: -18, length: 6, lineStyle: { color: '#0f172a', width: 1.5 } },
      splitLine: { distance: -18, length: 12, lineStyle: { color: '#0f172a', width: 2 } },
      axisLabel: { distance: 8, color: CHART_THEME.textColor, fontSize: 9, formatter: v => v + '%' },
      detail: { valueAnimation: true, formatter: v => v.toFixed(1) + '%', color, fontSize: 28, fontWeight: 'bold', offsetCenter: [0, '10%'] },
      data: [{ value: parseFloat((churnProb * 100).toFixed(1)) }],
      title: { show: false },
    }]
  });
}

// ===== DEPLOYMENT CHARTS =====
function renderApiUsage() {
  const chart = initChart('chart-api-usage');
  if (!chart) return;
  const d = MOCK.apiUsage;
  chart.setOption({
    tooltip: { trigger: 'axis', backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0' }, formatter: p => `${p[0].axisValue}<br/>呼叫次數: ${p[0].data.toLocaleString()}` },
    grid: { left: 60, right: 20, top: 15, bottom: 30 },
    xAxis: { type: 'category', data: d.hours, axisLine: { lineStyle: { color: CHART_THEME.axisLineColor } }, axisLabel: { color: CHART_THEME.textColor, fontSize: 10 } },
    yAxis: { type: 'value', axisLine: { show: false }, axisLabel: { color: CHART_THEME.textColor, fontSize: 10 }, splitLine: { lineStyle: { color: CHART_THEME.splitLineColor } } },
    series: [{
      type: 'bar', data: d.calls,
      itemStyle: {
        color: new echarts.graphic.LinearGradient(0,0,0,1,[
          { offset: 0, color: CHART_THEME.primary },
          { offset: 1, color: 'rgba(59,130,246,0.2)' }
        ]),
        borderRadius: [3, 3, 0, 0],
      },
      barWidth: '60%',
    }]
  });
}

// ===== REAL DATA CHARTS =====

function renderRealHealthScore(scores) {
  const chart = initChart('chart-health-score');
  if (!chart) return;
  chart.setOption({
    radar: {
      indicator: [
        { name: '完整性', max: 100 },
        { name: '一致性', max: 100 },
        { name: '異常值', max: 100 },
        { name: '平衡度', max: 100 },
        { name: '特徵品質', max: 100 },
      ],
      shape: 'polygon', splitNumber: 4,
      axisName: { color: CHART_THEME.textColor, fontSize: 11 },
      splitLine: { lineStyle: { color: CHART_THEME.splitLineColor } },
      splitArea: { areaStyle: { color: ['rgba(59,130,246,0.02)', 'rgba(59,130,246,0.04)'] } },
      axisLine: { lineStyle: { color: CHART_THEME.splitLineColor } },
    },
    series: [{
      type: 'radar', symbol: 'circle', symbolSize: 6,
      data: [{
        value: [scores.completeness, scores.consistency, scores.outlierScore, scores.balanceScore, scores.featureQuality],
        name: '數據健康度',
        areaStyle: { color: 'rgba(59,130,246,0.15)' },
        lineStyle: { color: CHART_THEME.primary, width: 2 },
        itemStyle: { color: CHART_THEME.primary },
      }]
    }]
  });
}

function renderCorrelationHeatmap(corrData, topN) {
  const chart = initChart('chart-correlation');
  if (!chart || !corrData) return;
  let { names, matrix } = corrData;

  // If topN is set and there are more features, pick top N by highest avg |r|
  if (topN && topN > 0 && names.length > topN) {
    // Compute average |correlation| per feature (excluding self)
    const avgCorr = names.map((name, i) => {
      const vals = matrix.filter(m => m[0] === i || m[1] === i).filter(m => m[0] !== m[1]).map(m => Math.abs(m[2]));
      const avg = vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
      return { idx: i, name, avg };
    });
    avgCorr.sort((a, b) => b.avg - a.avg);
    const topIndices = avgCorr.slice(0, topN).map(x => x.idx).sort((a, b) => a - b);
    const idxMap = {};
    topIndices.forEach((origIdx, newIdx) => { idxMap[origIdx] = newIdx; });
    names = topIndices.map(i => names[i]);
    matrix = matrix
      .filter(m => topIndices.includes(m[0]) && topIndices.includes(m[1]))
      .map(m => [idxMap[m[0]], idxMap[m[1]], m[2]]);
  }

  const n = names.length;
  const maxLen = Math.max(...names.map(nm => nm.length));
  const leftMargin = Math.min(140, maxLen * 7 + 10);

  // Dynamic height: scale with feature count
  const cellSize = n <= 10 ? 36 : n <= 20 ? 28 : 22;
  const chartHeight = Math.max(300, n * cellSize + 100);
  const el = document.getElementById('chart-correlation');
  if (el) el.style.height = chartHeight + 'px';
  chart.resize();

  chart.setOption({
    tooltip: {
      backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0', fontSize: 11 },
      formatter: p => `${names[p.data[0]]} vs ${names[p.data[1]]}<br/>r = ${p.data[2]}`
    },
    grid: { left: leftMargin, right: 30, top: 10, bottom: 80 },
    xAxis: { type: 'category', data: names, axisLabel: { color: CHART_THEME.textColor, fontSize: n > 20 ? 8 : 9, rotate: 40, interval: 0 }, axisLine: { show: false }, axisTick: { show: false } },
    yAxis: { type: 'category', data: names, axisLabel: { color: CHART_THEME.textColor, fontSize: n > 20 ? 8 : 9, interval: 0 }, axisLine: { show: false }, axisTick: { show: false } },
    visualMap: { min: -1, max: 1, calculable: false, orient: 'horizontal', left: 'center', bottom: 2, itemWidth: 10, itemHeight: 60,
      inRange: { color: ['#3b82f6', '#0f172a', '#ef4444'] },
      textStyle: { color: CHART_THEME.textColor, fontSize: 9 },
    },
    series: [{
      type: 'heatmap', data: matrix,
      label: { show: n <= 12, color: '#e2e8f0', fontSize: n > 10 ? 8 : 9, formatter: p => p.data[2].toFixed(2) },
      itemStyle: { borderColor: '#0f172a', borderWidth: n > 20 ? 1 : 1.5, borderRadius: 2 },
    }]
  });
}

function renderColumnDetail(colInfo) {
  const chart = initChart('chart-col-detail');
  if (!chart) return;

  if (colInfo.type === 'numeric' && colInfo.distribution.length > 0) {
    // Histogram
    chart.setOption({
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0', fontSize: 11 } },
      grid: { left: 50, right: 15, top: 15, bottom: 30 },
      xAxis: { type: 'category', data: colInfo.distribution.map(b => b.label || ''), axisLine: { lineStyle: { color: CHART_THEME.axisLineColor } }, axisLabel: { color: CHART_THEME.textColor, fontSize: 8, rotate: 30, interval: 0 } },
      yAxis: { type: 'value', axisLine: { show: false }, axisLabel: { color: CHART_THEME.textColor, fontSize: 9 }, splitLine: { lineStyle: { color: CHART_THEME.splitLineColor } } },
      series: [{
        type: 'bar', data: colInfo.distribution.map(b => b.count),
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0,0,0,1,[
            { offset: 0, color: CHART_THEME.accent },
            { offset: 1, color: 'rgba(6,182,212,0.15)' }
          ]),
          borderRadius: [3, 3, 0, 0],
        },
        barWidth: '80%',
      }]
    });
  } else if ((colInfo.type === 'categorical' || colInfo.type === 'boolean' || colInfo.type === 'text') && colInfo.topValues && colInfo.topValues.length > 0) {
    // Bar chart of top values
    const items = colInfo.topValues.slice(0, 10);
    const maxLabelLen = Math.max(...items.map(i => String(i.value).length));
    chart.setOption({
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0', fontSize: 11 } },
      grid: { left: Math.min(140, maxLabelLen * 7 + 10), right: 30, top: 10, bottom: 15 },
      xAxis: { type: 'value', axisLine: { show: false }, axisLabel: { color: CHART_THEME.textColor, fontSize: 9 }, splitLine: { lineStyle: { color: CHART_THEME.splitLineColor } } },
      yAxis: { type: 'category', data: items.map(i => i.value).reverse(), axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: '#e2e8f0', fontSize: 10, formatter: v => v.length > 18 ? v.slice(0, 16) + '...' : v } },
      series: [{
        type: 'bar', data: items.map(i => i.count).reverse(),
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0,0,1,0,[
            { offset: 0, color: CHART_THEME.primary },
            { offset: 1, color: CHART_THEME.accent }
          ]),
          borderRadius: [0, 4, 4, 0],
        },
        barWidth: '55%',
        label: { show: true, position: 'right', color: CHART_THEME.textColor, fontSize: 9, formatter: p => p.value },
      }]
    });
  } else if (colInfo.type === 'datetime' && colInfo.distribution && colInfo.distribution.length > 0) {
    // Timeline bar chart by month
    chart.setOption({
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0', fontSize: 11 },
        formatter: p => `${p[0].name}<br/>數量: ${p[0].value.toLocaleString()}` },
      grid: { left: 50, right: 15, top: 15, bottom: 35 },
      xAxis: { type: 'category', data: colInfo.distribution.map(b => b.label),
        axisLine: { lineStyle: { color: CHART_THEME.axisLineColor } },
        axisLabel: { color: CHART_THEME.textColor, fontSize: 8, rotate: 35, interval: 0 } },
      yAxis: { type: 'value', axisLine: { show: false },
        axisLabel: { color: CHART_THEME.textColor, fontSize: 9 },
        splitLine: { lineStyle: { color: CHART_THEME.splitLineColor } } },
      series: [{
        type: 'bar', data: colInfo.distribution.map(b => b.count),
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0,0,0,1,[
            { offset: 0, color: CHART_THEME.warning },
            { offset: 1, color: 'rgba(245,158,11,0.15)' }
          ]),
          borderRadius: [3, 3, 0, 0],
        },
        barWidth: '70%',
      }]
    });
  } else {
    chart.clear();
    chart.setOption({
      title: { text: '此欄位無法繪製分佈圖', left: 'center', top: 'center', textStyle: { color: '#475569', fontSize: 13 } }
    });
  }
}

// ===== BEFORE/AFTER PROCESSING CHARTS =====

function renderBeforeAfterHistogram(domId, distribution, color, highlightIndices) {
  const chart = initChart(domId);
  if (!chart) return;

  if (!distribution || distribution.length === 0) {
    chart.clear();
    chart.setOption({ title: { text: '無分佈數據', left: 'center', top: 'center', textStyle: { color: '#475569', fontSize: 12 } } });
    return;
  }

  chart.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: '#1e293b', borderColor: '#334155',
      textStyle: { color: '#e2e8f0', fontSize: 11 },
      formatter: p => `${p[0].name}<br/>數量: ${p[0].value}` },
    grid: { left: 45, right: 12, top: 10, bottom: 28 },
    xAxis: {
      type: 'category',
      data: distribution.map(b => b.label || ''),
      axisLine: { lineStyle: { color: CHART_THEME.axisLineColor } },
      axisLabel: { color: CHART_THEME.textColor, fontSize: 7, rotate: 30, interval: 0 },
    },
    yAxis: {
      type: 'value', axisLine: { show: false },
      axisLabel: { color: CHART_THEME.textColor, fontSize: 9 },
      splitLine: { lineStyle: { color: CHART_THEME.splitLineColor } },
    },
    series: [{
      type: 'bar',
      data: distribution.map(b => b.count),
      itemStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: color },
          { offset: 1, color: color + '30' },
        ]),
        borderRadius: [3, 3, 0, 0],
      },
      barWidth: '75%',
    }],
  });
}

function renderBeforeAfterCategorical(domId, topValues, color) {
  const chart = initChart(domId);
  if (!chart) return;

  if (!topValues || topValues.length === 0) {
    chart.clear();
    chart.setOption({ title: { text: '無分佈數據', left: 'center', top: 'center', textStyle: { color: '#475569', fontSize: 12 } } });
    return;
  }

  const items = topValues.slice(0, 8);
  const maxLabelLen = Math.max(...items.map(i => String(i.value).length));
  chart.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: '#1e293b', borderColor: '#334155', textStyle: { color: '#e2e8f0', fontSize: 11 } },
    grid: { left: Math.min(100, maxLabelLen * 7 + 10), right: 25, top: 8, bottom: 8 },
    xAxis: { type: 'value', axisLine: { show: false }, axisLabel: { color: CHART_THEME.textColor, fontSize: 9 }, splitLine: { lineStyle: { color: CHART_THEME.splitLineColor } } },
    yAxis: { type: 'category', data: items.map(i => i.value).reverse(), axisLine: { show: false }, axisTick: { show: false },
      axisLabel: { color: '#e2e8f0', fontSize: 9, formatter: v => v.length > 14 ? v.slice(0, 12) + '..' : v } },
    series: [{
      type: 'bar', data: items.map(i => i.count).reverse(),
      itemStyle: { color, borderRadius: [0, 4, 4, 0] },
      barWidth: '55%',
      label: { show: true, position: 'right', color: CHART_THEME.textColor, fontSize: 9, formatter: p => p.value },
    }],
  });
}

// ===== RESIZE HANDLER =====
window.addEventListener('resize', () => {
  Object.values(chartInstances).forEach(chart => {
    if (chart && !chart.isDisposed()) chart.resize();
  });
});
