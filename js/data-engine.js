// ===== DATA ENGINE: CSV Parsing + Real Data Analysis =====

const DataEngine = {
  // Current dataset state
  currentDataset: null,
  // All loaded datasets
  datasets: [],

  // ---- CSV PARSER ----
  parseCSV(text, delimiter) {
    if (!delimiter) delimiter = this.detectDelimiter(text);
    const rows = [];
    let current = '';
    let inQuotes = false;
    let row = [];

    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      const next = text[i + 1];

      if (inQuotes) {
        if (ch === '"' && next === '"') {
          current += '"';
          i++;
        } else if (ch === '"') {
          inQuotes = false;
        } else {
          current += ch;
        }
      } else {
        if (ch === '"') {
          inQuotes = true;
        } else if (ch === delimiter) {
          row.push(current.trim());
          current = '';
        } else if (ch === '\n' || (ch === '\r' && next === '\n')) {
          row.push(current.trim());
          current = '';
          if (row.length > 1 || row[0] !== '') rows.push(row);
          row = [];
          if (ch === '\r') i++;
        } else {
          current += ch;
        }
      }
    }
    // Last field
    row.push(current.trim());
    if (row.length > 1 || row[0] !== '') rows.push(row);

    return rows;
  },

  detectDelimiter(text) {
    const firstLine = text.split(/\r?\n/)[0];
    const counts = { ',': 0, '\t': 0, ';': 0, '|': 0 };
    for (const ch of firstLine) {
      if (ch in counts) counts[ch]++;
    }
    let best = ',', bestCount = 0;
    for (const [d, c] of Object.entries(counts)) {
      if (c > bestCount) { best = d; bestCount = c; }
    }
    return best;
  },

  // ---- LOAD & ANALYZE ----
  loadCSV(text, fileName) {
    const rows = this.parseCSV(text);
    if (rows.length < 2) throw new Error('CSV 至少需要包含標題列和一行資料');

    const headers = rows[0];
    const data = rows.slice(1).filter(r => r.length === headers.length);

    // Build column arrays
    const columns = {};
    headers.forEach((h, i) => {
      columns[h] = data.map(row => row[i]);
    });

    const analysis = this.analyzeColumns(headers, columns, data);

    const dataset = {
      id: Date.now(),
      fileName,
      headers,
      data,
      columns,
      rowCount: data.length,
      colCount: headers.length,
      analysis,
      loadedAt: new Date(),
    };

    // Check if same fileName already exists; replace it
    const existIdx = this.datasets.findIndex(d => d.fileName === fileName);
    if (existIdx >= 0) {
      this.datasets[existIdx] = dataset;
    } else {
      this.datasets.push(dataset);
    }

    this.currentDataset = dataset;
    return this.currentDataset;
  },

  // Switch to a dataset by id
  switchDataset(id) {
    const ds = this.datasets.find(d => d.id === id);
    if (ds) this.currentDataset = ds;
    return this.currentDataset;
  },

  // Remove a dataset by id
  removeDataset(id) {
    this.datasets = this.datasets.filter(d => d.id !== id);
    if (this.currentDataset && this.currentDataset.id === id) {
      this.currentDataset = this.datasets.length > 0 ? this.datasets[this.datasets.length - 1] : null;
    }
  },

  // ---- COLUMN ANALYSIS ----
  analyzeColumns(headers, columns, data) {
    const result = [];

    headers.forEach(name => {
      const raw = columns[name];
      const colInfo = {
        name,
        type: 'unknown',
        dtype: 'unknown',
        totalCount: raw.length,
        missingCount: 0,
        missingPct: 0,
        uniqueCount: 0,
        uniqueValues: [],
        outlierCount: 0,
        stats: null,
        distribution: [],
        topValues: [],
      };

      // Count missing
      const isMissingVal = v => {
        if (v === null || v === undefined || v === '') return true;
        const lower = String(v).trim().toLowerCase();
        return ['na','nan','null','n/a','none','missing','undefined','?','-','--','.'].includes(lower);
      };
      const missing = raw.filter(isMissingVal);
      colInfo.missingCount = missing.length;
      colInfo.missingPct = parseFloat(((missing.length / raw.length) * 100).toFixed(1));

      // Non-missing values
      const valid = raw.filter(v => !isMissingVal(v));

      // Unique values
      const uniqueSet = new Set(valid);
      colInfo.uniqueCount = uniqueSet.size;

      // Detect type
      const typeInfo = this.detectColumnType(valid, name);
      colInfo.type = typeInfo.type;
      colInfo.dtype = typeInfo.dtype;

      if (typeInfo.type === 'numeric') {
        const nums = valid.map(Number).filter(n => !isNaN(n));
        colInfo.stats = this.computeNumericStats(nums);
        colInfo.distribution = this.computeHistogram(nums, 15);
        colInfo.outlierCount = this.countOutliers(nums, colInfo.stats);
      } else if (typeInfo.type === 'categorical' || typeInfo.type === 'boolean') {
        colInfo.topValues = this.computeValueCounts(valid, 10);
        colInfo.distribution = colInfo.topValues.map(v => v.count);
      } else if (typeInfo.type === 'datetime') {
        const parsed = valid.map(v => new Date(v)).filter(d => !isNaN(d.getTime())).sort((a, b) => a - b);
        if (parsed.length > 0) {
          colInfo.stats = {
            min: parsed[0].toISOString().slice(0, 10),
            max: parsed[parsed.length - 1].toISOString().slice(0, 10),
            count: parsed.length,
          };
          // Build monthly distribution
          const monthCounts = {};
          parsed.forEach(d => {
            const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;
            monthCounts[key] = (monthCounts[key] || 0) + 1;
          });
          const sortedMonths = Object.keys(monthCounts).sort();
          colInfo.distribution = sortedMonths.map(m => ({ label: m, count: monthCounts[m] }));
        } else {
          colInfo.stats = { min: valid[0], max: valid[valid.length - 1] };
        }
      } else if (typeInfo.type === 'text') {
        // Text columns: show top values like categorical
        colInfo.topValues = this.computeValueCounts(valid, 10);
        colInfo.distribution = colInfo.topValues.map(v => v.count);
      }

      result.push(colInfo);
    });

    return result;
  },

  detectColumnType(values, name) {
    if (values.length === 0) return { type: 'unknown', dtype: 'unknown' };

    const sample = values.slice(0, Math.min(200, values.length));

    // Boolean check
    const boolSet = new Set(sample.map(v => v.toLowerCase()));
    const boolValues = new Set(['true','false','yes','no','0','1','t','f','y','n']);
    if (boolSet.size <= 3 && [...boolSet].every(v => boolValues.has(v))) {
      return { type: 'boolean', dtype: 'bool' };
    }

    // Numeric check
    const numericCount = sample.filter(v => !isNaN(Number(v)) && v.trim() !== '').length;
    if (numericCount / sample.length > 0.85) {
      const hasDecimal = sample.some(v => v.includes('.'));
      return { type: 'numeric', dtype: hasDecimal ? 'float64' : 'int64' };
    }

    // Date check
    const datePatterns = [
      /^\d{4}[-/]\d{1,2}[-/]\d{1,2}/,
      /^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}/,
      /^\d{4}\d{2}\d{2}$/,
    ];
    const dateCount = sample.filter(v => datePatterns.some(p => p.test(v)) || !isNaN(Date.parse(v))).length;
    if (dateCount / sample.length > 0.7) {
      return { type: 'datetime', dtype: 'datetime' };
    }

    // Categorical
    const uniqueRatio = new Set(sample).size / sample.length;
    if (uniqueRatio < 0.3 || new Set(values).size <= 20) {
      return { type: 'categorical', dtype: 'category' };
    }

    return { type: 'text', dtype: 'string' };
  },

  computeNumericStats(nums) {
    if (nums.length === 0) return null;
    const sorted = [...nums].sort((a, b) => a - b);
    const n = sorted.length;
    const sum = sorted.reduce((a, b) => a + b, 0);
    const mean = sum / n;
    const variance = sorted.reduce((acc, v) => acc + (v - mean) ** 2, 0) / n;
    const std = Math.sqrt(variance);

    const q1 = sorted[Math.floor(n * 0.25)];
    const median = sorted[Math.floor(n * 0.5)];
    const q3 = sorted[Math.floor(n * 0.75)];
    const iqr = q3 - q1;

    return {
      count: n,
      mean: parseFloat(mean.toFixed(4)),
      std: parseFloat(std.toFixed(4)),
      min: sorted[0],
      q1: parseFloat(q1.toFixed(4)),
      median: parseFloat(median.toFixed(4)),
      q3: parseFloat(q3.toFixed(4)),
      max: sorted[n - 1],
      iqr: parseFloat(iqr.toFixed(4)),
      skewness: parseFloat((sorted.reduce((acc, v) => acc + ((v - mean) / std) ** 3, 0) / n).toFixed(4)),
    };
  },

  computeHistogram(nums, bins) {
    if (nums.length === 0) return [];
    const min = Math.min(...nums);
    const max = Math.max(...nums);
    if (min === max) return [{ min, max, count: nums.length }];

    const binWidth = (max - min) / bins;
    const histogram = [];
    for (let i = 0; i < bins; i++) {
      const lo = min + i * binWidth;
      const hi = lo + binWidth;
      const count = nums.filter(v => i === bins - 1 ? (v >= lo && v <= hi) : (v >= lo && v < hi)).length;
      histogram.push({
        min: parseFloat(lo.toFixed(4)),
        max: parseFloat(hi.toFixed(4)),
        count,
        label: `${lo.toFixed(1)}-${hi.toFixed(1)}`
      });
    }
    return histogram;
  },

  countOutliers(nums, stats) {
    if (!stats) return 0;
    const lower = stats.q1 - 1.5 * stats.iqr;
    const upper = stats.q3 + 1.5 * stats.iqr;
    return nums.filter(v => v < lower || v > upper).length;
  },

  computeValueCounts(values, topN) {
    const counts = {};
    values.forEach(v => { counts[v] = (counts[v] || 0) + 1; });
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, topN)
      .map(([value, count]) => ({ value, count, pct: parseFloat(((count / values.length) * 100).toFixed(1)) }));
  },

  // ---- CORRELATION MATRIX (Pearson, numeric cols only) ----
  computeCorrelationMatrix() {
    if (!this.currentDataset) return null;
    const { headers, columns, analysis } = this.currentDataset;
    const numericCols = analysis.filter(c => c.type === 'numeric');
    if (numericCols.length < 2) return null;

    const names = numericCols.map(c => c.name);
    const numArrays = names.map(name => columns[name].map(Number).filter(n => !isNaN(n)));

    // Pairwise correlation
    const matrix = [];
    for (let i = 0; i < names.length; i++) {
      for (let j = 0; j < names.length; j++) {
        const r = this.pearsonCorrelation(numArrays[i], numArrays[j]);
        matrix.push([i, j, parseFloat(r.toFixed(3))]);
      }
    }

    return { names, matrix };
  },

  pearsonCorrelation(x, y) {
    const n = Math.min(x.length, y.length);
    if (n === 0) return 0;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
    for (let i = 0; i < n; i++) {
      sumX += x[i]; sumY += y[i];
      sumXY += x[i] * y[i];
      sumX2 += x[i] * x[i];
      sumY2 += y[i] * y[i];
    }
    const num = n * sumXY - sumX * sumY;
    const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    return den === 0 ? 0 : num / den;
  },

  // ---- DATA HEALTH SCORE ----
  computeHealthScore() {
    if (!this.currentDataset) return null;
    const { analysis, rowCount } = this.currentDataset;

    // Completeness: 100 - avg missing pct
    const avgMissing = analysis.reduce((sum, c) => sum + c.missingPct, 0) / analysis.length;
    const completeness = Math.max(0, Math.round(100 - avgMissing));

    // Consistency: based on type detection confidence
    const typedCols = analysis.filter(c => c.type !== 'unknown' && c.type !== 'text');
    const consistency = Math.round((typedCols.length / analysis.length) * 100);

    // Outliers: 100 - % of total outliers
    const totalOutliers = analysis.reduce((sum, c) => sum + c.outlierCount, 0);
    const outlierScore = Math.max(0, Math.round(100 - (totalOutliers / rowCount) * 100 * 5));

    // Balance: check if any categorical column is heavily skewed
    const catCols = analysis.filter(c => c.type === 'categorical' || c.type === 'boolean');
    let balanceScore = 85;
    catCols.forEach(c => {
      if (c.topValues.length > 0) {
        const topPct = c.topValues[0].pct;
        if (topPct > 90) balanceScore = Math.min(balanceScore, 40);
        else if (topPct > 80) balanceScore = Math.min(balanceScore, 60);
        else if (topPct > 70) balanceScore = Math.min(balanceScore, 75);
      }
    });

    // Feature quality: unique ratio check
    const goodFeatures = analysis.filter(c => {
      if (c.type === 'numeric') return c.stats && c.stats.std > 0;
      if (c.type === 'categorical') return c.uniqueCount > 1 && c.uniqueCount < c.totalCount * 0.5;
      return true;
    });
    const featureQuality = Math.round((goodFeatures.length / analysis.length) * 100);

    const overall = Math.round((completeness + consistency + outlierScore + balanceScore + featureQuality) / 5);

    return { completeness, consistency, outlierScore, balanceScore, featureQuality, overall };
  },

  // ---- AUTO-PROCESSING LOG (rich objects) ----
  generateProcessingLog() {
    if (!this.currentDataset) return [];
    const { analysis } = this.currentDataset;
    const logs = [];
    const now = new Date();
    const timeStr = now.toLocaleTimeString('zh-TW', { hour: '2-digit', minute: '2-digit' });

    analysis.forEach(col => {
      // Missing value handling
      if (col.missingPct > 0 && col.missingPct < 50) {
        const defaultMethod = col.type === 'numeric' ? 'median' : 'mode';
        const methodLabel = col.type === 'numeric' ? '中位數' : '眾數';
        logs.push({
          type: 'success', action: 'fill_missing', colName: col.name, time: timeStr,
          text: `已自動填補 <strong class="text-accent-400">${col.name}</strong> 欄位 ${col.missingPct}% 的缺失值（使用${methodLabel}填補）`,
          selectedMethod: defaultMethod,
        });
      } else if (col.missingPct >= 50) {
        logs.push({
          type: 'warning', action: 'drop_column', colName: col.name, time: timeStr,
          text: `<strong class="text-warning-400">${col.name}</strong> 欄位缺失率高達 ${col.missingPct}%，建議考慮移除此欄位`,
          selectedMethod: 'drop_column',
        });
      }

      // Outlier handling
      if (col.outlierCount > 0) {
        logs.push({
          type: col.outlierCount > 10 ? 'warning' : 'success',
          action: 'handle_outlier', colName: col.name, time: timeStr,
          text: `偵測到 <strong class="${col.outlierCount > 10 ? 'text-warning-400' : 'text-accent-400'}">${col.name}</strong> 欄位有 ${col.outlierCount} 筆異常值（IQR 法）`,
          selectedMethod: 'keep',
        });
      }

      // Encoding
      if (col.type === 'categorical' && col.uniqueCount <= 10) {
        logs.push({
          type: 'success', action: 'encode', colName: col.name, time: timeStr,
          text: `已自動將 <strong class="text-accent-400">${col.name}</strong> 欄位進行 One-Hot 編碼（${col.uniqueCount} 個類別）`,
          selectedMethod: 'onehot',
        });
      }

      // Datetime decomposition
      if (col.type === 'datetime') {
        logs.push({
          type: 'success', action: 'decompose_date', colName: col.name, time: timeStr,
          text: `已將 <strong class="text-accent-400">${col.name}</strong> 拆解為年、月、星期、季度特徵`,
          selectedMethod: 'decompose',
        });
      }
    });

    // Interaction features
    const numCols = analysis.filter(c => c.type === 'numeric' && c.stats && c.stats.std > 0);
    if (numCols.length >= 2) {
      const top2 = numCols.slice(0, 2);
      logs.push({
        type: 'success', action: 'interaction', colName: `${top2[0].name} × ${top2[1].name}`, time: timeStr,
        text: `已自動產生交互特徵: <strong class="text-accent-400">${top2[0].name} × ${top2[1].name}</strong>`,
        selectedMethod: 'multiply',
      });
    }

    return logs;
  },

  // ---- AVAILABLE METHODS per action type ----
  getMethodOptions(action, colType) {
    switch (action) {
      case 'fill_missing':
        if (colType === 'numeric') {
          return [
            { value: 'median', label: '中位數填補', desc: '使用中位數填補缺失值，對異常值穩健' },
            { value: 'mean', label: '平均數填補', desc: '使用平均數填補缺失值，可能受異常值影響' },
            { value: 'zero', label: '填補為 0', desc: '將所有缺失值設為 0' },
            { value: 'ffill', label: '前值填補 (Forward Fill)', desc: '使用前一個有效值填補' },
            { value: 'drop_rows', label: '刪除含缺失的列', desc: '移除所有包含此欄位缺失值的資料列' },
          ];
        }
        return [
          { value: 'mode', label: '眾數填補', desc: '使用出現次數最多的值填補' },
          { value: 'unknown', label: '填補為 "Unknown"', desc: '以 Unknown 標記缺失值' },
          { value: 'drop_rows', label: '刪除含缺失的列', desc: '移除所有包含此欄位缺失值的資料列' },
        ];
      case 'handle_outlier':
        return [
          { value: 'keep', label: '保留不處理', desc: '維持原始數據不做修改' },
          { value: 'clip', label: '截斷至 IQR 邊界', desc: '將超出 Q1-1.5*IQR ~ Q3+1.5*IQR 的值截斷至邊界' },
          { value: 'remove', label: '刪除異常值列', desc: '移除所有含異常值的資料列' },
          { value: 'median', label: '替換為中位數', desc: '將異常值替換為該欄位中位數' },
        ];
      case 'encode':
        return [
          { value: 'onehot', label: 'One-Hot 編碼', desc: '每個類別產生一個二元欄位' },
          { value: 'label', label: 'Label 編碼', desc: '將類別對應為整數 0, 1, 2...' },
          { value: 'none', label: '不編碼', desc: '保持原始類別值' },
        ];
      case 'decompose_date':
        return [
          { value: 'decompose', label: '拆解為年/月/星期/季度', desc: '將日期拆解為多個數值特徵' },
          { value: 'timestamp', label: '轉為時間戳 (Unix)', desc: '轉換為秒數數值' },
          { value: 'none', label: '不處理', desc: '保持原始日期值' },
        ];
      case 'drop_column':
        return [
          { value: 'drop_column', label: '移除此欄位', desc: '從數據集中移除整個欄位' },
          { value: 'keep_fill', label: '保留並用 0/Unknown 填補', desc: '保留欄位但填補所有缺失值' },
        ];
      case 'interaction':
        return [
          { value: 'multiply', label: '相乘 (A × B)', desc: '產生兩個欄位的乘積特徵' },
          { value: 'add', label: '相加 (A + B)', desc: '產生兩個欄位的加總特徵' },
          { value: 'ratio', label: '比值 (A / B)', desc: '產生兩個欄位的比值特徵' },
          { value: 'none', label: '不產生交互特徵', desc: '跳過此步驟' },
        ];
      default:
        return [];
    }
  },

  // ---- COMPUTE BEFORE/AFTER SNAPSHOT ----
  computeBeforeAfter(colName, action, method) {
    if (!this.currentDataset) return null;
    const { columns, analysis } = this.currentDataset;
    const col = analysis.find(c => c.name === colName);
    if (!col && action !== 'interaction') return null;

    const raw = columns[colName] || [];
    const isMissing = v => {
      if (v === null || v === undefined || v === '') return true;
      const lower = String(v).trim().toLowerCase();
      return ['na','nan','null','n/a','none','missing','undefined','?','-','--','.'].includes(lower);
    };

    const result = {
      before: { values: [...raw], stats: null, distribution: [] },
      after: { values: [...raw], stats: null, distribution: [] },
      affectedIndices: [],
    };

    if (action === 'fill_missing' && col) {
      const validNums = raw.filter(v => !isMissing(v)).map(Number).filter(n => !isNaN(n));
      const missingIndices = raw.map((v, i) => isMissing(v) ? i : -1).filter(i => i >= 0);
      result.affectedIndices = missingIndices;

      let fillValue;
      switch (method) {
        case 'median':
          const sorted = [...validNums].sort((a, b) => a - b);
          fillValue = sorted[Math.floor(sorted.length / 2)];
          break;
        case 'mean':
          fillValue = validNums.reduce((a, b) => a + b, 0) / validNums.length;
          fillValue = parseFloat(fillValue.toFixed(4));
          break;
        case 'zero':
          fillValue = 0;
          break;
        case 'mode':
          const counts = {};
          raw.filter(v => !isMissing(v)).forEach(v => { counts[v] = (counts[v] || 0) + 1; });
          fillValue = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] || 'Unknown';
          break;
        case 'unknown':
          fillValue = 'Unknown';
          break;
        case 'ffill':
          result.after.values = [...raw];
          let lastValid = null;
          for (let i = 0; i < result.after.values.length; i++) {
            if (isMissing(result.after.values[i])) {
              if (lastValid !== null) result.after.values[i] = lastValid;
            } else {
              lastValid = result.after.values[i];
            }
          }
          break;
        case 'drop_rows':
          result.after.values = raw.map((v, i) => isMissing(v) ? '__DROPPED__' : v);
          break;
      }

      if (method !== 'ffill' && method !== 'drop_rows') {
        result.after.values = raw.map(v => isMissing(v) ? String(fillValue) : v);
      }

    } else if (action === 'handle_outlier' && col && col.stats) {
      const s = col.stats;
      const lower = s.q1 - 1.5 * s.iqr;
      const upper = s.q3 + 1.5 * s.iqr;
      const outlierIndices = [];
      raw.forEach((v, i) => {
        const n = Number(v);
        if (!isNaN(n) && (n < lower || n > upper)) outlierIndices.push(i);
      });
      result.affectedIndices = outlierIndices;

      switch (method) {
        case 'keep':
          result.after.values = [...raw];
          break;
        case 'clip':
          result.after.values = raw.map(v => {
            const n = Number(v);
            if (isNaN(n)) return v;
            if (n < lower) return String(parseFloat(lower.toFixed(4)));
            if (n > upper) return String(parseFloat(upper.toFixed(4)));
            return v;
          });
          break;
        case 'remove':
          result.after.values = raw.map((v, i) => outlierIndices.includes(i) ? '__DROPPED__' : v);
          break;
        case 'median':
          result.after.values = raw.map((v, i) => outlierIndices.includes(i) ? String(s.median) : v);
          break;
      }
    } else if (action === 'encode' || action === 'decompose_date' || action === 'interaction') {
      // These don't modify values directly — just show informational preview
      result.affectedIndices = raw.slice(0, 20).map((_, i) => i);
      result.after.values = [...raw];
    }

    // Compute stats for before/after
    if (col && col.type === 'numeric') {
      const beforeNums = result.before.values.filter(v => !isMissing(v)).map(Number).filter(n => !isNaN(n));
      const afterNums = result.after.values.filter(v => !isMissing(v) && v !== '__DROPPED__').map(Number).filter(n => !isNaN(n));
      result.before.stats = beforeNums.length > 0 ? this.computeNumericStats(beforeNums) : null;
      result.after.stats = afterNums.length > 0 ? this.computeNumericStats(afterNums) : null;
      result.before.distribution = beforeNums.length > 0 ? this.computeHistogram(beforeNums, 12) : [];
      result.after.distribution = afterNums.length > 0 ? this.computeHistogram(afterNums, 12) : [];
    }

    return result;
  },

  // ---- EXPORT CLEANED DATA ----
  exportCleanedCSV() {
    if (!this.currentDataset) return '';
    const { headers, data } = this.currentDataset;
    const headerLine = headers.map(h => `"${h}"`).join(',');
    const dataLines = data.map(row => row.map(v => `"${v}"`).join(','));
    return [headerLine, ...dataLines].join('\n');
  },
};
