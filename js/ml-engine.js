// ===== ML ENGINE — Pure JavaScript ML Algorithms =====
const MLEngine = {

  trainedModels: [],
  trainingHistory: [],

  // All available algorithms
  ALGORITHMS: {
    linear_regression: { name: 'Linear Regression', type: 'regression', label: '線性回歸' },
    ridge:             { name: 'Ridge Regression', type: 'regression', label: '嶺回歸 (Ridge)' },
    lasso:             { name: 'Lasso Regression', type: 'regression', label: 'Lasso 回歸' },
    knn_3:             { name: 'KNN (k=3)', type: 'both', label: 'KNN (k=3)' },
    knn_5:             { name: 'KNN (k=5)', type: 'both', label: 'KNN (k=5)' },
    knn_7:             { name: 'KNN (k=7)', type: 'both', label: 'KNN (k=7)' },
    decision_tree:     { name: 'Decision Tree', type: 'both', label: '決策樹' },
    random_forest:     { name: 'Random Forest', type: 'both', label: '隨機森林' },
    gradient_boosting: { name: 'Gradient Boosting', type: 'both', label: '梯度提升 (GBDT)' },
    xgboost:           { name: 'XGBoost', type: 'both', label: 'XGBoost' },
    naive_bayes:       { name: 'Naive Bayes', type: 'classification', label: '樸素貝葉斯' },
    logistic:          { name: 'Logistic Regression', type: 'classification', label: '邏輯回歸' },
    svr:               { name: 'SVR', type: 'regression', label: '支持向量回歸 (SVR)' },
  },

  // ===== DATE FEATURE EXTRACTION =====
  _extractDateFeatures(dateStr) {
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return null;
    const dayOfYear = Math.floor((d - new Date(d.getFullYear(), 0, 0)) / 86400000);
    const month = d.getMonth() + 1; // 1-12
    const year = d.getFullYear();
    return {
      year,
      month,
      day_of_year: dayOfYear,
      month_sin: Math.sin(2 * Math.PI * month / 12),
      month_cos: Math.cos(2 * Math.PI * month / 12),
      day_sin: Math.sin(2 * Math.PI * dayOfYear / 365),
      day_cos: Math.cos(2 * Math.PI * dayOfYear / 365),
    };
  },

  // Get list of date-derived feature names for a datetime column
  getDateFeatureNames(colName) {
    return [
      `${colName}_year`, `${colName}_month`, `${colName}_day_of_year`,
      `${colName}_month_sin`, `${colName}_month_cos`,
      `${colName}_day_sin`, `${colName}_day_cos`,
    ];
  },

  // ===== DATA PREPARATION =====
  prepareData(ds, targetCol, options = {}) {
    const headers = ds.headers;
    const targetIdx = headers.indexOf(targetCol);
    if (targetIdx < 0) throw new Error(`目標變數 "${targetCol}" 不存在`);

    const analysis = ds.analysis || [];
    const selectedFeatures = options.features || null; // null = auto (all numeric + date-derived)
    const taskTypeOverride = options.taskType || null; // null = auto-detect
    const isTimeSeries = options.timeSeries || false;

    // Detect datetime columns and generate derived feature definitions
    const dateCols = []; // { colIdx, featureNames: [...] }
    headers.forEach((h, i) => {
      if (i === targetIdx) return;
      const info = analysis.find(a => a.name === h);
      if (info && info.type === 'datetime') {
        dateCols.push({ colIdx: i, colName: h, featureNames: this.getDateFeatureNames(h) });
      }
    });

    // Determine feature columns (numeric)
    const featureCols = [];
    // Also track which date-derived features are selected
    const activeDateCols = [];
    const activeDateFeatures = new Set();

    headers.forEach((h, i) => {
      if (i === targetIdx) return;
      if (selectedFeatures) {
        if (selectedFeatures.includes(h)) {
          const info = analysis.find(a => a.name === h);
          if (info && info.type === 'numeric') featureCols.push(i);
        }
      } else {
        const info = analysis.find(a => a.name === h);
        if (info && info.type === 'numeric') featureCols.push(i);
      }
    });

    // Check if any date-derived feature is selected
    for (const dc of dateCols) {
      const selectedDf = selectedFeatures
        ? dc.featureNames.filter(fn => selectedFeatures.includes(fn))
        : dc.featureNames; // auto: include all
      if (selectedDf.length > 0) {
        activeDateCols.push(dc);
        selectedDf.forEach(fn => activeDateFeatures.add(fn));
      }
    }

    if (featureCols.length === 0 && activeDateFeatures.size === 0) throw new Error('沒有可用的特徵欄位');

    // Build feature name list
    const featureNames = featureCols.map(i => headers[i]);
    for (const dc of activeDateCols) {
      dc.featureNames.forEach(fn => {
        if (activeDateFeatures.has(fn)) featureNames.push(fn);
      });
    }

    // Build X, y (skip missing)
    const X = [], y = [];
    ds.data.forEach(row => {
      const targetVal = parseFloat(row[targetIdx]);
      if (isNaN(targetVal)) return;
      const features = [];
      let valid = true;
      // Numeric features
      for (const ci of featureCols) {
        const v = parseFloat(row[ci]);
        if (isNaN(v)) { valid = false; break; }
        features.push(v);
      }
      if (!valid) return;
      // Date-derived features
      for (const dc of activeDateCols) {
        const dateFeats = this._extractDateFeatures(row[dc.colIdx]);
        if (!dateFeats) { valid = false; break; }
        const allDf = this.getDateFeatureNames(dc.colName);
        const vals = [dateFeats.year, dateFeats.month, dateFeats.day_of_year,
                      dateFeats.month_sin, dateFeats.month_cos, dateFeats.day_sin, dateFeats.day_cos];
        allDf.forEach((fn, fi) => {
          if (activeDateFeatures.has(fn)) features.push(vals[fi]);
        });
      }
      if (valid) { X.push(features); y.push(targetVal); }
    });

    if (X.length < 10) throw new Error('有效數據不足 10 筆，無法訓練');

    // Detect task type
    const uniqueY = [...new Set(y)];
    let taskType;
    if (taskTypeOverride && taskTypeOverride !== 'auto') {
      taskType = taskTypeOverride;
    } else {
      taskType = uniqueY.length <= 10 ? 'classification' : 'regression';
    }

    // Normalize (z-score)
    const nFeats = featureNames.length;
    const means = new Array(nFeats).fill(0);
    const stds = new Array(nFeats).fill(0);
    for (let j = 0; j < nFeats; j++) {
      let sum = 0;
      for (let i = 0; i < X.length; i++) sum += X[i][j];
      means[j] = sum / X.length;
      let ss = 0;
      for (let i = 0; i < X.length; i++) ss += (X[i][j] - means[j]) ** 2;
      stds[j] = Math.sqrt(ss / X.length) || 1;
    }
    const Xnorm = X.map(row => row.map((v, j) => (v - means[j]) / stds[j]));

    // Train/test split
    const indices = Array.from({ length: X.length }, (_, i) => i);
    if (!isTimeSeries) {
      // Shuffle for cross-sectional
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
    }
    // Time-series: last 20% as test (no shuffle)
    const splitIdx = Math.floor(X.length * 0.8);
    const trainIdx = indices.slice(0, splitIdx);
    const testIdx = indices.slice(splitIdx);

    return {
      X, y, Xnorm,
      featureNames,
      targetName: targetCol,
      taskType, isTimeSeries,
      means, stds,
      trainIdx, testIdx,
      nSamples: X.length,
      nFeatures: nFeats,
    };
  },

  // ===== LINEAR REGRESSION =====
  trainLinearRegression(data) {
    return this._trainLinearModel(data, 0, 0, 'Linear Regression');
  },

  // ===== RIDGE =====
  trainRidge(data, alpha = 1.0) {
    return this._trainLinearModel(data, alpha, 0, 'Ridge Regression');
  },

  // ===== LASSO (coordinate descent approximation) =====
  trainLasso(data, alpha = 0.1) {
    return this._trainLinearModel(data, 0, alpha, 'Lasso Regression');
  },

  _trainLinearModel(data, l2 = 0, l1 = 0, modelName = 'Linear') {
    const t0 = performance.now();
    const { Xnorm, y, trainIdx, testIdx } = data;

    const Xtrain = trainIdx.map(i => [1, ...Xnorm[i]]);
    const ytrain = trainIdx.map(i => y[i]);
    const n = Xtrain[0].length;
    const m = Xtrain.length;

    let weights;
    if (l1 > 0) {
      // Coordinate descent for Lasso
      weights = new Array(n).fill(0);
      for (let iter = 0; iter < 200; iter++) {
        for (let j = 0; j < n; j++) {
          let num = 0, den = 0;
          for (let i = 0; i < m; i++) {
            let pred = 0;
            for (let k = 0; k < n; k++) { if (k !== j) pred += Xtrain[i][k] * weights[k]; }
            const residual = ytrain[i] - pred;
            num += Xtrain[i][j] * residual;
            den += Xtrain[i][j] ** 2;
          }
          if (den === 0) continue;
          const raw = num / den;
          if (j === 0) { weights[j] = raw; } // no penalty on bias
          else {
            const penalty = l1 * m;
            if (raw > penalty / den) weights[j] = raw - penalty / den;
            else if (raw < -penalty / den) weights[j] = raw + penalty / den;
            else weights[j] = 0;
          }
        }
      }
    } else {
      // Normal equation with L2
      const XtX = Array.from({ length: n }, () => new Array(n).fill(0));
      const Xty = new Array(n).fill(0);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          Xty[j] += Xtrain[i][j] * ytrain[i];
          for (let k = 0; k < n; k++) XtX[j][k] += Xtrain[i][j] * Xtrain[i][k];
        }
      }
      const reg = l2 > 0 ? l2 * m : 0.01 * m;
      for (let i = 1; i < n; i++) XtX[i][i] += reg;
      weights = this._solveLinearSystem(XtX, Xty);
    }

    const trainTime = performance.now() - t0;
    const predict = (Xn) => { let s = weights[0]; for (let j = 0; j < Xn.length; j++) s += Xn[j] * weights[j + 1]; return s; };

    return this._evalModel(modelName, 'linear', predict, weights.slice(1), data, trainTime);
  },

  // ===== LOGISTIC REGRESSION (gradient descent) =====
  trainLogistic(data, lr = 0.1, epochs = 300) {
    const t0 = performance.now();
    const { Xnorm, y, trainIdx, testIdx, taskType } = data;
    const nf = Xnorm[0].length;
    const weights = new Array(nf + 1).fill(0); // bias + features

    const sigmoid = z => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, z))));
    const classes = [...new Set(y)].sort();
    const posClass = classes.length === 2 ? classes[1] : classes[0];

    const Xtrain = trainIdx.map(i => Xnorm[i]);
    const ytrain = trainIdx.map(i => y[i] === posClass ? 1 : 0);

    for (let ep = 0; ep < epochs; ep++) {
      const grad = new Array(nf + 1).fill(0);
      for (let i = 0; i < Xtrain.length; i++) {
        let z = weights[0];
        for (let j = 0; j < nf; j++) z += weights[j + 1] * Xtrain[i][j];
        const p = sigmoid(z);
        const err = p - ytrain[i];
        grad[0] += err;
        for (let j = 0; j < nf; j++) grad[j + 1] += err * Xtrain[i][j];
      }
      for (let j = 0; j <= nf; j++) weights[j] -= lr * grad[j] / Xtrain.length;
    }

    const trainTime = performance.now() - t0;
    const predict = (xq) => {
      let z = weights[0];
      for (let j = 0; j < xq.length; j++) z += weights[j + 1] * xq[j];
      return sigmoid(z) >= 0.5 ? posClass : classes.find(c => c !== posClass);
    };

    return this._evalModel('Logistic Regression', 'logistic', predict, weights.slice(1).map(Math.abs), data, trainTime);
  },

  // ===== NAIVE BAYES (Gaussian) =====
  trainNaiveBayes(data) {
    const t0 = performance.now();
    const { Xnorm, y, trainIdx, testIdx } = data;
    const nf = Xnorm[0].length;
    const classes = [...new Set(trainIdx.map(i => y[i]))];

    // Compute mean/std per class per feature
    const stats = {};
    const priors = {};
    classes.forEach(c => {
      const idx = trainIdx.filter(i => y[i] === c);
      priors[c] = idx.length / trainIdx.length;
      stats[c] = [];
      for (let j = 0; j < nf; j++) {
        const vals = idx.map(i => Xnorm[i][j]);
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
        const std = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length) || 0.001;
        stats[c].push({ mean, std });
      }
    });

    const gaussian = (x, mean, std) => Math.exp(-0.5 * ((x - mean) / std) ** 2) / (std * Math.sqrt(2 * Math.PI));

    const predict = (xq) => {
      let bestClass = classes[0], bestScore = -Infinity;
      for (const c of classes) {
        let logP = Math.log(priors[c]);
        for (let j = 0; j < nf; j++) logP += Math.log(gaussian(xq[j], stats[c][j].mean, stats[c][j].std) + 1e-300);
        if (logP > bestScore) { bestScore = logP; bestClass = c; }
      }
      return bestClass;
    };

    const trainTime = performance.now() - t0;
    const importance = new Array(nf).fill(1 / nf); // uniform
    return this._evalModel('Naive Bayes', 'naive_bayes', predict, importance, data, trainTime, true);
  },

  // ===== KNN =====
  trainKNN(data, k = 5) {
    const t0 = performance.now();
    const { Xnorm, y, trainIdx, testIdx, taskType } = data;
    const Xtrain = trainIdx.map(i => Xnorm[i]);
    const ytrain = trainIdx.map(i => y[i]);

    const predict = (xq) => {
      const dists = Xtrain.map((xt, i) => {
        let d = 0;
        for (let j = 0; j < xt.length; j++) d += (xt[j] - xq[j]) ** 2;
        return { d, y: ytrain[i] };
      });
      dists.sort((a, b) => a.d - b.d);
      const neighbors = dists.slice(0, k);
      if (taskType === 'regression') {
        return neighbors.reduce((s, n) => s + n.y, 0) / k;
      }
      const votes = {};
      neighbors.forEach(n => { votes[n.y] = (votes[n.y] || 0) + 1; });
      let best = null, bc = 0;
      for (const [cls, cnt] of Object.entries(votes)) { if (cnt > bc) { best = parseFloat(cls); bc = cnt; } }
      return best;
    };

    const trainTime = performance.now() - t0;
    // Permutation importance (simplified)
    const importance = this._permutationImportance(predict, data);
    return this._evalModel(`KNN (k=${k})`, 'knn', predict, importance, data, trainTime, true);
  },

  // ===== DECISION TREE =====
  trainDecisionTree(data, maxDepth = 6, minSamples = 5) {
    const t0 = performance.now();
    const { Xnorm, y, trainIdx, taskType } = data;
    const tree = this._buildTree(trainIdx.map(i => Xnorm[i]), trainIdx.map(i => y[i]), 0, maxDepth, minSamples, taskType);
    const trainTime = performance.now() - t0;
    const predict = (xq) => this._predictTree(tree, xq);
    const splitCounts = new Array(data.nFeatures).fill(0);
    this._countSplits(tree, splitCounts);
    return this._evalModel('Decision Tree', 'decision_tree', predict, splitCounts, data, trainTime);
  },

  // ===== RANDOM FOREST =====
  trainRandomForest(data, nTrees = 20, maxDepth = 5) {
    const t0 = performance.now();
    const { Xnorm, y, trainIdx, taskType } = data;
    const trees = [];
    for (let t = 0; t < nTrees; t++) {
      const sampleIdx = Array.from({ length: trainIdx.length }, () => trainIdx[Math.floor(Math.random() * trainIdx.length)]);
      trees.push(this._buildTree(sampleIdx.map(i => Xnorm[i]), sampleIdx.map(i => y[i]), 0, maxDepth, 5, taskType));
    }
    const trainTime = performance.now() - t0;
    const predict = (xq) => this._aggregateTreePredictions(trees, xq, taskType);
    const splitCounts = new Array(data.nFeatures).fill(0);
    trees.forEach(tree => this._countSplits(tree, splitCounts));
    return this._evalModel(`Random Forest (${nTrees} trees)`, 'random_forest', predict, splitCounts, data, trainTime);
  },

  // ===== GRADIENT BOOSTING =====
  trainGradientBoosting(data, nTrees = 30, maxDepth = 3, learningRate = 0.1) {
    const t0 = performance.now();
    const { Xnorm, y, trainIdx, testIdx, taskType } = data;
    const Xtrain = trainIdx.map(i => Xnorm[i]);
    const ytrain = trainIdx.map(i => y[i]);
    const n = Xtrain.length;

    // Initial prediction (mean for regression, log-odds for classification)
    let initPred;
    if (taskType === 'regression') {
      initPred = ytrain.reduce((a, b) => a + b, 0) / n;
    } else {
      const posCount = ytrain.filter(v => v === Math.max(...new Set(ytrain))).length;
      initPred = Math.log((posCount + 1) / (n - posCount + 1));
    }

    const predictions = new Array(n).fill(initPred);
    const trees = [];

    for (let t = 0; t < nTrees; t++) {
      // Compute residuals
      const residuals = new Array(n);
      if (taskType === 'regression') {
        for (let i = 0; i < n; i++) residuals[i] = ytrain[i] - predictions[i];
      } else {
        for (let i = 0; i < n; i++) {
          const p = 1 / (1 + Math.exp(-predictions[i]));
          const target = ytrain[i] === Math.max(...new Set(ytrain)) ? 1 : 0;
          residuals[i] = target - p;
        }
      }
      const tree = this._buildTree(Xtrain, residuals, 0, maxDepth, 5, 'regression');
      trees.push(tree);
      for (let i = 0; i < n; i++) {
        predictions[i] += learningRate * this._predictTree(tree, Xtrain[i]);
      }
    }

    const trainTime = performance.now() - t0;
    const classes = [...new Set(y)].sort();
    const posClass = classes.length >= 2 ? Math.max(...classes) : classes[0];

    const predict = (xq) => {
      let pred = initPred;
      for (const tree of trees) pred += learningRate * this._predictTree(tree, xq);
      if (taskType === 'regression') return pred;
      return 1 / (1 + Math.exp(-pred)) >= 0.5 ? posClass : classes.find(c => c !== posClass);
    };

    const splitCounts = new Array(data.nFeatures).fill(0);
    trees.forEach(tree => this._countSplits(tree, splitCounts));
    return this._evalModel('Gradient Boosting', 'gradient_boosting', predict, splitCounts, data, trainTime);
  },

  // ===== XGBOOST (simplified) =====
  trainXGBoost(data, nTrees = 40, maxDepth = 4, learningRate = 0.1, lambda = 1.0) {
    const t0 = performance.now();
    const { Xnorm, y, trainIdx, taskType } = data;
    const Xtrain = trainIdx.map(i => Xnorm[i]);
    const ytrain = trainIdx.map(i => y[i]);
    const n = Xtrain.length;

    const classes = [...new Set(y)].sort();
    const posClass = classes.length >= 2 ? Math.max(...classes) : classes[0];
    const isClassification = taskType === 'classification';

    let initPred;
    if (isClassification) {
      const posCount = ytrain.filter(v => v === posClass).length;
      initPred = Math.log((posCount + 1) / (n - posCount + 1));
    } else {
      initPred = ytrain.reduce((a, b) => a + b, 0) / n;
    }

    const preds = new Array(n).fill(initPred);
    const trees = [];

    for (let t = 0; t < nTrees; t++) {
      const grad = new Array(n);
      const hess = new Array(n);
      if (isClassification) {
        for (let i = 0; i < n; i++) {
          const p = 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, preds[i]))));
          const label = ytrain[i] === posClass ? 1 : 0;
          grad[i] = p - label;
          hess[i] = p * (1 - p) + 1e-8;
        }
      } else {
        for (let i = 0; i < n; i++) { grad[i] = preds[i] - ytrain[i]; hess[i] = 1; }
      }

      const tree = this._buildXGBTree(Xtrain, grad, hess, 0, maxDepth, lambda);
      trees.push(tree);
      for (let i = 0; i < n; i++) preds[i] += learningRate * this._predictTree(tree, Xtrain[i]);
    }

    const trainTime = performance.now() - t0;
    const predict = (xq) => {
      let pred = initPred;
      for (const tree of trees) pred += learningRate * this._predictTree(tree, xq);
      if (isClassification) return 1 / (1 + Math.exp(-pred)) >= 0.5 ? posClass : classes.find(c => c !== posClass);
      return pred;
    };

    const splitCounts = new Array(data.nFeatures).fill(0);
    trees.forEach(tree => this._countSplits(tree, splitCounts));
    return this._evalModel('XGBoost', 'xgboost', predict, splitCounts, data, trainTime);
  },

  _buildXGBTree(X, grad, hess, depth, maxDepth, lambda) {
    const n = X.length;
    if (depth >= maxDepth || n < 5) {
      const sumG = grad.reduce((a, b) => a + b, 0);
      const sumH = hess.reduce((a, b) => a + b, 0);
      return { leaf: true, value: -sumG / (sumH + lambda) };
    }

    const nf = X[0].length;
    let bestGain = 0, bestF = -1, bestThr = 0;
    const sumG = grad.reduce((a, b) => a + b, 0);
    const sumH = hess.reduce((a, b) => a + b, 0);

    for (let f = 0; f < nf; f++) {
      const sorted = Array.from({ length: n }, (_, i) => i).sort((a, b) => X[a][f] - X[b][f]);
      let leftG = 0, leftH = 0;
      for (let s = 0; s < n - 1; s++) {
        const i = sorted[s];
        leftG += grad[i]; leftH += hess[i];
        const rightG = sumG - leftG, rightH = sumH - leftH;
        if (s % Math.max(1, Math.floor(n / 20)) !== 0) continue;
        const gain = (leftG ** 2 / (leftH + lambda)) + (rightG ** 2 / (rightH + lambda)) - (sumG ** 2 / (sumH + lambda));
        if (gain > bestGain) {
          bestGain = gain;
          bestF = f;
          bestThr = (X[sorted[s]][f] + X[sorted[s + 1]][f]) / 2;
        }
      }
    }

    if (bestF < 0) {
      return { leaf: true, value: -sumG / (sumH + lambda) };
    }

    const leftIdx = [], rightIdx = [];
    for (let i = 0; i < n; i++) {
      if (X[i][bestF] <= bestThr) leftIdx.push(i); else rightIdx.push(i);
    }
    if (leftIdx.length === 0 || rightIdx.length === 0) {
      return { leaf: true, value: -sumG / (sumH + lambda) };
    }

    return {
      leaf: false, feature: bestF, threshold: bestThr,
      left: this._buildXGBTree(leftIdx.map(i => X[i]), leftIdx.map(i => grad[i]), leftIdx.map(i => hess[i]), depth + 1, maxDepth, lambda),
      right: this._buildXGBTree(rightIdx.map(i => X[i]), rightIdx.map(i => grad[i]), rightIdx.map(i => hess[i]), depth + 1, maxDepth, lambda),
    };
  },

  // ===== SVR (simplified linear kernel with SGD) =====
  trainSVR(data, C = 1.0, epsilon = 0.1, epochs = 200) {
    const t0 = performance.now();
    const { Xnorm, y, trainIdx } = data;
    const nf = Xnorm[0].length;
    const weights = new Array(nf).fill(0);
    let bias = 0;
    const lr = 0.01;

    for (let ep = 0; ep < epochs; ep++) {
      for (const i of trainIdx) {
        let pred = bias;
        for (let j = 0; j < nf; j++) pred += weights[j] * Xnorm[i][j];
        const diff = y[i] - pred;
        if (Math.abs(diff) > epsilon) {
          const sign = diff > 0 ? 1 : -1;
          bias += lr * C * sign;
          for (let j = 0; j < nf; j++) {
            weights[j] += lr * (C * sign * Xnorm[i][j] - (weights[j] / (trainIdx.length)));
          }
        } else {
          for (let j = 0; j < nf; j++) weights[j] *= (1 - lr / trainIdx.length);
        }
      }
    }

    const trainTime = performance.now() - t0;
    const predict = (xq) => { let s = bias; for (let j = 0; j < nf; j++) s += weights[j] * xq[j]; return s; };
    return this._evalModel('SVR', 'svr', predict, weights.map(Math.abs), data, trainTime);
  },

  // ===== SHARED HELPERS =====
  _evalModel(name, type, predict, rawImportance, data, trainTime, importanceIsNormalized = false) {
    const { Xnorm, y, trainIdx, testIdx, taskType } = data;
    const trainPred = trainIdx.map(i => predict(Xnorm[i]));
    const testPred = testIdx.map(i => predict(Xnorm[i]));
    const testTrue = testIdx.map(i => y[i]);
    const trainTrue = trainIdx.map(i => y[i]);

    const metrics = taskType === 'regression'
      ? this._regressionMetrics(testTrue, testPred, trainTrue, trainPred)
      : this._classificationMetrics(testTrue, testPred, trainTrue, trainPred);

    let featureImportance;
    if (importanceIsNormalized) {
      featureImportance = rawImportance;
    } else {
      const abs = rawImportance.map(Math.abs);
      const sum = abs.reduce((a, b) => a + b, 0) || 1;
      featureImportance = abs.map(v => v / sum);
    }

    return { name, type, predict, metrics, trainTime, featureImportance, testPred, testTrue };
  },

  _permutationImportance(predict, data) {
    const { Xnorm, y, testIdx, taskType, trainIdx } = data;
    const testTrue = testIdx.map(i => y[i]);
    const trainTrue = trainIdx.map(i => y[i]);
    const trainPred = trainIdx.map(i => predict(Xnorm[i]));
    const testPred = testIdx.map(i => predict(Xnorm[i]));
    const baseMetrics = taskType === 'regression'
      ? this._regressionMetrics(testTrue, testPred, trainTrue, trainPred)
      : this._classificationMetrics(testTrue, testPred, trainTrue, trainPred);
    const baseScore = baseMetrics.testScore;
    const nf = Xnorm[0].length;
    const imp = new Array(nf).fill(0);
    for (let f = 0; f < nf; f++) {
      const shuffled = testIdx.map(i => {
        const x = [...Xnorm[i]];
        x[f] = Xnorm[testIdx[Math.floor(Math.random() * testIdx.length)]][f];
        return x;
      });
      const sp = shuffled.map(x => predict(x));
      const sm = taskType === 'regression'
        ? this._regressionMetrics(testTrue, sp, trainTrue, trainPred)
        : this._classificationMetrics(testTrue, sp, trainTrue, trainPred);
      imp[f] = Math.max(0, baseScore - sm.testScore);
    }
    const sum = imp.reduce((a, b) => a + b, 0) || 1;
    return imp.map(v => v / sum);
  },

  _aggregateTreePredictions(trees, xq, taskType) {
    const preds = trees.map(tree => this._predictTree(tree, xq));
    if (taskType === 'regression') return preds.reduce((a, b) => a + b, 0) / preds.length;
    const counts = {};
    preds.forEach(p => { counts[p] = (counts[p] || 0) + 1; });
    let best = preds[0], bc = 0;
    for (const [cls, cnt] of Object.entries(counts)) { if (cnt > bc) { best = parseFloat(cls); bc = cnt; } }
    return best;
  },

  _buildTree(X, y, depth, maxDepth, minSamples, taskType) {
    if (depth >= maxDepth || y.length < minSamples) return { leaf: true, value: this._leafValue(y, taskType) };
    if (new Set(y).size === 1) return { leaf: true, value: y[0] };

    let bestF = -1, bestThr = 0, bestScore = Infinity;
    const nf = X[0].length;
    for (let f = 0; f < nf; f++) {
      const vals = X.map(r => r[f]).sort((a, b) => a - b);
      const step = Math.max(1, Math.floor(vals.length / 20));
      for (let i = step; i < vals.length; i += step) {
        const thr = (vals[i - 1] + vals[i]) / 2;
        const lIdx = [], rIdx = [];
        for (let j = 0; j < X.length; j++) { if (X[j][f] <= thr) lIdx.push(j); else rIdx.push(j); }
        if (lIdx.length < 2 || rIdx.length < 2) continue;
        const score = (lIdx.length * this._impurity(lIdx.map(j => y[j]), taskType) +
                       rIdx.length * this._impurity(rIdx.map(j => y[j]), taskType)) / y.length;
        if (score < bestScore) { bestScore = score; bestF = f; bestThr = thr; }
      }
    }
    if (bestF < 0) return { leaf: true, value: this._leafValue(y, taskType) };

    const lIdx = [], rIdx = [];
    for (let i = 0; i < X.length; i++) { if (X[i][bestF] <= bestThr) lIdx.push(i); else rIdx.push(i); }
    return {
      leaf: false, feature: bestF, threshold: bestThr,
      left: this._buildTree(lIdx.map(i => X[i]), lIdx.map(i => y[i]), depth + 1, maxDepth, minSamples, taskType),
      right: this._buildTree(rIdx.map(i => X[i]), rIdx.map(i => y[i]), depth + 1, maxDepth, minSamples, taskType),
    };
  },

  _predictTree(node, x) {
    if (node.leaf) return node.value;
    return x[node.feature] <= node.threshold ? this._predictTree(node.left, x) : this._predictTree(node.right, x);
  },

  _leafValue(y, taskType) {
    if (taskType === 'regression') return y.reduce((a, b) => a + b, 0) / y.length;
    const counts = {};
    y.forEach(v => { counts[v] = (counts[v] || 0) + 1; });
    let best = y[0], bc = 0;
    for (const [cls, cnt] of Object.entries(counts)) { if (cnt > bc) { best = parseFloat(cls); bc = cnt; } }
    return best;
  },

  _impurity(y, taskType) {
    if (taskType === 'regression') { const m = y.reduce((a, b) => a + b, 0) / y.length; return y.reduce((s, v) => s + (v - m) ** 2, 0) / y.length; }
    const counts = {};
    y.forEach(v => { counts[v] = (counts[v] || 0) + 1; });
    let g = 1;
    for (const cnt of Object.values(counts)) { const p = cnt / y.length; g -= p * p; }
    return g;
  },

  _countSplits(node, counts) {
    if (node.leaf) return;
    if (node.feature < counts.length) counts[node.feature] += 1;
    this._countSplits(node.left, counts);
    this._countSplits(node.right, counts);
  },

  // ===== METRICS =====
  _regressionMetrics(testTrue, testPred, trainTrue, trainPred) {
    const mse = (yt, yp) => yt.reduce((s, v, i) => s + (v - yp[i]) ** 2, 0) / yt.length;
    const mae = (yt, yp) => yt.reduce((s, v, i) => s + Math.abs(v - yp[i]), 0) / yt.length;
    const r2 = (yt, yp) => { const m = yt.reduce((a, b) => a + b, 0) / yt.length; const tot = yt.reduce((s, v) => s + (v - m) ** 2, 0); const res = yt.reduce((s, v, i) => s + (v - yp[i]) ** 2, 0); return tot > 0 ? 1 - res / tot : 0; };
    const testR2 = r2(testTrue, testPred);
    return { taskType: 'regression', testMSE: mse(testTrue, testPred), testMAE: mae(testTrue, testPred), testRMSE: Math.sqrt(mse(testTrue, testPred)), testR2, trainR2: r2(trainTrue, trainPred), trainMSE: mse(trainTrue, trainPred), testScore: testR2 };
  },

  _classificationMetrics(testTrue, testPred, trainTrue, trainPred) {
    const accuracy = (yt, yp) => yt.filter((v, i) => v === yp[i]).length / yt.length;
    const testAcc = accuracy(testTrue, testPred);
    const trainAcc = accuracy(trainTrue, trainPred);
    const classes = [...new Set([...testTrue, ...trainTrue])].sort();
    let precision = testAcc, recall = testAcc, f1 = testAcc;
    if (classes.length === 2) {
      const pos = classes[1];
      let tp = 0, fp = 0, fn = 0;
      for (let i = 0; i < testTrue.length; i++) {
        if (testPred[i] === pos && testTrue[i] === pos) tp++;
        if (testPred[i] === pos && testTrue[i] !== pos) fp++;
        if (testPred[i] !== pos && testTrue[i] === pos) fn++;
      }
      precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
    }
    return { taskType: 'classification', testAccuracy: testAcc, trainAccuracy: trainAcc, precision, recall, f1, classes, testScore: testAcc };
  },

  _solveLinearSystem(A, b) {
    const n = b.length;
    const aug = A.map((row, i) => [...row, b[i]]);
    for (let col = 0; col < n; col++) {
      let maxRow = col, maxVal = Math.abs(aug[col][col]);
      for (let row = col + 1; row < n; row++) { if (Math.abs(aug[row][col]) > maxVal) { maxVal = Math.abs(aug[row][col]); maxRow = row; } }
      [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
      if (Math.abs(aug[col][col]) < 1e-12) continue;
      for (let row = col + 1; row < n; row++) { const f = aug[row][col] / aug[col][col]; for (let j = col; j <= n; j++) aug[row][j] -= f * aug[col][j]; }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) { x[i] = aug[i][n]; for (let j = i + 1; j < n; j++) x[i] -= aug[i][j] * x[j]; x[i] /= aug[i][i] || 1; }
    return x;
  },

  // ===== RUN EXPERIMENT =====
  async runExperiment(ds, targetCol, options = {}, onProgress) {
    const data = this.prepareData(ds, targetCol, options);

    const log = (msg, type = 'info') => { if (onProgress) onProgress({ type: 'log', msg, logType: type }); };

    log(`資料準備完成: ${data.nSamples} 筆有效數據, ${data.nFeatures} 個特徵`, 'success');
    log(`任務類型: ${data.taskType === 'regression' ? '回歸' : '分類'}${data.isTimeSeries ? ' (時間序列)' : ''}`, 'info');
    log(`訓練集: ${data.trainIdx.length} 筆, 測試集: ${data.testIdx.length} 筆${data.isTimeSeries ? ' (依時間順序切分)' : ''}`, 'info');

    // Build algorithm list
    const selectedAlgos = options.algorithms || null; // null = all compatible
    const algoRegistry = {
      linear_regression: () => this.trainLinearRegression(data),
      ridge: () => this.trainRidge(data),
      lasso: () => this.trainLasso(data),
      logistic: () => this.trainLogistic(data),
      naive_bayes: () => this.trainNaiveBayes(data),
      knn_3: () => this.trainKNN(data, 3),
      knn_5: () => this.trainKNN(data, 5),
      knn_7: () => this.trainKNN(data, 7),
      decision_tree: () => this.trainDecisionTree(data),
      random_forest: () => this.trainRandomForest(data),
      gradient_boosting: () => this.trainGradientBoosting(data),
      xgboost: () => this.trainXGBoost(data),
      svr: () => this.trainSVR(data),
    };

    let algoKeys;
    if (selectedAlgos && selectedAlgos.length > 0) {
      algoKeys = selectedAlgos.filter(k => algoRegistry[k]);
    } else {
      // Auto: pick all compatible
      algoKeys = Object.keys(this.ALGORITHMS).filter(k => {
        const a = this.ALGORITHMS[k];
        return a.type === 'both' || a.type === data.taskType;
      });
    }

    const models = [];
    for (let i = 0; i < algoKeys.length; i++) {
      const key = algoKeys[i];
      const info = this.ALGORITHMS[key];
      if (onProgress) onProgress({ type: 'progress', pct: Math.round((i / algoKeys.length) * 100), step: `訓練 ${info.name}` });
      log(`開始訓練 ${info.name}...`, 'info');

      await new Promise(r => setTimeout(r, 80));

      try {
        const model = algoRegistry[key]();
        model.featureNames = data.featureNames;
        model.taskType = data.taskType;
        model.targetName = data.targetName;
        model.trainSize = data.trainIdx.length;
        model.testSize = data.testIdx.length;
        model.means = data.means;
        model.stds = data.stds;
        // Attach per-feature min/mean/max from raw X for What-If slider ranges
        const fStats = [];
        for (let f = 0; f < data.nFeatures; f++) {
          let mn = Infinity, mx = -Infinity, sum = 0;
          for (let i = 0; i < data.X.length; i++) {
            const v = data.X[i][f];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
          }
          fStats.push({ min: mn, max: mx, mean: sum / data.X.length });
        }
        model.featureStats = fStats;
        models.push(model);

        const scoreLabel = data.taskType === 'regression' ? 'R²' : 'Accuracy';
        const score = model.metrics.testScore;
        log(`${info.name} 完成 — ${scoreLabel}: ${score.toFixed(4)}, 耗時: ${model.trainTime.toFixed(0)}ms`, score > 0.5 ? 'success' : 'warning');
      } catch (err) {
        log(`${info.name} 訓練失敗: ${err.message}`, 'error');
      }

      await new Promise(r => setTimeout(r, 30));
    }

    models.sort((a, b) => b.metrics.testScore - a.metrics.testScore);

    if (models.length > 0) {
      const best = models[0];
      const scoreLabel = data.taskType === 'regression' ? 'R²' : 'Accuracy';
      log(`所有模型訓練完成！最佳模型: ${best.name} (${scoreLabel}=${best.metrics.testScore.toFixed(4)})`, 'best');
    }

    if (onProgress) onProgress({ type: 'progress', pct: 100, step: '完成' });

    this.trainedModels = models;
    this.trainingHistory.push({
      timestamp: new Date().toISOString(),
      dataset: ds.fileName || 'unknown',
      target: targetCol,
      taskType: data.taskType,
      models: models.map(m => ({ name: m.name, score: m.metrics.testScore, time: m.trainTime })),
    });

    return { models, data };
  },
};
