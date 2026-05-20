// =============================================================================
// new-ui/api-client.js — thin fetch wrapper that knows the shared JWT
// 不重用 js/api.js,避免綁住舊 UI 的 ApiClient / AuthClient 模組順序。
// 與舊 UI 透過共用的 localStorage key 互通登入狀態:
//   - automl_jwt   舊 UI 設的 JWT token
//   - automl_user  舊 UI 設的 user 物件 cache (login 後寫入)
//   - apiBaseUrl   舊 UI 設定頁的 API 位址 (沒設就用 Render 預設)
// =============================================================================

window.NewUI = window.NewUI || {};

NewUI.api = (() => {
  const TOKEN_KEY = 'automl_jwt';
  const USER_KEY  = 'automl_user';
  const DEFAULT_BASE = 'https://i2ai-automl-api.onrender.com';

  function baseUrl() {
    try { return (localStorage.getItem('apiBaseUrl') || DEFAULT_BASE).replace(/\/$/, ''); }
    catch { return DEFAULT_BASE; }
  }
  function token() { try { return localStorage.getItem(TOKEN_KEY); } catch { return null; } }
  function cachedUser() {
    try {
      const raw = localStorage.getItem(USER_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch { return null; }
  }
  function setUserCache(u) {
    try { localStorage.setItem(USER_KEY, JSON.stringify(u)); } catch (e) {}
  }
  function clearSession() {
    try {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
    } catch (e) {}
  }

  // Pull token out of ?token=... (OAuth callback landing on index-new.html directly)
  function consumeUrlToken() {
    const url = new URL(window.location.href);
    const t = url.searchParams.get('token');
    if (t) {
      try { localStorage.setItem(TOKEN_KEY, t); } catch (e) {}
      url.searchParams.delete('token');
      url.searchParams.delete('auth_error');
      window.history.replaceState({}, '', url.pathname + (url.search || '') + url.hash);
    }
  }

  async function request(path, opts = {}) {
    const headers = new Headers(opts.headers || {});
    const t = token();
    if (t && !headers.has('Authorization')) headers.set('Authorization', `Bearer ${t}`);
    if (opts.body && !(opts.body instanceof FormData) && !headers.has('Content-Type')) {
      headers.set('Content-Type', 'application/json');
    }
    const res = await fetch(`${baseUrl()}${path}`, { ...opts, headers });
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`;
      try {
        const err = await res.json();
        if (err.detail) detail = err.detail;
      } catch (e) {}
      const e = new Error(detail);
      e.status = res.status;
      throw e;
    }
    const ct = res.headers.get('content-type') || '';
    return ct.includes('application/json') ? res.json() : res.text();
  }

  // ----------------------------------------------------------------
  // Simple in-memory GET cache (跨頁切換不用每次重抓)
  //   - 預設 TTL 60 秒 (.getCached)
  //   - 可帶 { force: true } 強制重抓 (例如使用者點重新整理)
  //   - 寫入操作 (upload / delete / train) 用 invalidate(prefix) 清掉相關 entry
  // ----------------------------------------------------------------
  const _cache = new Map();   // url → { data, expires }

  function _cacheGet(url) {
    const e = _cache.get(url);
    if (!e) return null;
    if (e.expires < Date.now()) { _cache.delete(url); return null; }
    return e.data;
  }
  function _cacheSet(url, data, ttlMs) {
    _cache.set(url, { data, expires: Date.now() + ttlMs });
  }
  function invalidate(prefix) {
    for (const k of Array.from(_cache.keys())) {
      if (!prefix || k.startsWith(prefix)) _cache.delete(k);
    }
  }
  function clearCache() { _cache.clear(); }

  async function getCached(path, opts = {}) {
    const ttl = opts.ttlMs ?? 60_000;
    if (!opts.force) {
      const hit = _cacheGet(path);
      if (hit !== null) return hit;
    }
    const data = await request(path, { method: 'GET' });
    _cacheSet(path, data, ttl);
    return data;
  }

  // ----------------------------------------------------------------
  // SSE streaming — POST 一個 body,逐個解析 "data: {...}\n\n" 事件。
  //   body: JSON object (會 JSON.stringify) 或 FormData
  //   onEvent: (ev) => void  每個事件回呼
  //   signal: AbortSignal (取消用)
  // 回傳 Promise,結束時 resolve。
  // ----------------------------------------------------------------
  async function streamSSE(path, body, onEvent, signal) {
    const headers = new Headers();
    const t = token();
    if (t) headers.set('Authorization', `Bearer ${t}`);
    let payload = body;
    if (body && !(body instanceof FormData)) {
      headers.set('Content-Type', 'application/json');
      payload = JSON.stringify(body);
    }
    const res = await fetch(`${baseUrl()}${path}`, {
      method: 'POST', headers, body: payload, signal,
    });
    if (!res.ok || !res.body) {
      let detail = `${res.status} ${res.statusText}`;
      try { const e = await res.json(); if (e.detail) detail = e.detail; } catch (e) {}
      throw new Error(detail);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buf = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) !== -1) {
        const chunk = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        for (const line of chunk.split('\n')) {
          const s = line.trim();
          if (!s.startsWith('data:')) continue;
          const json = s.slice(5).trim();
          if (!json) continue;
          try { onEvent(JSON.parse(json)); }
          catch (e) { console.warn('[streamSSE] parse fail', e, json.slice(0, 80)); }
        }
      }
    }
  }

  return {
    baseUrl,
    token,
    cachedUser,
    setUserCache,
    clearSession,
    consumeUrlToken,
    request,
    // shortcuts
    get:  (path)        => request(path, { method: 'GET' }),
    getCached,
    post: (path, body)  => request(path, { method: 'POST', body: body instanceof FormData ? body : JSON.stringify(body) }),
    del:  (path)        => request(path, { method: 'DELETE' }),
    streamSSE,
    // Cache control
    invalidate,
    clearCache,
    // Auth helpers
    async me() { return this.request('/api/auth/me'); },
    isAuthed() { return !!token(); },
  };
})();
