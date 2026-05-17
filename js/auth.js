// ===== AUTH MODULE =====
// 對外暴露:
//   AuthClient.token          當前 JWT (null 表示 guest)
//   AuthClient.user           當前 user object (null 表示 guest)
//   AuthClient.isAuthenticated() bool
//   AuthClient.headers()      回傳 fetch 用的 headers (含 Authorization 或空物件)
//   AuthClient.init()         page load 呼叫:讀 localStorage / 接 OAuth callback
//   AuthClient.login(email, password)
//   AuthClient.register(email, password, displayName)
//   AuthClient.logout()
//   AuthClient.startOAuth(provider)  跳轉到 backend OAuth start

const AuthClient = {
  token: null,
  user: null,
  _TOKEN_KEY: 'automl_jwt',
  _USER_KEY: 'automl_user',

  isAuthenticated() { return !!this.token; },

  headers() {
    return this.token ? { Authorization: `Bearer ${this.token}` } : {};
  },

  // 攔截所有對 ApiClient.baseUrl 的 fetch,自動加 Authorization header。
  // 比逐一改 api.js 裡 12 個 fetch 乾淨。其他 origin 的 fetch 不受影響。
  _installFetchInterceptor() {
    if (window.__authFetchInstalled) return;
    window.__authFetchInstalled = true;
    const origFetch = window.fetch.bind(window);
    const self = this;
    window.fetch = function(url, opts = {}) {
      const targetUrl = typeof url === 'string' ? url : (url && url.url) || '';
      if (self.token && typeof ApiClient !== 'undefined' && targetUrl.startsWith(ApiClient.baseUrl)) {
        const headers = new Headers(opts.headers || {});
        if (!headers.has('Authorization')) {
          headers.set('Authorization', `Bearer ${self.token}`);
          opts = { ...opts, headers };
        }
      }
      return origFetch(url, opts);
    };
  },

  async init() {
    this._installFetchInterceptor();
    // 1. 看 URL 有沒有 OAuth callback 帶來的 token / 錯誤
    const url = new URL(window.location.href);
    const tokenFromUrl = url.searchParams.get('token');
    const authError = url.searchParams.get('auth_error');
    if (tokenFromUrl) {
      this.token = tokenFromUrl;
      localStorage.setItem(this._TOKEN_KEY, tokenFromUrl);
      // 清掉 URL 上的 query (避免 reload 時誤用)
      url.searchParams.delete('token');
      window.history.replaceState({}, '', url.pathname + (url.search || '') + url.hash);
    } else {
      // 從 localStorage 撈舊 token
      this.token = localStorage.getItem(this._TOKEN_KEY) || null;
    }

    // 2. 拿到 token → 去 /api/auth/me 驗證並取 user 資料
    if (this.token) {
      try {
        const res = await fetch(`${ApiClient.baseUrl}/api/auth/me`, { headers: this.headers() });
        if (res.ok) {
          const data = await res.json();
          if (data.user) {
            this.user = data.user;
            localStorage.setItem(this._USER_KEY, JSON.stringify(data.user));
          } else {
            this._clear(); // token 失效
          }
        } else {
          this._clear();
        }
      } catch (e) {
        // 後端不通 — 暫時用 cache 的 user
        const cached = localStorage.getItem(this._USER_KEY);
        if (cached) try { this.user = JSON.parse(cached); } catch (e) {}
      }
    }

    if (authError) {
      this._showAuthError(`OAuth 失敗: ${authError}`);
      url.searchParams.delete('auth_error');
      window.history.replaceState({}, '', url.pathname + (url.search || '') + url.hash);
    }

    this._renderHeader();
    this._wireUI();
    // 通知 app.js 認證狀態變了 (例如要重抓 datasets)
    window.dispatchEvent(new CustomEvent('auth:changed', { detail: { user: this.user } }));
  },

  async login(email, password) {
    const res = await fetch(`${ApiClient.baseUrl}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `登入失敗 (${res.status})`);
    }
    const data = await res.json();
    this._setSession(data.token, data.user);
  },

  async register(email, password, displayName) {
    const res = await fetch(`${ApiClient.baseUrl}/api/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, displayName }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `註冊失敗 (${res.status})`);
    }
    const data = await res.json();
    this._setSession(data.token, data.user);
  },

  startOAuth(provider) {
    // 跳到 backend 的 start endpoint;它會 302 到 Google / GitHub
    window.location.href = `${ApiClient.baseUrl}/api/auth/oauth/${provider}/start`;
  },

  async logout() {
    try { await fetch(`${ApiClient.baseUrl}/api/auth/logout`, { method: 'POST', headers: this.headers() }); }
    catch (e) {}
    this._clear();
    this._renderHeader();
    window.dispatchEvent(new CustomEvent('auth:changed', { detail: { user: null } }));
  },

  _setSession(token, user) {
    this.token = token;
    this.user = user;
    localStorage.setItem(this._TOKEN_KEY, token);
    localStorage.setItem(this._USER_KEY, JSON.stringify(user));
    this._renderHeader();
    this._closeModal();
    window.dispatchEvent(new CustomEvent('auth:changed', { detail: { user } }));
  },

  _clear() {
    this.token = null;
    this.user = null;
    localStorage.removeItem(this._TOKEN_KEY);
    localStorage.removeItem(this._USER_KEY);
  },

  // ========== UI ==========
  _renderHeader() {
    const guest = document.getElementById('auth-guest-section');
    const user = document.getElementById('auth-user-section');
    if (!guest || !user) return;
    if (this.user) {
      guest.classList.add('hidden');
      user.classList.remove('hidden');
      const u = this.user;
      const nameEl = document.getElementById('auth-user-name');
      const emailEl = document.getElementById('auth-user-email');
      const provEl = document.getElementById('auth-user-provider');
      const avatarImg = document.getElementById('auth-user-avatar');
      const avatarLetter = document.getElementById('auth-user-avatar-letter');
      if (nameEl) nameEl.textContent = u.displayName || u.email.split('@')[0];
      if (emailEl) emailEl.textContent = u.email;
      if (provEl) provEl.textContent = u.oauthProvider ? `透過 ${u.oauthProvider} 登入` : 'Email 登入';
      if (u.avatarUrl && avatarImg) {
        avatarImg.src = u.avatarUrl;
        avatarImg.classList.remove('hidden');
        if (avatarLetter) avatarLetter.classList.add('hidden');
      } else {
        if (avatarImg) avatarImg.classList.add('hidden');
        if (avatarLetter) {
          avatarLetter.textContent = (u.displayName || u.email).charAt(0).toUpperCase();
          avatarLetter.classList.remove('hidden');
        }
      }
    } else {
      guest.classList.remove('hidden');
      user.classList.add('hidden');
    }
  },

  _wireUI() {
    // Open / close modal
    document.getElementById('btn-auth-open')?.addEventListener('click', () => this._openModal('login'));
    document.getElementById('btn-auth-close')?.addEventListener('click', () => this._closeModal());
    document.getElementById('auth-modal')?.addEventListener('click', e => {
      if (e.target.id === 'auth-modal') this._closeModal();
    });

    // Tabs
    document.querySelectorAll('.auth-tab').forEach(btn => {
      btn.addEventListener('click', () => this._switchTab(btn.dataset.authTab));
    });

    // Email form submit
    document.getElementById('auth-form')?.addEventListener('submit', async e => {
      e.preventDefault();
      const mode = document.querySelector('.auth-tab[data-auth-tab].text-primary-400')?.dataset.authTab || 'login';
      const email = document.getElementById('auth-input-email').value.trim();
      const password = document.getElementById('auth-input-password').value;
      const name = document.getElementById('auth-input-name')?.value.trim();
      const submitBtn = document.getElementById('auth-form-submit');
      submitBtn.disabled = true;
      submitBtn.textContent = mode === 'login' ? '登入中...' : '註冊中...';
      try {
        if (mode === 'login') await this.login(email, password);
        else                  await this.register(email, password, name || null);
      } catch (err) {
        this._showAuthError(err.message);
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = mode === 'login' ? '登入' : '註冊';
      }
    });

    // OAuth buttons
    document.getElementById('btn-oauth-google')?.addEventListener('click', () => this.startOAuth('google'));
    document.getElementById('btn-oauth-github')?.addEventListener('click', () => this.startOAuth('github'));

    // User menu toggle
    const menuBtn = document.getElementById('btn-auth-menu');
    const menu = document.getElementById('auth-user-menu');
    if (menuBtn && menu) {
      menuBtn.addEventListener('click', e => {
        e.stopPropagation();
        menu.classList.toggle('hidden');
      });
      document.addEventListener('click', e => {
        if (!menu.classList.contains('hidden') && !menu.contains(e.target) && e.target !== menuBtn) {
          menu.classList.add('hidden');
        }
      });
    }

    // Logout
    document.getElementById('btn-auth-logout')?.addEventListener('click', () => {
      document.getElementById('auth-user-menu')?.classList.add('hidden');
      this.logout();
    });
  },

  _openModal(tab = 'login') {
    document.getElementById('auth-modal')?.classList.remove('hidden');
    this._switchTab(tab);
    this._showAuthError('');
  },

  _closeModal() {
    document.getElementById('auth-modal')?.classList.add('hidden');
    document.getElementById('auth-form')?.reset();
    this._showAuthError('');
  },

  _switchTab(tab) {
    const title = document.getElementById('auth-modal-title');
    const submit = document.getElementById('auth-form-submit');
    const nameWrap = document.getElementById('auth-name-wrap');
    document.querySelectorAll('.auth-tab').forEach(btn => {
      const active = btn.dataset.authTab === tab;
      btn.classList.toggle('text-primary-400', active);
      btn.classList.toggle('border-primary-500', active);
      btn.classList.toggle('text-dark-400', !active);
      btn.classList.toggle('border-transparent', !active);
    });
    if (title) title.textContent = tab === 'login' ? '登入' : '註冊';
    if (submit) submit.textContent = tab === 'login' ? '登入' : '註冊';
    if (nameWrap) nameWrap.classList.toggle('hidden', tab === 'login');
    this._showAuthError('');
  },

  _showAuthError(msg) {
    const el = document.getElementById('auth-form-error');
    if (!el) return;
    if (msg) { el.textContent = msg; el.classList.remove('hidden'); }
    else     { el.textContent = ''; el.classList.add('hidden'); }
  },
};
