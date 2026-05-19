// =============================================================================
// App shell — routing + chrome + auth bootstrap
// =============================================================================

function App() {
  // 1. 把 OAuth callback 帶來的 ?token=... 收進 localStorage (idempotent)
  React.useMemo(() => { NewUI.api.consumeUrlToken(); }, []);

  // 2. 沒 token → 跳回舊介面的登入頁,整個 app 不 render
  const authed = NewUI.api.isAuthed();
  React.useEffect(() => {
    if (!authed) {
      const next = encodeURIComponent('index-new.html' + window.location.hash);
      window.location.href = `index.html?need_login=1&next=${next}`;
    }
  }, [authed]);
  if (!authed) {
    return (
      <div style={{ padding: 40, color: 'var(--fg-muted)', fontFamily: 'Inter' }}>
        未登入,正在跳回舊介面的登入頁...
      </div>
    );
  }

  // 3. 拉真 user info (失敗 / 401 → 跳登入頁)
  const [authedUser, setAuthedUser] = React.useState(() => NewUI.api.cachedUser());
  React.useEffect(() => {
    NewUI.api.me().then(d => {
      const u = d && d.user;
      if (!u) return;
      setAuthedUser(u);
      NewUI.api.setUserCache(u);
      // 同步 MOCK.user 給目前還沒用 useAuthUser 的元件
      MOCK.user = {
        name: u.displayName || u.email.split('@')[0],
        email: u.email,
        provider: u.oauthProvider || 'email',
        avatarUrl: u.avatarUrl || null,
      };
    }).catch(err => {
      if (err && err.status === 401) {
        NewUI.api.clearSession();
        window.location.href = 'index.html?need_login=1&reason=expired';
      } else {
        console.warn('[new-ui] /api/auth/me 失敗,使用 cached user', err);
      }
    });
  }, []);

  const [page, setPage] = React.useState(() => {
    const hash = window.location.hash.slice(1);
    const valid = ['dashboard', 'data', 'experiments', 'models', 'insights', 'settings'];
    return valid.includes(hash) ? hash : 'dashboard';
  });

  function navigate(p) {
    setPage(p);
    window.location.hash = p;
    window.scrollTo(0, 0);
  }

  React.useEffect(() => {
    const onHash = () => {
      const h = window.location.hash.slice(1);
      if (h) setPage(h);
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const breadcrumbs = {
    dashboard: null,
    data: '客戶資料 2024',
    experiments: null,
    models: '客戶流失預測 v3',
    insights: null,
    settings: null,
  };

  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      <Sidebar page={page} onNavigate={navigate} user={authedUser} />
      <main style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
        <TopBar page={page} breadcrumb={breadcrumbs[page]} />
        <div style={{ flex: 1 }} key={page}>
          {page === 'dashboard'   && <PageDashboard   onNavigate={navigate} />}
          {page === 'data'        && <PageData        onNavigate={navigate} />}
          {page === 'experiments' && <PageExperiments onNavigate={navigate} />}
          {page === 'models'      && <PageModels      onNavigate={navigate} />}
          {page === 'insights'    && <PageInsights    onNavigate={navigate} />}
          {page === 'settings'    && <PageSettings    onNavigate={navigate} user={authedUser} />}
        </div>
      </main>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
