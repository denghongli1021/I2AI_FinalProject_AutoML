// =============================================================================
// App shell — routing + chrome + auth bootstrap
// =============================================================================

function App() {
  // 1. 把 OAuth callback 帶來的 ?token=... 收進 localStorage (idempotent)
  React.useMemo(() => { NewUI.api.consumeUrlToken(); }, []);

  // 2. 未登入 → 顯示 landing 提供 (a) 登入 (b) Demo 模式繼續
  //    Demo 模式:設一個 sessionStorage flag,讓 API client / UI 知道用 mock,不真的打後端
  const [authed, setAuthed]   = React.useState(() => NewUI.api.isAuthed());
  const [demoMode, setDemoMode] = React.useState(() => sessionStorage.getItem('newUiDemoMode') === '1');

  if (!authed && !demoMode) {
    return <UnauthedLanding
      onLogin={() => {
        const next = encodeURIComponent('index-new.html' + window.location.hash);
        window.location.href = `index.html?need_login=1&next=${next}`;
      }}
      onDemo={() => {
        sessionStorage.setItem('newUiDemoMode', '1');
        setDemoMode(true);
      }}
    />;
  }

  // 3. Demo 模式:用 mock user;真實登入:拉 /api/auth/me
  const [authedUser, setAuthedUser] = React.useState(() => {
    if (demoMode) return { displayName: '訪客 (Demo)', email: 'guest@demo', oauthProvider: 'guest' };
    return NewUI.api.cachedUser();
  });
  React.useEffect(() => {
    if (demoMode) return;     // demo 模式不打 API
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
  }, [demoMode]);

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

  // Breadcrumb 留空 — 各頁的 subtitle 應由各 page component 透過 props 帶出
  // (例如選中的 dataset/model 名稱),避免在這裡寫死假資料。
  const breadcrumbs = {
    dashboard: null,
    data: null,
    experiments: null,
    models: null,
    insights: null,
    settings: null,
  };

  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      <Sidebar page={page} onNavigate={navigate} user={authedUser} />
      <main style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
        {demoMode && <DemoBanner onExit={() => {
          sessionStorage.removeItem('newUiDemoMode');
          window.location.href = 'index.html?need_login=1&next=' + encodeURIComponent('index-new.html');
        }} />}
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

// ─── 未登入著陸頁 ─────────────────────────────────────────────────────────────
function UnauthedLanding({ onLogin, onDemo }) {
  return (
    <div style={{
      minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'var(--bg, #0f172a)', fontFamily: 'Inter, system-ui, sans-serif',
    }}>
      <div style={{
        maxWidth: 480, width: '90%', padding: 32, borderRadius: 16,
        background: 'var(--surface, #1e293b)', border: '1px solid var(--border, #334155)',
        boxShadow: '0 20px 50px rgba(0,0,0,0.4)',
      }}>
        <div style={{ fontSize: 28, fontWeight: 700, color: 'var(--fg, #e2e8f0)', marginBottom: 8 }}>
          AutoML Platform
        </div>
        <div style={{ fontSize: 14, color: 'var(--fg-muted, #94a3b8)', marginBottom: 28 }}>
          歡迎來到新介面。先選擇進入方式:
        </div>
        <button onClick={onLogin} style={{
          width: '100%', padding: '12px 20px', borderRadius: 10, border: 'none',
          background: '#3b82f6', color: 'white', fontSize: 14, fontWeight: 600,
          cursor: 'pointer', marginBottom: 12,
        }}>
          登入 / 註冊
        </button>
        <button onClick={onDemo} style={{
          width: '100%', padding: '12px 20px', borderRadius: 10,
          background: 'transparent', border: '1px solid var(--border, #334155)',
          color: 'var(--fg, #e2e8f0)', fontSize: 14, fontWeight: 500,
          cursor: 'pointer',
        }}>
          以 Demo 模式繼續 (免登入,看示範資料)
        </button>
        <div style={{ marginTop: 20, fontSize: 11, color: 'var(--fg-muted, #94a3b8)', lineHeight: 1.6 }}>
          <strong style={{ color: 'var(--fg, #e2e8f0)' }}>Demo 模式</strong>:
          不會打後端 API,所有頁面顯示內建的示範資料,僅供瀏覽 UI 與圖表。
        </div>
      </div>
    </div>
  );
}

function DemoBanner({ onExit }) {
  return (
    <div style={{
      padding: '8px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      background: 'linear-gradient(90deg, rgba(168,85,247,0.15), rgba(59,130,246,0.10))',
      borderBottom: '1px solid var(--border, #334155)', fontSize: 12,
      color: 'var(--fg, #e2e8f0)',
    }}>
      <div>
        <span style={{
          padding: '2px 8px', borderRadius: 999, background: 'rgba(168,85,247,0.3)',
          color: '#e9d5ff', marginRight: 10, fontWeight: 600, fontSize: 11,
        }}>DEMO</span>
        目前以訪客模式瀏覽,所有資料皆為示範用途。登入後可上傳自己的資料並訓練模型。
      </div>
      <button onClick={onExit} style={{
        padding: '4px 12px', borderRadius: 6, background: 'transparent',
        border: '1px solid var(--border, #334155)', color: 'var(--fg, #e2e8f0)',
        fontSize: 11, cursor: 'pointer',
      }}>
        登入
      </button>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
