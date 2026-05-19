// =============================================================================
// Sidebar + TopBar — the app chrome
// =============================================================================

function Sidebar({ page, onNavigate, user }) {
  // user 可能來自 App (登入後),失敗 fallback 到 MOCK.user
  const u = user
    ? { name: user.displayName || (user.email || '').split('@')[0], email: user.email, provider: user.oauthProvider || 'email', avatarUrl: user.avatarUrl }
    : MOCK.user;
  const items = [
    { id: 'dashboard',   label: '儀表板', icon: 'dashboard' },
    { id: 'data',        label: '數據',   icon: 'data',  count: MOCK.counts.datasets },
    { id: 'experiments', label: '實驗',   icon: 'flask', count: MOCK.counts.experiments },
    { id: 'models',      label: '模型',   icon: 'bars',  count: MOCK.counts.models },
    { id: 'insights',    label: '洞察',   icon: 'bulb' },
  ];
  return (
    <aside style={{
      width: 220, flexShrink: 0, height: '100vh',
      background: 'var(--bg-sunken)',
      borderRight: '1px solid var(--bd-subtle)',
      display: 'flex', flexDirection: 'column',
      position: 'sticky', top: 0,
    }}>
      {/* Logo / workspace */}
      <div className="divider-b" style={{ padding: 14 }}>
        <Row gap={10}>
          <div style={{
            width: 28, height: 28, borderRadius: 7,
            background: 'var(--primary-soft)', border: '1px solid var(--primary-line)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: 'var(--primary)', fontWeight: 700, fontSize: 13,
          }} className="mono">A</div>
          <div style={{ minWidth: 0, flex: 1 }}>
            <div className="t-title" style={{ fontSize: 13 }}>AutoML</div>
            <div className="t-label mono" style={{ fontSize: 10 }}>v0.2 · dev</div>
          </div>
          <button className="btn-bare" style={{ padding: 4, color: 'var(--fg-muted)' }}>
            <Icon name="chevronDown" size={12} />
          </button>
        </Row>
      </div>

      {/* Primary nav */}
      <nav style={{ padding: 8, flex: 1, overflow: 'auto' }}>
        <Stack gap={2}>
          {items.map(it => (
            <button key={it.id}
                    onClick={() => onNavigate(it.id)}
                    className={`nav-item ${page === it.id ? 'active' : ''}`}>
              <Icon name={it.icon} size={16} className="nav-icon" />
              <span>{it.label}</span>
              {it.count != null && <span className="nav-count">{it.count}</span>}
            </button>
          ))}
        </Stack>
        <div style={{ height: 1, background: 'var(--bd-subtle)', margin: '14px 6px' }} />
        <Stack gap={2}>
          <button onClick={() => onNavigate('settings')}
                  className={`nav-item ${page === 'settings' ? 'active' : ''}`}>
            <Icon name="cog" size={16} className="nav-icon" />
            <span>設定</span>
          </button>
        </Stack>
      </nav>

      {/* Switch back to classic UI */}
      <div className="divider-t" style={{ padding: 8 }}>
        <button
          onClick={() => {
            try { localStorage.setItem('ui_pref', 'classic'); } catch (e) {}
            window.location.href = 'index.html';
          }}
          className="nav-item"
          style={{ width: '100%' }}
          title="切回原本的介面"
        >
          <Icon name="arrowLeft" size={16} className="nav-icon" />
          <span>返回舊介面</span>
          <span className="nav-count mono" style={{ fontSize: 9 }}>BETA</span>
        </button>
      </div>

      {/* User */}
      <div className="divider-t" style={{ padding: 12 }}>
        <Row gap={10}>
          {u.avatarUrl ? (
            <img src={u.avatarUrl} alt="" style={{
              width: 28, height: 28, borderRadius: '50%',
              objectFit: 'cover',
            }} />
          ) : (
            <div style={{
              width: 28, height: 28, borderRadius: '50%',
              background: 'var(--bg-hover)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: 'var(--fg-strong)', fontSize: 12, fontWeight: 600,
            }}>{(u.name || '?').charAt(0).toUpperCase()}</div>
          )}
          <div style={{ minWidth: 0, flex: 1 }}>
            <div className="t-body-lg fg-1" style={{ lineHeight: 1.2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{u.name}</div>
            <div className="t-label mono" style={{ fontSize: 10, lineHeight: 1.3, marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {u.email}
            </div>
          </div>
          <button className="btn-bare" style={{ padding: 4, color: 'var(--fg-muted)' }} onClick={() => onNavigate('settings')} title="開啟設定">
            <Icon name="ellipsis" size={14} />
          </button>
        </Row>
      </div>
    </aside>
  );
}

function TopBar({ page, breadcrumb }) {
  const titles = {
    dashboard: '儀表板',
    data: '數據',
    experiments: '實驗',
    models: '模型',
    insights: '洞察',
    settings: '設定',
  };
  return (
    <header style={{
      padding: '12px 24px',
      borderBottom: '1px solid var(--bd-subtle)',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      gap: 16,
      background: 'var(--bg-canvas)',
      position: 'sticky', top: 0, zIndex: 10,
    }}>
      <Row gap={8}>
        <span className="fg-3 t-body">{titles[page]}</span>
        {breadcrumb && (
          <>
            <span className="fg-4 mono" style={{ fontSize: 10 }}>/</span>
            <span className="fg-1 t-body" style={{ fontWeight: 500 }}>{breadcrumb}</span>
          </>
        )}
      </Row>

      <Row gap={8}>
        {/* Search */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          background: 'var(--bg-sunken)',
          border: '1px solid var(--bd-subtle)',
          borderRadius: 7, padding: '5px 10px',
          color: 'var(--fg-muted)', fontSize: 12,
          minWidth: 200,
        }}>
          <Icon name="search" size={13} />
          <span>搜尋實驗、模型...</span>
          <span className="mono" style={{ marginLeft: 'auto', color: 'var(--fg-faint)', fontSize: 10, padding: '1px 4px', border: '1px solid var(--bd-default)', borderRadius: 3 }}>⌘K</span>
        </div>

        {/* Mode toggle (測試 / 實作) */}
        <ModeBadge />

        {/* API status (real ping every 30s) */}
        <ApiHealthPill />

        {/* Notifications */}
        <button className="btn btn-ghost btn-sm" style={{ position: 'relative', padding: '5px 8px' }}>
          <Icon name="bell" size={14} />
          <span style={{
            position: 'absolute', top: 2, right: 2,
            width: 7, height: 7, borderRadius: '50%',
            background: 'var(--primary)',
            border: '2px solid var(--bg-canvas)',
          }} />
        </button>
      </Row>
    </header>
  );
}

// ---- Mode badge: 測試 / 實作 (跟舊 UI 的 #mode-badge 對齊) ----
function ModeBadge() {
  const [mode, setMode] = React.useState(() => {
    try { return localStorage.getItem('app_mode') || 'real'; } catch { return 'real'; }
  });
  const isDemo = mode === 'demo';

  function toggle() {
    const next = isDemo ? 'real' : 'demo';
    setMode(next);
    try { localStorage.setItem('app_mode', next); } catch (e) {}
    // 觸發全域事件,讓各頁面可以監聽
    window.dispatchEvent(new CustomEvent('app-mode-change', { detail: { mode: next } }));
  }

  return (
    <button
      onClick={toggle}
      className="btn btn-ghost btn-sm"
      title={isDemo ? '目前為測試模式 — 切換到實作模式' : '目前為實作模式 — 切換到測試模式'}
      style={{
        background: isDemo ? 'var(--warn-soft, color-mix(in srgb, var(--warn) 20%, transparent))' : 'var(--primary-soft)',
        borderColor: isDemo ? 'var(--warn)' : 'var(--primary-line)',
        color: isDemo ? 'var(--warn)' : 'var(--primary)',
      }}
    >
      <Dot tone={isDemo ? 'warn' : 'good'} />
      <span style={{ fontWeight: 600 }}>{isDemo ? '測試模式' : '實作模式'}</span>
      <Icon name="refresh" size={11} />
    </button>
  );
}

// ---- Live API health ping ----
function ApiHealthPill() {
  const [state, setState] = React.useState({ ok: null, latency: null, label: '檢查中…' });

  async function ping() {
    const t0 = performance.now();
    try {
      await NewUI.api.get('/api/health');
      setState({ ok: true, latency: Math.round(performance.now() - t0), label: 'OK' });
    } catch (e) {
      setState({ ok: false, latency: null, label: e.message || 'fail' });
    }
  }

  React.useEffect(() => {
    ping();
    const id = setInterval(ping, 30_000);
    return () => clearInterval(id);
  }, []);

  return (
    <button className="btn btn-ghost btn-sm" onClick={ping}
            title={state.ok === false ? `後端連線失敗: ${state.label}` : '後端連線正常'}>
      <Dot tone={state.ok === true ? 'good' : state.ok === false ? 'bad' : 'muted'} />
      <span>API</span>
      <span className="mono fg-3" style={{ fontSize: 11 }}>
        {state.latency != null ? `${state.latency}ms` : '—'}
      </span>
    </button>
  );
}

Object.assign(window, { Sidebar, TopBar, ModeBadge, ApiHealthPill });
