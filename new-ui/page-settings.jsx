// =============================================================================
// Page: Settings
// =============================================================================

function PageSettings({ user }) {
  // user 是後端 /api/auth/me 回來的;fallback 到 cache 或 MOCK
  const apiUser = user || (window.NewUI && NewUI.api.cachedUser()) || null;
  const u = apiUser
    ? { name: apiUser.displayName || (apiUser.email || '').split('@')[0], email: apiUser.email, provider: apiUser.oauthProvider || 'email', avatarUrl: apiUser.avatarUrl }
    : MOCK.user;

  const [useApi, setUseApi] = React.useState(true);
  const [apiUrl, setApiUrl] = React.useState(() => {
    try { return localStorage.getItem('apiBaseUrl') || 'https://i2ai-automl-api.onrender.com'; }
    catch { return 'https://i2ai-automl-api.onrender.com'; }
  });
  const [testSize, setTestSize] = React.useState(20);
  const [seed, setSeed] = React.useState(42);
  const [shapSamples, setShapSamples] = React.useState(200);
  const [showPwModal, setShowPwModal] = React.useState(false);

  function doLogout() {
    if (!confirm('確定要登出嗎?')) return;
    // 呼叫後端 logout (best-effort,不阻塞)
    NewUI.api.request('/api/auth/logout', { method: 'POST' }).catch(() => {});
    NewUI.api.clearSession();
    window.location.href = 'index.html';
  }

  return (
    <div style={{ padding: 24, maxWidth: 880 }} >
      <h1 className="t-h1" style={{ marginBottom: 4 }}>設定</h1>
      <p className="t-label" style={{ marginBottom: 24 }}>個人偏好與系統選項</p>

      <Stack gap={14}>
        {/* ===== 帳號資訊 ===== */}
        <SettingsSection title="帳號資訊" desc="登入身份與身份提供者">
          <div style={{ padding: '14px 16px' }}>
            <Row gap={14} align="center">
              {u.avatarUrl ? (
                <img src={u.avatarUrl} alt="" style={{ width: 56, height: 56, borderRadius: '50%', objectFit: 'cover', border: '1px solid var(--primary-line)' }} />
              ) : (
                <div style={{
                  width: 56, height: 56, borderRadius: '50%',
                  background: 'var(--primary-soft)',
                  border: '1px solid var(--primary-line)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: 'var(--primary)', fontSize: 22, fontWeight: 700,
                }}>{(u.name || '?').charAt(0).toUpperCase()}</div>
              )}
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="t-title">{u.name}</div>
                <div className="t-label mono" style={{ marginTop: 2 }}>{u.email}</div>
                <Row gap={6} style={{ marginTop: 6 }}>
                  <Chip tone="primary" icon="checkCircle" className="mono">
                    {u.provider === 'google' ? 'Google' : u.provider === 'github' ? 'GitHub' : '本地'}
                  </Chip>
                  <Chip className="mono">已驗證</Chip>
                </Row>
              </div>
              <Stack gap={6}>
                <Button variant="ghost" size="sm" icon="external">編輯資料</Button>
                <Button variant="danger" size="sm" icon="arrowLeft" onClick={doLogout}>登出</Button>
              </Stack>
            </Row>
          </div>
        </SettingsSection>

        {/* ===== 第三方登入 ===== */}
        <SettingsSection title="第三方登入" desc="連結 OAuth 帳號以快速登入">
          <SettingRow label="Google" hint={u.provider === 'google' ? `已連結 · ${u.email}` : '尚未連結'}>
            {u.provider === 'google'
              ? <Row gap={6}><Chip tone="good" icon="checkCircle">已連結</Chip><Button variant="ghost" size="sm">解除</Button></Row>
              : <Button variant="ghost" size="sm" icon="external"
                        onClick={() => { window.location.href = `${NewUI.api.baseUrl()}/api/auth/oauth/google/start`; }}>連結 Google</Button>}
          </SettingRow>
          <SettingRow label="GitHub" hint={u.provider === 'github' ? '已連結' : '尚未連結'}>
            {u.provider === 'github'
              ? <Row gap={6}><Chip tone="good" icon="checkCircle">已連結</Chip><Button variant="ghost" size="sm">解除</Button></Row>
              : <Button variant="ghost" size="sm" icon="external"
                        onClick={() => { window.location.href = `${NewUI.api.baseUrl()}/api/auth/oauth/github/start`; }}>連結 GitHub</Button>}
          </SettingRow>
        </SettingsSection>

        {/* ===== 密碼與安全 ===== */}
        <SettingsSection title="密碼與安全" desc="管理本機密碼登入">
          <SettingRow label="修改密碼" hint="OAuth 登入者可建立本機密碼作為備用">
            <Button variant="ghost" size="sm" icon="cog" onClick={() => setShowPwModal(true)}>修改密碼</Button>
          </SettingRow>
          <SettingRow label="兩步驟驗證" hint="目前未啟用">
            <Toggle checked={false} onChange={() => {}} />
          </SettingRow>
          <SettingRow label="登入裝置" hint="目前: Chrome · macOS · 台灣">
            <Button variant="ghost" size="sm">查看活動</Button>
          </SettingRow>
        </SettingsSection>

        {showPwModal && <PasswordModal onClose={() => setShowPwModal(false)} />}

        <SettingsSection title="後端連線" desc="與 FastAPI 服務的連線設定">
          <SettingRow label="使用 Python 後端 API" hint="關閉後僅本地端模擬,不會送出真實訓練">
            <Toggle checked={useApi} onChange={setUseApi} />
          </SettingRow>
          <SettingRow label="API Base URL" hint="目前指向 Render 部署,可改成本地 localhost:8000">
            <Row gap={6}>
              <input className="input mono" value={apiUrl} onChange={e => setApiUrl(e.target.value)} style={{ width: 280, fontSize: 12 }} />
              <Button variant="ghost" size="sm" onClick={() => {
                try { localStorage.setItem('apiBaseUrl', apiUrl.replace(/\/$/, '')); } catch (e) {}
                alert('已儲存,下次 fetch 會用新位址');
              }}>儲存</Button>
            </Row>
          </SettingRow>
          <SettingRow label="連線測試" hint="最近一次測試: 200 OK, 42ms">
            <Row gap={8}>
              <Chip tone="good" icon="checkCircle">健康</Chip>
              <Button variant="ghost" size="sm" icon="refresh">重新測試</Button>
            </Row>
          </SettingRow>
        </SettingsSection>

        <SettingsSection title="訓練預設值" desc="新實驗會自動套用這些預設">
          <SettingRow label="預設測試集比例">
            <div style={{ width: 240 }}>
              <Range value={testSize} onChange={setTestSize} min={10} max={50} step={5} format={v => v + '%'} />
            </div>
          </SettingRow>
          <SettingRow label="隨機種子 (random_state)" hint="固定種子讓切分與結果可重現">
            <input className="input mono" type="number" value={seed} onChange={e => setSeed(parseInt(e.target.value) || 0)}
                   style={{ width: 100, textAlign: 'right' }} />
          </SettingRow>
          <SettingRow label="預設任務類型">
            <select className="input" style={{ width: 160 }}>
              <option>自動偵測</option>
              <option>分類</option>
              <option>回歸</option>
            </select>
          </SettingRow>
        </SettingsSection>

        <SettingsSection title="預設啟用演算法" desc="新實驗會預先勾選這些">
          <div style={{ padding: 14, display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
            {MOCK.algorithms.map(a => (
              <label key={a.id} style={{
                padding: '6px 10px', background: 'var(--bg-sunken)', borderRadius: 6,
                border: '1px solid var(--bd-subtle)', cursor: 'pointer',
                display: 'flex', alignItems: 'center', gap: 8, fontSize: 13,
              }}>
                <input type="checkbox" defaultChecked={a.enabled} style={{ accentColor: 'var(--primary)' }} />
                <span className="fg-1">{a.name}</span>
                <span className="t-label mono fg-4" style={{ marginLeft: 'auto', fontSize: 10 }}>{a.kind}</span>
              </label>
            ))}
          </div>
        </SettingsSection>

        <SettingsSection title="視覺化" desc="洞察頁的計算參數">
          <SettingRow label="SHAP 最大樣本數" hint="樣本越多越準但越慢">
            <input className="input mono" type="number" value={shapSamples} onChange={e => setShapSamples(parseInt(e.target.value) || 0)}
                   style={{ width: 100, textAlign: 'right' }} />
          </SettingRow>
        </SettingsSection>

        <SettingsSection title="其他">
          <SettingRow label="重置所有設定" hint="清除瀏覽器儲存的設定,回到預設值">
            <Button variant="danger" size="sm">重置</Button>
          </SettingRow>
        </SettingsSection>

        <Row style={{ justifyContent: 'flex-end', paddingTop: 4 }}>
          <Button variant="primary">儲存設定</Button>
        </Row>
      </Stack>
    </div>
  );
}

function SettingsSection({ title, desc, children }) {
  return (
    <Surface>
      <CardHeader title={title} subtitle={desc} />
      <div>{children}</div>
    </Surface>
  );
}

function SettingRow({ label, hint, children }) {
  return (
    <Row align="center" style={{
      padding: '14px 16px',
      borderBottom: '1px solid var(--bd-subtle)',
      gap: 16, justifyContent: 'space-between',
    }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div className="t-body-lg fg-1">{label}</div>
        {hint && <div className="t-label" style={{ marginTop: 2 }}>{hint}</div>}
      </div>
      <div style={{ flexShrink: 0 }}>{children}</div>
    </Row>
  );
}

// ---- Change-password modal ----
function PasswordModal({ onClose }) {
  const [oldPw, setOldPw] = React.useState('');
  const [newPw, setNewPw] = React.useState('');
  const [confirmPw, setConfirmPw] = React.useState('');
  const canSubmit = newPw.length >= 8 && newPw === confirmPw;

  return (
    <div style={{
      position: 'fixed', inset: 0,
      background: 'rgba(0,0,0,0.55)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      zIndex: 50,
    }} onClick={onClose}>
      <div className="surface" onClick={e => e.stopPropagation()}
           style={{ width: 420, padding: 0, background: 'var(--bg-surface)' }}>
        <CardHeader title="修改密碼" subtitle="新密碼至少 8 個字元" right={
          <Button variant="bare" size="sm" icon="close" onClick={onClose} />
        } />
        <div style={{ padding: 16 }}>
          <Stack gap={12}>
            <div>
              <div className="t-label" style={{ marginBottom: 6 }}>目前密碼 (OAuth 用戶可留空)</div>
              <input className="input" type="password" value={oldPw} onChange={e => setOldPw(e.target.value)}
                     style={{ width: '100%' }} placeholder="目前密碼" />
            </div>
            <div>
              <div className="t-label" style={{ marginBottom: 6 }}>新密碼</div>
              <input className="input" type="password" value={newPw} onChange={e => setNewPw(e.target.value)}
                     style={{ width: '100%' }} placeholder="至少 8 個字元" />
            </div>
            <div>
              <div className="t-label" style={{ marginBottom: 6 }}>確認新密碼</div>
              <input className="input" type="password" value={confirmPw} onChange={e => setConfirmPw(e.target.value)}
                     style={{ width: '100%' }} placeholder="再輸入一次新密碼" />
              {confirmPw && newPw !== confirmPw && (
                <div className="t-label" style={{ color: 'var(--bad)', marginTop: 4 }}>兩次密碼不一致</div>
              )}
            </div>
          </Stack>
          <Row gap={8} style={{ marginTop: 18, justifyContent: 'flex-end' }}>
            <Button variant="ghost" onClick={onClose}>取消</Button>
            <Button variant="primary" disabled={!canSubmit} onClick={() => { alert('密碼修改 mock — 接後端後實作'); onClose(); }}>
              更新密碼
            </Button>
          </Row>
        </div>
      </div>
    </div>
  );
}

window.PageSettings = PageSettings;
