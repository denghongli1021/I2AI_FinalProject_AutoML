// =============================================================================
// Shared UI primitives — Icon, Surface, Button, Chip, Dot, Tabs, etc.
// Exposed on window so other Babel scripts can use them.
// =============================================================================

// ---- Icons: just paths/shapes; the Icon component wraps the svg attrs ----
const ICON_PATHS = {
  dashboard:   'M3 12l2-2 7-7 7 7 2 2M5 10v10a1 1 0 001 1h12a1 1 0 001-1V10',
  data:        'M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4',
  flask:       'M9 3h6M10 3v6.5L4.8 18A2 2 0 006.6 21h10.8a2 2 0 001.8-3L14 9.5V3',
  bars:        'M5 21V11M12 21V3M19 21v-7',
  bulb:        'M9 17h6M10 21h4M12 3a6 6 0 00-4 10c1 1 2 2 2 4h4c0-2 1-3 2-4a6 6 0 00-4-10z',
  cog:         'M19.4 15a1.7 1.7 0 00.4 1.9l.1.1a2 2 0 11-2.9 2.9l-.1-.1a1.7 1.7 0 00-1.9-.4 1.7 1.7 0 00-1 1.5V21a2 2 0 11-4 0v-.1a1.7 1.7 0 00-1.1-1.5 1.7 1.7 0 00-1.9.4l-.1.1a2 2 0 11-2.9-2.9l.1-.1a1.7 1.7 0 00.4-1.9 1.7 1.7 0 00-1.5-1H3a2 2 0 110-4h.1a1.7 1.7 0 001.5-1.1 1.7 1.7 0 00-.4-1.9l-.1-.1a2 2 0 112.9-2.9l.1.1a1.7 1.7 0 001.9.4H9a1.7 1.7 0 001-1.5V3a2 2 0 114 0v.1a1.7 1.7 0 001 1.5 1.7 1.7 0 001.9-.4l.1-.1a2 2 0 112.9 2.9l-.1.1a1.7 1.7 0 00-.4 1.9V9a1.7 1.7 0 001.5 1H21a2 2 0 110 4h-.1a1.7 1.7 0 00-1.5 1z',
  search:      'M11 19a8 8 0 100-16 8 8 0 000 16zM21 21l-4-4',
  bell:        'M15 17h5l-1.4-1.4A2 2 0 0118 14.2V11a6 6 0 00-4-5.7V5a2 2 0 00-4 0v.3C7.7 6.2 6 8.4 6 11v3.2c0 .5-.2 1-.6 1.4L4 17h5m6 0a3 3 0 11-6 0',
  check:       'M5 12l4 4L19 8',
  checkCircle: 'M9 12l2 2 4-4m-6.5 9a9 9 0 110-18 9 9 0 010 18z',
  arrowUp:     'M5 10l7-7 7 7M12 3v18',
  arrowRight:  'M5 12h14M13 5l7 7-7 7',
  arrowLeft:   'M19 12H5M11 5l-7 7 7 7',
  plus:        'M12 5v14M5 12h14',
  close:       'M6 6l12 12M6 18L18 6',
  chevronDown: 'M6 9l6 6 6-6',
  chevronUp:   'M6 15l6-6 6 6',
  chevronRight:'M9 6l6 6-6 6',
  download:    'M12 3v12M7 10l5 5 5-5M5 21h14',
  upload:      'M12 21V3m-5 7l5-5 5 5M5 21h14',
  play:        'M6 4l14 8-14 8V4z',
  pause:       'M6 4h4v16H6zM14 4h4v16h-4z',
  filter:      'M3 5h18l-7 9v6l-4-2v-4L3 5z',
  refresh:     'M4 12a8 8 0 0114-5.3L20 9M20 4v5h-5M20 12a8 8 0 01-14 5.3L4 15M4 20v-5h5',
  warning:     'M12 9v4M12 17h.01M4.93 19h14.14a2 2 0 001.75-3L13.74 4.5a2 2 0 00-3.48 0L3.18 16a2 2 0 001.75 3z',
  info:        'M12 8h.01M11 12h1v5h1M21 12a9 9 0 11-18 0 9 9 0 0118 0z',
  zap:         'M13 2L3 14h8l-1 8 10-12h-8l1-8z',
  eye:         'M2 12s3-7 10-7 10 7 10 7-3 7-10 7S2 12 2 12zM12 15a3 3 0 110-6 3 3 0 010 6z',
  ellipsis:    'M5 12h.01M12 12h.01M19 12h.01',
  trash:       'M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6',
  trending:    'M3 17l6-6 4 4 8-8M14 7h7v7',
  sliders:     'M3 6h18M3 12h18M3 18h18M8 6V4m0 4v2M14 12v-2m0 4v2M11 18v-2m0 4v2',
  copy:        'M9 9h10v10H9zM5 5h10v4H9v6H5z',
  external:    'M14 4h6v6M10 14L20 4M19 13v6a1 1 0 01-1 1H5a1 1 0 01-1-1V6a1 1 0 011-1h6',
  cube:        'M3 7l9-4 9 4-9 4-9-4zM3 17l9 4 9-4M3 12l9 4 9-4',
};

function Icon({ name, size = 16, strokeWidth = 1.7, className = '', style = {} }) {
  const d = ICON_PATHS[name];
  if (!d) return null;
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth={strokeWidth}
         strokeLinecap="round" strokeLinejoin="round"
         className={className} style={style}>
      <path d={d} />
    </svg>
  );
}

// ---- Generic primitives ----
function Surface({ children, className = '', style = {} }) {
  return <div className={`surface ${className}`} style={style}>{children}</div>;
}

function CardHeader({ title, subtitle, right }) {
  return (
    <div className="divider-b" style={{ padding: '12px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
      <div style={{ minWidth: 0 }}>
        <div className="t-title">{title}</div>
        {subtitle && <div className="t-label" style={{ marginTop: 2 }}>{subtitle}</div>}
      </div>
      {right && <div style={{ flexShrink: 0 }}>{right}</div>}
    </div>
  );
}

function Button({ variant = 'ghost', size, icon, iconRight, children, onClick, disabled, type = 'button', className = '', style = {} }) {
  const cls = ['btn', `btn-${variant}`, size && `btn-${size}`, className].filter(Boolean).join(' ');
  return (
    <button type={type} className={cls} onClick={onClick} disabled={disabled} style={style}>
      {icon && <Icon name={icon} size={14} />}
      {children}
      {iconRight && <Icon name={iconRight} size={14} />}
    </button>
  );
}

function Chip({ tone, children, icon, className = '' }) {
  const cls = ['chip', tone && `chip-${tone}`, className].filter(Boolean).join(' ');
  return (
    <span className={cls}>
      {icon && <Icon name={icon} size={11} />}
      {children}
    </span>
  );
}

function Dot({ tone = 'muted' }) {
  return <span className={`dot dot-${tone}`} />;
}

function Tabs({ items, value, onChange, className = '' }) {
  return (
    <div className={`tab-row ${className}`}>
      {items.map(it => (
        <button key={it.value} className={`tab ${value === it.value ? 'active' : ''}`} onClick={() => onChange(it.value)}>
          {it.label}
        </button>
      ))}
    </div>
  );
}

function PageTabs({ items, value, onChange }) {
  return (
    <div className="page-tab-row">
      {items.map(it => (
        <button key={it.value} className={`page-tab ${value === it.value ? 'active' : ''}`} onClick={() => onChange(it.value)}>
          {it.label}
          {it.count != null && <span className="mono fg-3" style={{ marginLeft: 6, fontSize: 11 }}>{it.count}</span>}
        </button>
      ))}
    </div>
  );
}

function Toggle({ checked, onChange }) {
  return (
    <label className="toggle">
      <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} />
      <span className="toggle-slider" />
    </label>
  );
}

function Range({ value, onChange, min = 0, max = 100, step = 1, format = v => v }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
      <input type="range" className="range" value={value} min={min} max={max} step={step}
             onChange={e => onChange(parseFloat(e.target.value))} style={{ flex: 1 }} />
      <span className="mono fg-1" style={{ fontSize: 12, minWidth: 36, textAlign: 'right' }}>{format(value)}</span>
    </div>
  );
}

function Empty({ icon, title, body, action }) {
  return (
    <div style={{ padding: '48px 24px', textAlign: 'center' }}>
      {icon && (
        <div style={{ width: 48, height: 48, borderRadius: 12, background: 'var(--bg-hover)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', marginBottom: 14, color: 'var(--fg-muted)' }}>
          <Icon name={icon} size={20} />
        </div>
      )}
      <div className="t-title" style={{ marginBottom: 6 }}>{title}</div>
      {body && <div className="t-body fg-3" style={{ marginBottom: 16, maxWidth: 360, margin: '0 auto 16px' }}>{body}</div>}
      {action}
    </div>
  );
}

// ---- Mini chart primitives (inline SVG, no library) ----
function SparkLine({ data, height = 40, color = 'var(--primary)', strokeWidth = 1.5, fill = false }) {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const w = 100;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return [x, y];
  });
  const path = pts.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(' ');
  const fillPath = `${path} L${w},${height} L0,${height} Z`;
  return (
    <svg width="100%" height={height} viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none">
      {fill && <path d={fillPath} fill={color} opacity="0.1" />}
      <path d={path} fill="none" stroke={color} strokeWidth={strokeWidth} />
    </svg>
  );
}

function MiniBar({ data, height = 16, color = 'var(--primary)' }) {
  const max = Math.max(...data);
  return (
    <div className="minibar" style={{ height }}>
      {data.map((v, i) => (
        <div key={i} className={v === max ? 'peak' : ''}
             style={{ height: `${(v / max) * 100}%`, background: color, flex: 1 }} />
      ))}
    </div>
  );
}

// ---- Modal ----
function Modal({ open, onClose, title, children, footer, width = 640 }) {
  if (!open) return null;
  return (
    <div onClick={onClose}
         style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(2px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100, padding: 20 }}>
      <div onClick={e => e.stopPropagation()} className="surface fade-up"
           style={{ width: '100%', maxWidth: width, maxHeight: '85vh', display: 'flex', flexDirection: 'column' }}>
        <div className="divider-b" style={{ padding: '14px 18px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div className="t-title">{title}</div>
          <button onClick={onClose} className="btn btn-bare" style={{ padding: 4 }}>
            <Icon name="close" size={16} />
          </button>
        </div>
        <div style={{ padding: 18, overflow: 'auto', flex: 1 }}>{children}</div>
        {footer && (
          <div className="divider-t" style={{ padding: '12px 18px', display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}

// ---- Layout helpers ----
function Stack({ gap = 12, children, className = '', style = {} }) {
  return <div className={className} style={{ display: 'flex', flexDirection: 'column', gap, ...style }}>{children}</div>;
}
function Row({ gap = 12, align = 'center', children, className = '', style = {} }) {
  return <div className={className} style={{ display: 'flex', alignItems: align, gap, ...style }}>{children}</div>;
}

// Expose on window so other Babel files can pick them up
Object.assign(window, {
  Icon, Surface, CardHeader,
  Button, Chip, Dot, Tabs, PageTabs,
  Toggle, Range, Empty,
  SparkLine, MiniBar, Modal,
  Stack, Row,
});
