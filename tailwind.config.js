/** @type {import('tailwindcss').Config} */
// 舊 UI (index.html) 的 Tailwind 設定。
// 原本寫在 index.html 的 inline `tailwind.config`,搬來這裡讓 Tailwind CLI 在 build 時讀取。
// content = Tailwind 去哪裡掃描「有用到的 class」,只留有用到的進最終 css/tailwind.css。
// ※ 新 UI (index-new.html / new-ui/) 不是 Tailwind,用的是 new-ui/tokens.css,故不掃描。
module.exports = {
  content: [
    './index.html',
    './js/**/*.js',
  ],
  theme: {
    extend: {
      colors: {
        primary: { 50: '#eff6ff', 100: '#dbeafe', 200: '#bfdbfe', 300: '#93c5fd', 400: '#60a5fa', 500: '#3b82f6', 600: '#2563eb', 700: '#1d4ed8', 800: '#1e40af', 900: '#1e3a8a' },
        dark: { 50: '#f8fafc', 100: '#f1f5f9', 200: '#e2e8f0', 300: '#cbd5e1', 400: '#94a3b8', 500: '#64748b', 600: '#475569', 700: '#334155', 800: '#1e293b', 900: '#0f172a', 950: '#020617' },
        accent: { 400: '#22d3ee', 500: '#06b6d4', 600: '#0891b2' },
        success: { 400: '#34d399', 500: '#10b981', 600: '#059669' },
        warning: { 400: '#fbbf24', 500: '#f59e0b', 600: '#d97706' },
        danger: { 400: '#f87171', 500: '#ef4444', 600: '#dc2626' },
      },
      fontFamily: { sans: ['Inter', 'system-ui', 'sans-serif'] },
      // 「處理記錄」分類標頭用 bg-*-500/8 (8% 不透明度)。8 不在 Tailwind 預設 opacity 級距,
      // 註冊它,build 才會產生 bg-warning-500/8 這類 class。
      opacity: { 8: '0.08' },
    },
  },
  plugins: [],
};
