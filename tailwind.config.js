/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg:       '#040d1a',
        surface:  '#071629',
        panel:    '#0c1f3a',
        border1:  '#0c2040',
        border2:  '#1a3a6e',
        accent:   '#2563eb',
        'accent-l': '#3b82f6',
        'accent-d': '#1e3a8a',
        sky:      '#38bdf8',
        'sky-l':  '#7dd3fc',
        'sky-d':  '#0ea5e9',
        midnight: '#1e3a8a',
        success:  '#10b981',
        danger:   '#ef4444',
        warning:  '#f59e0b',
        'text-h': '#e0f2fe',
        'text-n': '#93c5fd',
        'text-m': '#4a6fa5',
        'text-s': '#1e3a5f',
      },
      fontFamily: {
        display: ['"Playfair Display"', 'serif'],
        mono:    ['"IBM Plex Mono"', 'monospace'],
        sans:    ['"DM Sans"', 'sans-serif'],
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
        '4xl': '2rem',
      },
      backdropBlur: {
        xs: '2px',
      },
      animation: {
        'blob-drift': 'blobDrift 12s ease-in-out infinite alternate',
        'blob-drift-2': 'blobDrift2 16s ease-in-out infinite alternate',
        'blob-drift-3': 'blobDrift3 20s ease-in-out infinite alternate',
        'slide-down': 'slideDown 0.4s cubic-bezier(0.34,1.56,0.64,1)',
        'slide-up': 'slideUp 0.3s ease-in',
        'fade-in': 'fadeIn 0.3s ease',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'shimmer': 'shimmer 2s linear infinite',
        'spin-slow': 'spin 2s linear infinite',
        'typewriter': 'typewriter 0.5s steps(20, end)',
      },
      keyframes: {
        blobDrift: {
          '0%':   { transform: 'translate(0,0) scale(1)' },
          '100%': { transform: 'translate(40px, -30px) scale(1.08)' },
        },
        blobDrift2: {
          '0%':   { transform: 'translate(0,0) scale(1.05)' },
          '100%': { transform: 'translate(-30px, 40px) scale(1)' },
        },
        blobDrift3: {
          '0%':   { transform: 'translate(0,0) scale(1)' },
          '100%': { transform: 'translate(20px, 30px) scale(1.1)' },
        },
        slideDown: {
          '0%':   { transform: 'translateY(-110%)', opacity: '0' },
          '100%': { transform: 'translateY(0)',     opacity: '1' },
        },
        slideUp: {
          '0%':   { transform: 'translateY(0)',     opacity: '1' },
          '100%': { transform: 'translateY(-110%)', opacity: '0' },
        },
        fadeIn: {
          '0%':   { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '0%':   { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
      boxShadow: {
        'glow-accent': '0 0 30px rgba(37,99,235,0.25)',
        'glow-sky':    '0 0 20px rgba(56,189,248,0.3)',
        'clay':        '0 8px 32px rgba(14,30,80,0.4), inset 0 1px 0 rgba(255,255,255,0.06)',
        'clay-hover':  '0 12px 48px rgba(37,99,235,0.2), inset 0 1px 0 rgba(255,255,255,0.08)',
        'glass':       '0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04)',
        'banner':      '0 16px 64px rgba(37,99,235,0.3), 0 4px 16px rgba(0,0,0,0.5)',
      },
    },
  },
  plugins: [],
}
