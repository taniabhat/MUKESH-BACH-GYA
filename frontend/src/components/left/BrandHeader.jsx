import React from 'react'
import { useApp } from '../../context/AppContext'

export default function BrandHeader({ onSidebarToggle }) {
  const { wsStatus, currentProject } = useApp()

  return (
    <div
      id="brand-header"
      className="flex items-center gap-3 px-2 py-4 flex-shrink-0"
      style={{ borderBottom: 'var(--left-panel-border)' }}
    >
      {/* Sidebar Toggle */}
      <button
        id="sidebar-toggle-btn"
        onClick={onSidebarToggle}
        className="w-7 h-7 flex flex-col items-center justify-center gap-[4px] rounded-lg hover:bg-white/5 transition-colors flex-shrink-0"
        title="Toggle sidebar"
      >
        <span className="w-4 h-[1.5px] rounded" style={{ background: 'var(--text-m)' }} />
        <span className="w-4 h-[1.5px] rounded" style={{ background: 'var(--text-m)' }} />
        <span className="w-3 h-[1.5px] rounded" style={{ background: 'var(--text-m)' }} />
      </button>

      {/* Logo Mark */}
      <div
        className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
        style={{
          background: 'linear-gradient(135deg, #8B008B, #D80073)',
          boxShadow: '0 4px 12px rgba(139,0,139,0.3)',
        }}
      >
        <span className="font-display font-bold text-[17px] text-white leading-none">R</span>
      </div>

      {/* Brand Text */}
      <div className="flex-1 min-w-0">
        <div
          className="text-[26px] font-bold leading-tight"
          style={{ 
            fontFamily: "'Playfair Display', serif",
            color: '#000000'
          }}
        >
          ResearchOS
        </div>
      </div>

      {/* Live indicator */}
      <div className="flex items-center gap-1.5 flex-shrink-0">
        <div
          className={`w-1.5 h-1.5 rounded-full ${
            wsStatus === 'connected' ? 'animate-pulse' : ''
          }`}
          style={{
            background: wsStatus === 'connected' ? 'var(--green)'
              : wsStatus === 'error' ? 'var(--red)'
              : 'var(--text-s)',
            boxShadow: wsStatus === 'connected' ? `0 0 6px var(--green)` : 'none',
          }}
        />
        {currentProject && (
          <span
            className="font-mono text-[9px] hidden sm:block"
            style={{ color: 'var(--text-s)' }}
          >
            {wsStatus === 'connected' ? 'LIVE' : 'POLL'}
          </span>
        )}
      </div>
    </div>
  )
}
