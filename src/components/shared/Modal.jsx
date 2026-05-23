import React, { useEffect } from 'react'

export default function Modal({ id, title, isOpen, onClose, children, footer }) {
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose() }
    if (isOpen) window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [isOpen, onClose])

  if (!isOpen) return null

  return (
    <div
      id={id}
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'var(--modal-overlay)', backdropFilter: 'blur(8px)' }}
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        className="glass-deep rounded-2xl w-full max-w-lg max-h-[88vh] overflow-y-auto animate-fade-in-up"
        style={{ boxShadow: '0 24px 80px rgba(0,0,0,0.7), 0 0 0 1px rgba(56,189,248,0.1)' }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[#1a3a6e]">
          <h3 className="font-display text-[17px] text-[#e0f2fe] font-semibold">{title}</h3>
          <button
            id={`${id}-close`}
            onClick={onClose}
            className="w-8 h-8 rounded-lg flex items-center justify-center text-[#4a6fa5] hover:text-[#e0f2fe] hover:bg-white/5 transition-all text-xl leading-none"
          >
            ×
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-5">{children}</div>

        {/* Footer */}
        {footer && (
          <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-[#0c2040]">
            {footer}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Reusable form field components ────────────────────────
export function FieldLabel({ children }) {
  return (
    <label className="block text-[11px] font-mono text-[#4a6fa5] mb-1.5 uppercase tracking-wider">
      {children}
    </label>
  )
}

export function FieldInput({ id, ...props }) {
  return <input id={id} className="field-input" {...props} />
}

export function FieldTextarea({ id, ...props }) {
  return <textarea id={id} className="field-input" style={{ minHeight: 90, resize: 'vertical' }} {...props} />
}

export function BtnPrimary({ children, onClick, disabled, id, className = '' }) {
  return (
    <button
      id={id}
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-accent hover:bg-accent-l text-white text-[13px] font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed ${className}`}
      style={{ background: 'linear-gradient(135deg, #2563eb, #1d4ed8)' }}
    >
      {children}
    </button>
  )
}

export function BtnGhost({ children, onClick, id, className = '', danger = false }) {
  return (
    <button
      id={id}
      onClick={onClick}
      className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl border text-[13px] font-medium transition-all
        ${danger
          ? 'border-red-800/50 text-red-400 hover:border-red-500 hover:bg-red-500/10'
          : 'border-[#1a3a6e] text-[#4a6fa5] hover:text-[#e0f2fe] hover:border-[#2563eb] hover:bg-white/5'
        } ${className}`}
    >
      {children}
    </button>
  )
}
