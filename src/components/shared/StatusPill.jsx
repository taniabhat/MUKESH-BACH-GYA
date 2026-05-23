import React from 'react'

export default function StatusPill({ status }) {
  const cls = {
    discovering: 'pill-running',
    analyzing:   'pill-running',
    drafting:    'pill-running',
    refining:    'pill-running',
    humanizing:  'pill-running',
    reviewing:   'pill-running',
    complete:    'pill-complete',
    error:       'pill-error',
    approved:    'pill-approved',
  }[status] || 'pill-idle'

  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-0.5 rounded-full border text-[11px] font-mono ${cls}`}>
      {['discovering','analyzing','drafting','refining','humanizing','reviewing'].includes(status) && (
        <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse inline-block" />
      )}
      {status}
    </span>
  )
}
