import React from 'react'
import BrandHeader from './BrandHeader'
import PromptInputBlock from './PromptInputBlock'
import ChatHistory from './ChatHistory'

export default function LeftPanel({ onSidebarToggle, onProjectCreated, style }) {
  return (
    <div
      id="left-panel"
      className="flex flex-col h-full overflow-hidden flex-shrink-0"
      style={{
        background: 'var(--left-panel-bg)',
        borderRight: 'var(--left-panel-border)',
        ...style,
      }}
    >
      {/* Brand Header */}
      <BrandHeader onSidebarToggle={onSidebarToggle} />

      {/* Prompt Input Block */}
      <PromptInputBlock onProjectCreated={onProjectCreated} />

      {/* Separator */}
      <div
        className="mx-4 mb-1 flex items-center gap-2"
        style={{ borderTop: '1px solid var(--border)' }}
      >
        <span
          className="font-mono text-[9px] uppercase tracking-widest py-2"
          style={{ color: 'var(--text-s)' }}
        >
          Conversation
        </span>
      </div>

      {/* Chat + Log */}
      <ChatHistory />
    </div>
  )
}
