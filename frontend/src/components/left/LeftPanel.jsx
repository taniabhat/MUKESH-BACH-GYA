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
      {/* Inner wrapper — 20px gap from both resize bars */}
      <div
        className="flex flex-col h-full overflow-hidden"
        style={{ margin: '5px 10px' }}
      >
        {/* Brand Header */}
        <BrandHeader onSidebarToggle={onSidebarToggle} />

        {/* Prompt Input Block */}
        <PromptInputBlock onProjectCreated={onProjectCreated} />

        {/* Separator */}
        <div
          style={{ borderTop: '2px solid var(--border)', marginTop: '24px', marginBottom: '4px' }}
          className="flex items-center gap-2"
        >
          <span
            className="font-display text-[16px] uppercase font-bold tracking-widest py-2"
            style={{ color: 'var(--text-h)' }}
          >
            Conversation
          </span>
        </div>

        {/* Chat + Log */}
        <ChatHistory />
      </div>
    </div>
  )
}
