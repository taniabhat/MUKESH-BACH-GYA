import React, { useEffect, useRef } from 'react'
import { useApp } from '../../context/AppContext'
import { FaRobot, FaServer, FaCube } from 'react-icons/fa'

export default function ChatHistory() {
  const { chatHistory, logLines } = useApp()
  const bottomRef = useRef(null)
  const logRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatHistory])

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logLines])

  return (
    <div className="flex-1 overflow-y-auto flex flex-col">
      {/* ── Chat Messages ─────────────────────────────── */}
      <div className="flex-1 overflow-y-auto px-2 py-3 flex flex-col gap-2.5">
        {chatHistory.length === 0 ? (
          <WelcomeState />
        ) : (
          chatHistory.map(msg => (
            <ChatMessage key={msg.id} msg={msg} />
          ))
        )}
        <div ref={bottomRef} />
      </div>

      {/* ── Live Log Terminal ──────────────────────────── */}
      {logLines.length > 0 && (
        <div className="px-2 pb-3 flex-shrink-0">
          <div className="flex items-center justify-between mb-1.5">
            <span className="font-mono text-[9px] text-[#1e3a5f] uppercase tracking-widest">
              Live Log
            </span>
            <span className="font-mono text-[9px] text-[#1e3a5f]">
              {logLines.length} events
            </span>
          </div>
          <div ref={logRef} className="log-terminal">
            {logLines.map((l, i) => (
              <div
                key={i}
                className={`leading-relaxed ${l.status === 'error' ? 'text-red-400' : ''}`}
              >
                <span style={{ color: 'var(--log-ts)' }}>{l.ts} </span>
                <span style={{ color: 'var(--log-event)' }}>{l.event}</span>
                {l.status && <span style={{ color: 'var(--log-ok)' }}> → {l.status}</span>}
                {l.extra && <span style={{ color: 'var(--log-extra)' }}> {l.extra}</span>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function ChatMessage({ msg }) {
  const isUser   = msg.role === 'user'
  const isError  = msg.isError

  return (
    <div className={`animate-fade-in-up flex flex-col gap-1 ${isUser ? 'items-end' : 'items-start'}`}>
      <div
        className={`max-w-[88%] px-3.5 py-2.5 text-[12.5px] leading-relaxed
          ${isUser   ? 'chat-msg-user text-[#e0f2fe]' : ''}
          ${!isUser && !isError ? 'chat-msg-system text-[#93c5fd]' : ''}
          ${isError  ? 'border border-red-800/30 bg-red-900/10 rounded-xl text-red-400' : ''}
        `}
      >
        {msg.content}
      </div>
      <span className="flex items-center gap-1 font-mono text-[9px] text-[#1e3a5f] px-1">
        {msg.role === 'user' ? 'You' : msg.role === 'agent' ? <><FaRobot /> Agent</> : <><FaServer /> System</>} · {msg.ts}
      </span>
    </div>
  )
}

function WelcomeState() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center text-center py-16 px-6">
      <div
        className="w-16 h-16 rounded-2xl flex items-center justify-center mb-5"
        style={{
          background: 'linear-gradient(135deg, rgba(37,99,235,0.2), rgba(30,58,138,0.1))',
          border: '1px solid rgba(37,99,235,0.2)',
          boxShadow: '0 8px 32px rgba(37,99,235,0.1)',
        }}
      >
        <FaCube className="text-3xl text-blue-500" />
      </div>
      <h2 className="font-display text-[18px] text-[#e0f2fe] mb-2">Autonomous Research Pipeline</h2>
      <p className="text-[12.5px] text-[#4a6fa5] max-w-[240px] leading-relaxed">
        Fill in your research topic above and initialize the agent to begin.
      </p>
    </div>
  )
}
