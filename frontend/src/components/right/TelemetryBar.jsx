import React, { useEffect, useRef, useState } from 'react'
import { useApp } from '../../context/AppContext'

const MESSAGES = [
  'Agents are standing by…',
  'Ready to initialize pipeline.',
]

export default function TelemetryBar() {
  const { agentStatus, currentProject, isRunning } = useApp()
  const [displayed, setDisplayed] = useState('')
  const [dots, setDots] = useState('.')
  const mountedRef = useRef(true)

  useEffect(() => () => { mountedRef.current = false }, [])

  // Typewriter effect for agent status
  useEffect(() => {
    const target = agentStatus || (currentProject ? '' : MESSAGES[0])
    if (!target) { setDisplayed(''); return }
    let i = 0
    setDisplayed('')
    const interval = setInterval(() => {
      if (!mountedRef.current) { clearInterval(interval); return }
      i++
      setDisplayed(target.slice(0, i))
      if (i >= target.length) clearInterval(interval)
    }, 22)
    return () => clearInterval(interval)
  }, [agentStatus, currentProject])

  // Animated dots
  useEffect(() => {
    if (!isRunning) return
    const t = setInterval(() => {
      setDots(d => d.length >= 3 ? '.' : d + '.')
    }, 400)
    return () => clearInterval(t)
  }, [isRunning])

  return (
    <div
      id="telemetry-bar"
      className="flex items-center gap-3 px-5 py-3 border-b border-[#0c2040] flex-shrink-0"
      style={{ background: 'var(--telemetry-bg)', backdropFilter: 'blur(8px)', borderBottom: '1px solid var(--border)' }}
    >
      {/* Agent Activity Indicator */}
      <div className="flex items-center gap-2 flex-shrink-0">
        {isRunning ? (
          <div className="relative w-3 h-3">
            <span className="absolute inset-0 rounded-full bg-[#38bdf8] opacity-30 animate-ping" />
            <span className="relative w-3 h-3 rounded-full bg-[#38bdf8] block" style={{ boxShadow: '0 0 8px #38bdf8' }} />
          </div>
        ) : (
          <span className="w-3 h-3 rounded-full bg-[#1e3a5f] block" />
        )}
      </div>

      {/* Status Message */}
      <span className="font-mono text-[11.5px] text-[#93c5fd] flex-1 min-w-0 truncate">
        {displayed || <span className="text-[#1e3a5f]">— waiting for pipeline —</span>}
        {isRunning && <span className="text-[#38bdf8]">{dots}</span>}
      </span>

      {/* Right Side: Status Label */}
      {currentProject && (
        <div className="flex items-center gap-2 flex-shrink-0">
          <StatusDot status={currentProject.status} />
          <span className="font-mono text-[10px] text-[#4a6fa5] uppercase">
            {currentProject.status}
          </span>
        </div>
      )}
    </div>
  )
}

function StatusDot({ status }) {
  const color = {
    idle: '#1e3a5f', discovering: '#38bdf8', analyzing: '#38bdf8',
    approved: '#f59e0b', drafting: '#38bdf8', refining: '#38bdf8',
    humanizing: '#38bdf8', reviewing: '#38bdf8',
    complete: '#10b981', error: '#ef4444',
  }[status] || '#1e3a5f'
  return <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
}
