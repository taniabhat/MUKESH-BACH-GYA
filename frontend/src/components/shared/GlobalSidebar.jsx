import React, { useState } from 'react'
import { useApp } from '../../context/AppContext'
import { useProject } from '../../hooks/useProject'
import StatusPill from './StatusPill'
import { FaAngleDoubleLeft } from 'react-icons/fa'

export default function GlobalSidebar({ isOpen, onToggle, onNewProject, widthPct, isDragging }) {
  const { projects, currentProject } = useApp()
  const { selectProject, deleteProject } = useProject()
  const { wsStatus } = useApp()

  const statusColor = (s) => ({
    idle: '#4a6fa5', discovering: '#38bdf8', analyzing: '#38bdf8',
    approved: '#f59e0b', drafting: '#38bdf8', refining: '#38bdf8',
    humanizing: '#38bdf8', reviewing: '#38bdf8',
    complete: '#10b981', error: '#ef4444',
  })[s] || '#4a6fa5'

  const handleDelete = async (e, id) => {
    e.stopPropagation()
    if (!window.confirm('Delete this project?')) return
    await deleteProject(id)
  }

  return (
    <aside
      id="global-sidebar"
      className="flex flex-col border-r border-[#0c2040] overflow-hidden flex-shrink-0"
      style={{
        width: isOpen ? `${widthPct}%` : 0,
        opacity: isOpen ? 1 : 0,
        background: 'var(--sidebar-bg)',
        backdropFilter: 'blur(20px)',
        borderRight: `1px solid var(--sidebar-border)`,
        transition: isDragging ? 'none' : 'width 0.3s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.3s ease'
      }}
    >
      {/* Sidebar Header */}
      <div className="flex items-center justify-between px-4 py-4 border-b border-[#0c2040]">
        <span 
          className="text-[20px] text-[#e0f2fe] uppercase font-bold tracking-widest"
          style={{ fontFamily: "'Playfair Display', serif" }}
        >
          Projects
        </span>
        <button
          id="sidebar-close-btn"
          onClick={onToggle}
          className="text-[#4a6fa5] hover:text-[#e0f2fe] transition-colors text-lg"
        >
          <FaAngleDoubleLeft />
        </button>
      </div>

      {/* New Project Button */}
      <button
        id="new-project-btn"
        onClick={onNewProject}
        className="mx-3 mt-3 flex items-center justify-center gap-2 px-3 py-3 rounded-xl border border-[#1a3a6e] text-[#38bdf8] text-[16px] font-bold hover:bg-[#2563eb]/10 hover:border-[#2563eb] transition-all"
      >
        <span className="text-[22px] leading-none">＋</span> New Project
      </button>

      {/* Project List */}
      <div className="flex-1 overflow-y-auto py-3 px-2">
        {!projects.length ? (
          <div className="text-center py-10 text-[#1e3a5f] text-[12px] font-mono">
            No projects yet.<br />Create one to start.
          </div>
        ) : (
          projects.map(p => (
            <div
              key={p.id}
              id={`project-item-${p.id}`}
              onClick={() => selectProject(p.id)}
              className={`group flex items-center gap-2.5 px-3 py-2.5 rounded-xl cursor-pointer mb-1 border transition-all ${
                currentProject?.id === p.id
                  ? 'bg-[#2563eb]/10 border-[#1a3a6e]'
                  : 'border-transparent hover:bg-white/[0.03] hover:border-[#0c2040]'
              }`}
            >
              <div
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ background: statusColor(p.status) }}
              />
              <div className="flex-1 min-w-0">
                <div className="text-[13px] text-[#e0f2fe] truncate">{p.title}</div>
                <div className="text-[10px] font-mono text-[#4a6fa5]">{p.status}</div>
              </div>
              <button
                onClick={(e) => handleDelete(e, p.id)}
                className="opacity-0 group-hover:opacity-100 text-[#1e3a5f] hover:text-red-400 transition-all text-xs px-1"
              >
                ✕
              </button>
            </div>
          ))
        )}
      </div>

      {/* WS Status Footer */}
      <div className="px-4 py-3 border-t border-[#0c2040] flex items-center gap-2">
        <div
          className={`w-2 h-2 rounded-full flex-shrink-0 ${
            wsStatus === 'connected' ? 'bg-[#10b981]' :
            wsStatus === 'error'     ? 'bg-[#ef4444]' :
            'bg-[#1e3a5f]'
          }`}
          style={wsStatus === 'connected' ? { boxShadow: '0 0 6px #10b981' } : {}}
        />
        <span className="text-[10px] font-mono text-[#1e3a5f]">
          {wsStatus === 'connected' ? 'WebSocket live' :
           wsStatus === 'error'     ? 'WS error' : 'disconnected'}
        </span>
      </div>
    </aside>
  )
}
