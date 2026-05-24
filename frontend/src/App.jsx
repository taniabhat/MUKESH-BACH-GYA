import React, { useEffect, useState } from 'react'
import { AppProvider } from './context/AppContext'
import { useProject } from './hooks/useProject'
import { useResize } from './hooks/useResize'
import LeftPanel from './components/left/LeftPanel'
import RightPanel from './components/right/RightPanel'
import GlobalSidebar from './components/shared/GlobalSidebar'

function AppInner() {
  const { loadProjects } = useProject()
  const { leftPct, onMouseDown } = useResize(42, 28, 60)
  const { leftPct: sidebarPct, onMouseDown: onSidebarResize, isDragging: isSidebarDragging } = useResize(18, 12, 35)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  useEffect(() => { loadProjects() }, [loadProjects])

  return (
    <div
      id="app"
      className="flex h-screen w-screen overflow-hidden"
      style={{ background: 'var(--bg)' }}
    >
      {/* Global Sidebar (collapsible) */}
      <GlobalSidebar
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(o => !o)}
        onNewProject={() => setSidebarOpen(false)}
        widthPct={sidebarPct}
        isDragging={isSidebarDragging}
      />
      {sidebarOpen && (
        <div
          className="resize-handle"
          onMouseDown={onSidebarResize}
          title="Drag to resize sidebar"
        />
      )}

      {/* Left Panel (Driver / Chat) */}
      <LeftPanel
        onSidebarToggle={() => setSidebarOpen(o => !o)}
        onProjectCreated={() => {}}
        style={{ width: `${leftPct}%` }}
      />

      {/* Resize Handle */}
      <div
        id="resize-handle"
        className="resize-handle"
        onMouseDown={onMouseDown}
        title="Drag to resize"
      />

      {/* Right Panel (Execution / Workspace) */}
      <RightPanel style={{ flex: 1 }} />
    </div>
  )
}

export default function App() {
  return (
    <AppProvider>
      <AppInner />
    </AppProvider>
  )
}
