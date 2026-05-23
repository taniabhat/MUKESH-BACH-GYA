import React, { createContext, useContext, useState, useCallback } from 'react'

const AppContext = createContext(null)

export function AppProvider({ children }) {
  const [projects, setProjects] = useState([])
  const [currentProject, setCurrentProject] = useState(null)
  const [logLines, setLogLines] = useState([])
  const [wsStatus, setWsStatus] = useState('idle') // idle | connected | error
  const [humanReview, setHumanReview] = useState(null) // null or { message, type, reviewer }
  const [chatHistory, setChatHistory] = useState([])
  const [isRunning, setIsRunning] = useState(false)
  const [agentStatus, setAgentStatus] = useState('')

  const pushLog = useCallback((event, status = '', extra = '') => {
    const ts = new Date().toLocaleTimeString('en-US', {
      hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
    })
    setLogLines(prev => {
      const next = [...prev, { ts, event, status, extra }]
      return next.length > 200 ? next.slice(-200) : next
    })
  }, [])

  const pushChat = useCallback((role, content, meta = {}) => {
    setChatHistory(prev => [...prev, {
      id: Date.now() + Math.random(),
      role, // 'user' | 'system' | 'agent'
      content,
      ts: new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' }),
      ...meta,
    }])
  }, [])

  const clearHumanReview = useCallback(() => setHumanReview(null), [])

  return (
    <AppContext.Provider value={{
      projects, setProjects,
      currentProject, setCurrentProject,
      logLines, setLogLines, pushLog,
      wsStatus, setWsStatus,
      humanReview, setHumanReview, clearHumanReview,
      chatHistory, setChatHistory, pushChat,
      isRunning, setIsRunning,
      agentStatus, setAgentStatus,
    }}>
      {children}
    </AppContext.Provider>
  )
}

export function useApp() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used within AppProvider')
  return ctx
}
