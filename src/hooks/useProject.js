import { useCallback, useRef } from 'react'
import { useApp } from '../context/AppContext'

const BASE = '/api/v1'
const WS_BASE = `ws://${window.location.host}/ws`

const RUNNING_STATUSES = ['discovering', 'analyzing', 'drafting', 'refining', 'humanizing', 'reviewing']

export function useProject() {
  const {
    projects, setProjects,
    currentProject, setCurrentProject,
    pushLog, pushChat,
    setWsStatus,
    setHumanReview,
    setIsRunning,
    setAgentStatus,
  } = useApp()

  const wsRef = useRef(null)
  const pollRef = useRef(null)

  // ── API Helper ─────────────────────────────────────────
  const api = useCallback(async (method, path, body) => {
    const opts = { method, headers: { 'Content-Type': 'application/json' } }
    if (body) opts.body = JSON.stringify(body)
    const r = await fetch(BASE + path, opts)
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }))
      let msg = err.detail || r.statusText
      if (Array.isArray(msg)) msg = msg.map(m => `${m.loc.join('.')}: ${m.msg}`).join('\n')
      throw new Error(msg)
    }
    const ct = r.headers.get('content-type') || ''
    return ct.includes('application/json') ? r.json() : r.blob()
  }, [])

  // ── Load All Projects ──────────────────────────────────
  const loadProjects = useCallback(async () => {
    try {
      const data = await api('GET', '/projects')
      setProjects(data)
    } catch (e) {
      console.warn('Backend not reachable:', e.message)
    }
  }, [api, setProjects])

  // ── Select Project ─────────────────────────────────────
  const selectProject = useCallback(async (id) => {
    try {
      const proj = await api('GET', `/projects/${id}`)
      setCurrentProject(proj)
      setIsRunning(RUNNING_STATUSES.includes(proj.status))
      pushChat('system', `Switched to project: ${proj.title}`, { status: proj.status })
      connectWS(id)
      startPolling(id)
    } catch (e) {
      pushLog('select', 'error', e.message)
    }
  }, [api, setCurrentProject]) // eslint-disable-line

  // ── Create Project ─────────────────────────────────────
  const createProject = useCallback(async ({ title, topic, goals, guidelines }) => {
    const research_idea = [topic, goals, guidelines].filter(Boolean).join('\n\n')
    try {
      const proj = await api('POST', '/projects', { title, research_idea })
      setProjects(prev => [proj, ...prev])
      setCurrentProject(proj)
      pushChat('user', `🚀 Initialized pipeline for: **${title}**\n\n${research_idea}`)
      pushChat('system', `Project created. Pipeline is ready. Status: ${proj.status}`, { status: proj.status })
      pushLog('project.create', 'success', proj.id)
      connectWS(proj.id)
      startPolling(proj.id)
      return proj
    } catch (e) {
      pushLog('project.create', 'error', e.message)
      throw e
    }
  }, [api, setProjects, setCurrentProject, pushChat, pushLog]) // eslint-disable-line

  // ── Delete Project ─────────────────────────────────────
  const deleteProject = useCallback(async (id) => {
    try {
      await api('DELETE', `/projects/${id}`)
      setProjects(prev => prev.filter(p => p.id !== id))
      setCurrentProject(null)
      disconnectWS()
      clearPolling()
    } catch (e) {
      pushLog('project.delete', 'error', e.message)
    }
  }, [api, setProjects, setCurrentProject, pushLog]) // eslint-disable-line

  // ── Refresh Project ────────────────────────────────────
  const refreshProject = useCallback(async (id) => {
    if (!id) return
    try {
      const fresh = await api('GET', `/projects/${id}`)
      setCurrentProject(fresh)
      setIsRunning(RUNNING_STATUSES.includes(fresh.status))
      setAgentStatus(statusToAgentMsg(fresh.status))
      setProjects(prev => prev.map(p => p.id === id ? fresh : p))
      return fresh
    } catch (_) {}
  }, [api, setCurrentProject, setIsRunning, setAgentStatus, setProjects])

  // ── Pipeline Triggers ──────────────────────────────────
  const triggerPipeline = useCallback(async (action, body, label) => {
    if (!currentProject) return
    try {
      pushLog(label, 'queued')
      pushChat('agent', `⚙️ ${label} stage initiated...`)
      const task = await api('POST', `/projects/${currentProject.id}/${action}`, body)
      pushLog(label, 'task_id: ' + (task?.task_id || '?'))
      await refreshProject(currentProject.id)
    } catch (e) {
      pushLog(label, 'error', e.message)
      pushChat('system', `❌ ${label} failed: ${e.message}`, { isError: true })
    }
  }, [api, currentProject, pushLog, pushChat, refreshProject])

  const discover  = useCallback(() => triggerPipeline('discover', undefined, 'discovery'), [triggerPipeline])
  const analyze   = useCallback(() => triggerPipeline('analyze', undefined, 'analysis'), [triggerPipeline])
  const refine    = useCallback(() => triggerPipeline('refine', undefined, 'refinement'), [triggerPipeline])
  const humanize  = useCallback(() => triggerPipeline('humanize', undefined, 'humanization'), [triggerPipeline])
  const review    = useCallback(() => triggerPipeline('review', undefined, 'peer_review'), [triggerPipeline])
  const genCode   = useCallback(() => triggerPipeline('generate-code', undefined, 'code_gen'), [triggerPipeline])
  const genDiag   = useCallback(() => triggerPipeline('generate-diagrams', undefined, 'diagrams'), [triggerPipeline])
  const exportPdf = useCallback((fmt) => triggerPipeline(`export?fmt=${fmt}`, undefined, `export_${fmt}`), [triggerPipeline])

  const draft = useCallback(async (plan) => {
    if (!currentProject) return
    try {
      pushLog('draft', 'queued')
      pushChat('agent', `✍️ Generating paper draft for **${plan.target_venue}**...`)
      const task = await api('POST', `/projects/${currentProject.id}/draft`, { plan })
      pushLog('draft', 'task_id: ' + (task?.task_id || '?'))
      await refreshProject(currentProject.id)
    } catch (e) {
      pushLog('draft', 'error', e.message)
    }
  }, [api, currentProject, pushLog, pushChat, refreshProject])

  const approve = useCallback(async (edits = '') => {
    if (!currentProject) return
    try {
      await api('POST', `/projects/${currentProject.id}/approve`, {
        user_edits: edits ? { notes: edits } : {},
        approved_at: new Date().toISOString(),
      })
      pushLog('approval', 'approved')
      pushChat('user', `✅ Gap report approved. ${edits ? 'Notes: ' + edits : ''}`)
      await refreshProject(currentProject.id)
    } catch (e) {
      pushLog('approval', 'error', e.message)
      throw e
    }
  }, [api, currentProject, pushLog, pushChat, refreshProject])

  // ── WebSocket ──────────────────────────────────────────
  const connectWS = useCallback((id) => {
    disconnectWS()
    try {
      wsRef.current = new WebSocket(`${WS_BASE}/${id}`)
      wsRef.current.onopen = () => {
        setWsStatus('connected')
        pushLog('websocket', 'connected')
      }
      wsRef.current.onclose = () => setWsStatus('idle')
      wsRef.current.onerror = () => setWsStatus('error')
      wsRef.current.onmessage = async (evt) => {
        try {
          const msg = JSON.parse(evt.data)
          const stage = msg.stage || msg.event || '?'
          const status = msg.status || msg.state || ''
          const detail = msg.details ? JSON.stringify(msg.details).slice(0, 80) : ''
          pushLog(stage, status, detail)
          setAgentStatus(statusToAgentMsg(status, stage))

          // Human-in-the-loop checkpoint
          if (msg.type === 'human_review_required' || msg.requires_human_input) {
            setHumanReview({
              message: msg.message || `${stage} requires your validation.`,
              type: msg.review_type || 'general',
              reviewer: msg.reviewer || stage,
            })
          }

          if (['complete', 'idle', 'error'].includes(status) ||
              ['complete', 'idle', 'error'].includes(msg.project_status)) {
            await refreshProject(id)
          }
        } catch (_) {}
      }
    } catch (e) {
      console.warn('WS failed:', e)
    }
  }, [pushLog, setWsStatus, setHumanReview, setAgentStatus, refreshProject])

  const disconnectWS = useCallback(() => {
    if (wsRef.current) {
      try { wsRef.current.close() } catch (_) {}
      wsRef.current = null
    }
  }, [])

  // ── Polling ────────────────────────────────────────────
  const startPolling = useCallback((id) => {
    clearPolling()
    pollRef.current = setInterval(() => refreshProject(id), 4000)
  }, [refreshProject])

  const clearPolling = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }, [])

  // ── Fetch Helpers ──────────────────────────────────────
  const fetchPapers = useCallback(async (id) => {
    return api('GET', `/projects/${id}/papers?size=50`)
  }, [api])

  const fetchGaps = useCallback(async (id) => {
    return api('GET', `/projects/${id}/gaps`)
  }, [api])

  const fetchDraft = useCallback(async (id) => {
    return api('GET', `/projects/${id}/draft`)
  }, [api])

  const fetchReview = useCallback(async (id) => {
    return api('GET', `/projects/${id}/review-report`)
  }, [api])

  const fetchCitations = useCallback(async (id) => {
    return api('GET', `/projects/${id}/citations`)
  }, [api])

  const fetchAssets = useCallback(async (id) => {
    return api('GET', `/projects/${id}/assets`)
  }, [api])

  const downloadExport = useCallback(async (fmt) => {
    if (!currentProject) return
    const blob = await api('GET', `/projects/${currentProject.id}/export/${fmt}`)
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = `paper.${fmt}`; a.click()
    URL.revokeObjectURL(url)
  }, [api, currentProject])

  return {
    api,
    loadProjects, selectProject, createProject, deleteProject, refreshProject,
    discover, analyze, draft, approve, refine, humanize, review, genCode, genDiag, exportPdf,
    downloadExport,
    connectWS, disconnectWS, startPolling, clearPolling,
    fetchPapers, fetchGaps, fetchDraft, fetchReview, fetchCitations, fetchAssets,
    RUNNING_STATUSES,
  }
}

// ── Status → Agent Message ─────────────────────────────
function statusToAgentMsg(status, stage = '') {
  const msgs = {
    discovering: '🔍 Searching Semantic Scholar, CrossRef, OpenAlex...',
    analyzing:   '📄 Parsing PDFs, extracting figures and embedding vectors...',
    approved:    '✅ Gap report approved. Ready to draft.',
    drafting:    '✍️ Generating full paper draft...',
    refining:    '✨ Refining prose and logical flow...',
    humanizing:  '🧬 Applying humanization pass...',
    reviewing:   '👁 Peer review agents evaluating manuscript...',
    complete:    '🎉 Pipeline complete. Paper ready for export.',
    error:       '❌ An error occurred. Check logs for details.',
    idle:        '',
  }
  return msgs[status] || (stage ? `⚙️ Running: ${stage}...` : '')
}
