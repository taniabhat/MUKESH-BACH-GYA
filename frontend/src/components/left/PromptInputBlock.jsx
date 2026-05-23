import React, { useState } from 'react'
import { useApp } from '../../context/AppContext'
import { useProject } from '../../hooks/useProject'

export default function PromptInputBlock({ onProjectCreated }) {
  const { currentProject, isRunning, pushChat } = useApp()
  const { createProject, discover, analyze, approve, draft, refine, humanize, review } = useProject()

  const [topic, setTopic]       = useState('')
  const [goals, setGoals]       = useState('')
  const [guidelines, setGuide]  = useState('')
  const [guideOpen, setGuideOpen] = useState(false)
  const [loading, setLoading]   = useState(false)
  const [draftVenue, setDraftVenue]    = useState('NeurIPS')
  const [draftSections, setDraftSections] = useState(
    'abstract,introduction,related_work,methodology,experiments,results,discussion,conclusion'
  )
  const [showDraftConf, setShowDraftConf] = useState(false)
  const [approveNotes, setApproveNotes] = useState('')
  const [showApprove, setShowApprove] = useState(false)

  const status = currentProject?.status

  const handleRun = async () => {
    if (!topic.trim()) return
    setLoading(true)
    try {
      const title = topic.trim().slice(0, 80)
      await createProject({ title, topic, goals, guidelines })
      onProjectCreated?.()
    } catch (e) {
      // error handled in hook
    } finally {
      setLoading(false)
    }
  }

  const handleStageAction = async (action) => {
    setLoading(true)
    try { await action() } finally { setLoading(false) }
  }

  const handleDraftSubmit = async () => {
    setShowDraftConf(false)
    const sections = draftSections.split(',').map(s => s.trim()).filter(Boolean)
    setLoading(true)
    try {
      await draft({ target_venue: draftVenue, sections })
    } finally {
      setLoading(false)
    }
  }

  const handleApprove = async () => {
    setShowApprove(false)
    setLoading(true)
    try { await approve(approveNotes) } finally { setLoading(false) }
  }

  const canCreate = !currentProject
  const isIdle    = status === 'idle' || !status

  return (
    <div className="px-4 py-4 flex flex-col gap-3 flex-shrink-0">
      {/* Input Container */}
      <div
        className="rounded-2xl border overflow-hidden"
        style={{
          background: 'var(--input-block-bg)',
          boxShadow: 'var(--input-block-shadow)',
          borderColor: 'var(--border2)',
        }}
      >
        {/* Topic Field */}
        <div className="px-4 pt-4 pb-2">
          <label className="block font-mono text-[10px] text-[#4a6fa5] uppercase tracking-widest mb-2">
            📄 Topic / Paper Draft
          </label>
          <textarea
            id="prompt-topic"
            value={topic}
            onChange={e => setTopic(e.target.value)}
            placeholder="Describe your research topic, paper draft, or core idea…"
            rows={3}
            className="field-input text-[13.5px]"
            disabled={!!currentProject && !canCreate}
          />
        </div>

        {/* Divider */}
        <div className="h-px bg-[#0c2040] mx-4" />

        {/* Research Goals Field */}
        <div className="px-4 pt-3 pb-2">
          <label className="block font-mono text-[10px] text-[#4a6fa5] uppercase tracking-widest mb-2">
            🎯 Research Goals
          </label>
          <textarea
            id="prompt-goals"
            value={goals}
            onChange={e => setGoals(e.target.value)}
            placeholder="What are your research objectives, target venue, expected contributions?"
            rows={2}
            className="field-input text-[13px]"
            disabled={!!currentProject && !canCreate}
          />
        </div>

        {/* Guidelines Collapsible */}
        <div className="px-4 pt-1 pb-3">
          <button
            onClick={() => setGuideOpen(o => !o)}
            className="flex items-center gap-2 font-mono text-[10px] text-[#4a6fa5] uppercase tracking-widest hover:text-[#38bdf8] transition-colors mb-2"
          >
            <span className={`transition-transform duration-200 ${guideOpen ? 'rotate-90' : ''}`}>▶</span>
            ⚙️ Custom Guidelines
          </button>
          {guideOpen && (
            <textarea
              id="prompt-guidelines"
              value={guidelines}
              onChange={e => setGuide(e.target.value)}
              placeholder="Any specific instructions: tone, style, sections to emphasize, constraints…"
              rows={2}
              className="field-input text-[13px] animate-fade-in-up"
              disabled={!!currentProject && !canCreate}
            />
          )}
        </div>
      </div>

      {/* ── Action Area ─────────────────────────────────── */}
      {canCreate ? (
        /* New Project Run Button */
        <button
          id="run-pipeline-btn"
          className="run-btn"
          onClick={handleRun}
          disabled={!topic.trim() || loading}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-3">
              <Spinner /> Initializing Agent…
            </span>
          ) : (
            <span className="flex items-center justify-center gap-2">
              <span className="text-[18px]">⚡</span> Initialize Agent
            </span>
          )}
        </button>
      ) : (
        /* Pipeline Stage Buttons */
        <div className="flex flex-col gap-2">
          {isIdle && renderStageButtons(status, currentProject, {
            discover: () => handleStageAction(discover),
            analyze:  () => handleStageAction(analyze),
            draft:    () => setShowDraftConf(true),
            approve:  () => setShowApprove(true),
            refine:   () => handleStageAction(refine),
            humanize: () => handleStageAction(humanize),
            review:   () => handleStageAction(review),
          }, loading, currentProject)}

          {isRunning && (
            <div className="flex items-center gap-3 px-4 py-3 rounded-xl border" style={{ borderColor: 'var(--border2)', background: 'var(--panel)' }}>
              <Spinner className="text-[#38bdf8]" />
              <span className="text-[13px] text-[#38bdf8]">Pipeline running…</span>
            </div>
          )}

          {status === 'complete' && (
            <div className="px-4 py-3 rounded-xl border border-[#10b981]/30 bg-[#10b981]/5 text-[13px] text-[#34d399]">
              ✓ Pipeline complete — see Export in workspace
            </div>
          )}

          {status === 'error' && (
            <button
              id="retry-btn"
              onClick={() => handleStageAction(discover)}
              className="run-btn opacity-80"
            >
              ↺ Retry Pipeline
            </button>
          )}
        </div>
      )}

      {/* ── Draft Config Inline ──────────────────────────── */}
      {showDraftConf && (
        <div className="rounded-xl border border-[#1a3a6e] bg-[#0c1f3a]/80 p-4 animate-fade-in-up">
          <div className="font-mono text-[10px] text-[#4a6fa5] uppercase tracking-widest mb-3">
            Configure Draft
          </div>
          <div className="mb-3">
            <label className="block text-[11px] text-[#4a6fa5] mb-1">Target Venue</label>
            <input
              id="draft-venue"
              className="field-input"
              value={draftVenue}
              onChange={e => setDraftVenue(e.target.value)}
            />
          </div>
          <div className="mb-4">
            <label className="block text-[11px] text-[#4a6fa5] mb-1">Sections (comma-separated)</label>
            <input
              id="draft-sections"
              className="field-input"
              value={draftSections}
              onChange={e => setDraftSections(e.target.value)}
            />
          </div>
          <div className="flex gap-2">
            <button onClick={handleDraftSubmit} className="run-btn flex-1 py-2.5 text-[13px]">
              ✍️ Generate Draft
            </button>
            <button
              onClick={() => setShowDraftConf(false)}
              className="px-4 py-2.5 rounded-xl border border-[#1a3a6e] text-[#4a6fa5] hover:text-[#e0f2fe] text-[13px] transition-all"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* ── Approve Inline ───────────────────────────────── */}
      {showApprove && (
        <div className="rounded-xl border border-[#f59e0b]/30 bg-[#f59e0b]/5 p-4 animate-fade-in-up">
          <div className="font-mono text-[10px] text-[#f59e0b] uppercase tracking-widest mb-3">
            ✅ Approve Gap Report
          </div>
          <textarea
            id="approve-notes"
            className="field-input mb-3"
            rows={2}
            placeholder="Optional notes or edits (leave empty to approve as-is)…"
            value={approveNotes}
            onChange={e => setApproveNotes(e.target.value)}
          />
          <div className="flex gap-2">
            <button
              id="approve-submit-btn"
              onClick={handleApprove}
              className="flex-1 py-2.5 rounded-xl font-medium text-[13px] text-white transition-all"
              style={{ background: 'linear-gradient(135deg, #d97706, #b45309)' }}
            >
              ✓ Approve & Continue
            </button>
            <button
              onClick={() => setShowApprove(false)}
              className="px-4 py-2.5 rounded-xl border border-[#1a3a6e] text-[#4a6fa5] hover:text-[#e0f2fe] text-[13px] transition-all"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function renderStageButtons(status, project, actions, loading, currentProject) {
  if (!project) return null
  const btn = (id, icon, label, onClick, colorClass = '') => (
    <button
      id={id}
      key={id}
      onClick={onClick}
      disabled={loading}
      className={`w-full flex items-center gap-2.5 px-4 py-3 rounded-xl border border-[#1a3a6e] text-[13px] font-medium transition-all disabled:opacity-50
        ${colorClass || 'text-[#93c5fd] hover:text-[#e0f2fe] hover:border-[#2563eb] hover:bg-[#2563eb]/10'}`}
    >
      <span className="text-base">{icon}</span> {label}
    </button>
  )

  const buttons = []

  if (!['analyzing','approved','drafting','refining','humanizing','reviewing','complete'].some(s => s === status)) {
    buttons.push(btn('btn-discover', '🔍', 'Start Discovery', actions.discover))
  }
  if (['idle'].includes(status) && currentProject?.status !== 'complete') {
    buttons.push(btn('btn-analyze', '📄', 'Analyse Papers', actions.analyze))
  }
  if (status === 'idle' || status === 'approved') {
    buttons.push(btn('btn-approve', '✅', 'Review & Approve Gaps', actions.approve,
      'text-[#fbbf24] hover:text-[#fef3c7] hover:border-[#f59e0b] hover:bg-[#f59e0b]/10 border-[#1a3a6e]'))
    buttons.push(btn('btn-draft', '✍️', 'Configure & Draft', actions.draft))
  }
  if (['idle'].includes(status)) {
    buttons.push(btn('btn-refine', '✨', 'Refine Draft', actions.refine))
    buttons.push(btn('btn-humanize', '🧬', 'Humanize', actions.humanize))
    buttons.push(btn('btn-review', '👁', 'Peer Review', actions.review))
  }

  return buttons
}

function Spinner({ className = '' }) {
  return (
    <span
      className={`inline-block w-4 h-4 rounded-full border-2 border-transparent border-t-current animate-spin ${className}`}
    />
  )
}
