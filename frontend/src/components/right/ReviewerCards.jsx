import React, { useState, useEffect } from 'react'
import { useApp } from '../../context/AppContext'
import { useProject } from '../../hooks/useProject'
import { FaMicroscope, FaBook, FaPen } from 'react-icons/fa'

const REVIEWERS = [
  {
    id: 'reviewer-a',
    name: 'Reviewer A',
    persona: 'Methodology Expert',
    icon: <FaMicroscope />,
    color: '#2563eb',
    scoreKeys: ['methodology', 'experimental_design', 'rigor'],
    accentClass: 'border-blue-700/30',
    glowClass: 'rgba(37,99,235,0.15)',
  },
  {
    id: 'reviewer-b',
    name: 'Reviewer B',
    persona: 'Literature Expert',
    icon: <FaBook />,
    color: '#0ea5e9',
    scoreKeys: ['novelty', 'related_work', 'contribution'],
    accentClass: 'border-sky-700/30',
    glowClass: 'rgba(14,165,233,0.15)',
  },
  {
    id: 'reviewer-c',
    name: 'Reviewer C',
    persona: 'Presentation Expert',
    icon: <FaPen />,
    color: '#38bdf8',
    scoreKeys: ['clarity', 'writing_quality', 'presentation'],
    accentClass: 'border-cyan-700/30',
    glowClass: 'rgba(56,189,248,0.15)',
  },
]

export default function ReviewerCards() {
  const { currentProject } = useApp()
  const { fetchReview } = useProject()
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(false)

  const shouldShow = currentProject && ['reviewing', 'complete'].includes(currentProject.status)

  useEffect(() => {
    if (!shouldShow) return
    setLoading(true)
    fetchReview(currentProject.id)
      .then(r => setReport(r))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [currentProject?.id, currentProject?.status])

  if (!shouldShow && !report) return null

  const scores = report?.reviewer_scores || {}
  const risk = report?.rejection_risk
  const content = report?.content || {}

  return (
    <div id="reviewer-cards-section" className="px-5 mb-8">
      {/* Section Header */}
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="font-display text-[20px] uppercase text-[#e0f2fe] font-bold">Peer Review Simulation</h3>
          {risk !== undefined && risk !== null && (
            <RejectionRisk risk={risk} />
          )}
        </div>
        <span className="font-mono text-[10px] text-[#1e3a5f]">3 reviewers</span>
      </div>

      {loading && <ShimmerCards />}

      {!loading && (
        <div className="grid grid-cols-1 gap-3">
          {REVIEWERS.map((r) => (
            <ReviewerCard
              key={r.id}
              reviewer={r}
              scores={scores}
              content={content}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function ReviewerCard({ reviewer, scores, content }) {
  const [expanded, setExpanded] = useState(false)
  const cardScores = reviewer.scoreKeys
    .map(k => ({ key: k, val: scores[k] }))
    .filter(s => s.val !== undefined)

  const feedbackKey = reviewer.id.replace('-', '_')
  const feedback = content[feedbackKey] || content[reviewer.name] || Object.values(content)[REVIEWERS.indexOf(reviewer)] || ''

  return (
    <div
      id={reviewer.id}
      className="clay-card p-4 cursor-pointer select-none animate-fade-in-up"
      style={{ boxShadow: `0 8px 32px ${reviewer.glowClass}, inset 0 1px 0 rgba(56,189,248,0.06)` }}
      onClick={() => setExpanded(e => !e)}
    >
      {/* Card Header */}
      <div className="flex items-center gap-3 mb-3">
        <div
          className="w-9 h-9 rounded-xl flex items-center justify-center text-lg flex-shrink-0"
          style={{
            background: `linear-gradient(135deg, ${reviewer.color}22, ${reviewer.color}10)`,
            border: `1px solid ${reviewer.color}33`,
          }}
        >
          {reviewer.icon}
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-[13px] font-semibold text-[#e0f2fe]">{reviewer.name}</div>
          <div className="font-mono text-[10px] text-[#4a6fa5]">{reviewer.persona}</div>
        </div>
        <div className="flex items-center gap-1.5">
          {cardScores.length > 0 && (
            <span
              className="font-mono text-[13px] font-semibold"
              style={{ color: reviewer.color }}
            >
              {(cardScores.reduce((a, s) => a + s.val, 0) / cardScores.length).toFixed(1)}
            </span>
          )}
          <span className="text-[#1e3a5f] text-xs">
            {cardScores.length > 0 && <span style={{ color: '#4a6fa5' }}>/10</span>}
          </span>
          <span
            className="text-[10px] text-[#4a6fa5] ml-1 transition-transform duration-200"
            style={{ transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)' }}
          >
            ▾
          </span>
        </div>
      </div>

      {/* Score Bars */}
      {cardScores.length > 0 && (
        <div className="flex flex-col gap-2 mb-2">
          {cardScores.map(s => (
            <ScoreBar key={s.key} label={s.key.replace(/_/g, ' ')} value={s.val} color={reviewer.color} />
          ))}
        </div>
      )}

      {/* Expandable Feedback */}
      {expanded && feedback && (
        <div
          className="mt-3 pt-3 border-t border-[#0c2040] text-[12px] text-[#93c5fd] leading-relaxed animate-fade-in-up"
          style={{ whiteSpace: 'pre-line' }}
        >
          {typeof feedback === 'string' ? feedback.slice(0, 400) + (feedback.length > 400 ? '…' : '') : JSON.stringify(feedback)}
        </div>
      )}

      {!expanded && cardScores.length === 0 && (
        <div className="text-[12px] text-[#1e3a5f] italic">Awaiting review…</div>
      )}
    </div>
  )
}

function ScoreBar({ label, value, color }) {
  const pct = Math.min((value / 10) * 100, 100)
  const fillColor = value >= 7 ? '#10b981' : value >= 5 ? '#f59e0b' : '#ef4444'

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="font-mono text-[10px] text-[#4a6fa5]">{label}</span>
        <span className="font-mono text-[11px]" style={{ color }}>{Number(value).toFixed(1)}</span>
      </div>
      <div className="score-bar-track">
        <div
          className="score-bar-fill"
          style={{ width: `${pct}%`, background: fillColor }}
        />
      </div>
    </div>
  )
}

function RejectionRisk({ risk }) {
  const pct = (risk * 100).toFixed(0)
  const { label, cls } = risk < 0.3
    ? { label: 'Low Rejection Risk', cls: 'text-[#34d399]' }
    : risk < 0.6
    ? { label: 'Medium Rejection Risk', cls: 'text-[#fbbf24]' }
    : { label: 'High Rejection Risk', cls: 'text-red-400' }

  return (
    <span className={`font-mono text-[10px] ${cls}`}>
      {label} — {pct}%
    </span>
  )
}

function ShimmerCards() {
  return (
    <div className="flex flex-col gap-3">
      {[0, 1, 2].map(i => (
        <div key={i} className="clay-card p-4 h-24">
          <div className="shimmer-bar rounded mb-3" />
          <div className="shimmer-bar rounded" style={{ width: '60%' }} />
        </div>
      ))}
    </div>
  )
}
