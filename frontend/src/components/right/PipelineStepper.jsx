import React from 'react'
import { useApp } from '../../context/AppContext'
import { FaSearch, FaFileAlt, FaMicroscope, FaCheckCircle, FaPen, FaStar, FaDna, FaEye, FaCog, FaFileExport } from 'react-icons/fa'

const STAGES = [
  { key: 'discovery',    label: 'Discovery',   icon: <FaSearch /> },
  { key: 'analysis',     label: 'Analysis',    icon: <FaFileAlt /> },
  { key: 'gap_analysis', label: 'Gap Analysis', icon: <FaMicroscope /> },
  { key: 'approval',     label: 'Approval',    icon: <FaCheckCircle /> },
  { key: 'draft',        label: 'Draft',       icon: <FaPen /> },
  { key: 'refinement',   label: 'Refinement',  icon: <FaStar /> },
  { key: 'humanization', label: 'Humanize',    icon: <FaDna /> },
  { key: 'review',       label: 'Review',      icon: <FaEye /> },
  { key: 'generation',   label: 'Generate',    icon: <FaCog /> },
  { key: 'export',       label: 'Export',      icon: <FaFileExport /> },
]


const STATUS_TO_STAGE = {
  idle: null, discovering: 'discovery', analyzing: 'analysis',
  approved: 'approval', drafting: 'draft', refining: 'refinement',
  humanizing: 'humanization', reviewing: 'review', complete: 'export', error: null,
}

const STAGE_ORDER = STAGES.map(s => s.key)

function doneStages(status) {
  const active = STATUS_TO_STAGE[status]
  if (status === 'complete') return [...STAGE_ORDER]
  const idx = STAGE_ORDER.indexOf(active)
  if (idx < 0) return []
  return STAGE_ORDER.slice(0, idx)
}

export default function PipelineStepper() {
  const { currentProject } = useApp()
  if (!currentProject) return null

  const status  = currentProject.status
  const active  = STATUS_TO_STAGE[status]
  const done    = doneStages(status)
  const isError = status === 'error'

  return (
    <div
      id="pipeline-stepper"
      className="px-5 py-4 border-b border-[#0c2040] flex-shrink-0 overflow-x-auto"
      style={{ background: 'var(--stepper-bg)', borderBottom: '1px solid var(--border)' }}
    >
      <div className="flex items-start gap-0 min-w-[700px]">
        {STAGES.map((s, i) => {
          const isDone   = done.includes(s.key)
          const isActive = s.key === active
          const isErr    = isError && s.key === active

          return (
            <div key={s.key} className="flex-1 flex flex-col items-center gap-1.5 relative">
              {/* Connector line */}
              {i < STAGES.length - 1 && (
                <div
                  className="pipe-connector"
                  style={isDone ? { background: '#2563eb' } : {}}
                />
              )}

              {/* Circle */}
              <div
                className={`relative z-10 w-7 h-7 rounded-full flex items-center justify-center text-[11px] border-2 transition-all duration-300
                  ${isDone   ? 'border-[#2563eb] bg-[#1e3a8a]'      : ''}
                  ${isActive && !isErr ? 'border-[#38bdf8] bg-[#0e3054] animate-pulse-glow' : ''}
                  ${isErr    ? 'border-[#ef4444] bg-[#3a0c0c]'       : ''}
                  ${!isDone && !isActive && !isErr ? '' : ''}
                `}
                style={!isDone && !isActive && !isErr ? {
                  borderColor: 'var(--border2)',
                  background: 'var(--panel)',
                  color: 'var(--text-s)',
                } : {}}
              >
                {isDone ? (
                  <span className="text-[#38bdf8] text-[10px]">✓</span>
                ) : isErr ? (
                  <span className="text-[#ef4444]">✗</span>
                ) : isActive ? (
                  <span className="animate-spin-slow">{s.icon}</span>
                ) : (
                  <span>{s.icon}</span>
                )}
              </div>

              {/* Label */}
              <span
                className={`font-mono text-[9px] text-center leading-tight max-w-[58px]
                  ${isDone   ? 'text-[#38bdf8]' : ''}
                  ${isActive ? 'text-[#7dd3fc]' : ''}
                  ${isErr    ? 'text-[#ef4444]' : ''}
                  ${!isDone && !isActive && !isErr ? 'text-[#1e3a5f]' : ''}
                `}
              >
                {s.label}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
