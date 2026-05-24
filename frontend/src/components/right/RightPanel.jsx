import React from 'react'
import TelemetryBar from './TelemetryBar'
import PipelineStepper from './PipelineStepper'
import HumanReviewBanner from './HumanReviewBanner'
import ReviewerCards from './ReviewerCards'
import PapersPanel, {
  GapsPanel, DraftPanel, CitationsPanel, AssetsPanel, ExportPanel
} from './PapersPanel'
import { useApp } from '../../context/AppContext'
import { FaFlask, FaSearch, FaFileAlt, FaMicroscope, FaPen, FaEye, FaFileExport } from 'react-icons/fa'

export default function RightPanel({ style }) {
  const { currentProject, humanReview } = useApp()

  return (
    <div
      id="right-panel"
      className="flex flex-col h-full overflow-hidden relative"
      style={style}
    >
      {/* ── Animated Background Blobs ─────────────────── */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none" style={{ zIndex: 0 }}>
        <div className="bg-blob bg-blob-1" />
        <div className="bg-blob bg-blob-2" />
        <div className="bg-blob bg-blob-3" />
      </div>

      {/* ── Main Content (above blobs) ────────────────── */}
      <div className="relative flex flex-col h-full overflow-hidden" style={{ zIndex: 1 }}>
        {/* Telemetry Bar */}
        <TelemetryBar />

        {/* Pipeline Stepper */}
        <PipelineStepper />

        {/* Scrollable Workspace Canvas */}
        <div className="flex-1 overflow-y-auto py-4">
          {/* Human Review Banner */}
          {humanReview && <HumanReviewBanner />}

          {/* No project selected state */}
          {!currentProject ? (
            <IdleWorkspace />
          ) : (
            <>
              {/* Reviewer Cards */}
              <ReviewerCards />

              {/* Dynamic Output Panels */}
              <PapersPanel />
              <GapsPanel />
              <DraftPanel />
              <CitationsPanel />
              <AssetsPanel />
              <ExportPanel />

              {/* Bottom padding */}
              <div className="h-6" />
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function IdleWorkspace() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-12">
      <div
        className="w-20 h-20 rounded-3xl flex items-center justify-center mb-6"
        style={{
          background: 'var(--idle-icon-bg)',
          border: '1px solid var(--idle-icon-border)',
          boxShadow: 'var(--idle-icon-shadow)',
        }}
      >
        <FaFlask className="text-4xl text-[#D80073]" />
      </div>
      <h2 className="font-display text-[26px] uppercase font-bold text-[#e0f2fe] mb-3">
        Workspace Canvas
      </h2>
      <p className="text-[13px] text-[#4a6fa5] max-w-[320px] leading-relaxed">
        Initialize an agent in the left panel to begin. The pipeline's live workings,
        reviewer feedback, and generated content will appear here.
      </p>

      {/* Feature Chips */}
      <div className="flex flex-wrap gap-2 justify-center mt-8">
        {[
          { icon: <FaSearch />, text: 'Discovery' },
          { icon: <FaFileAlt />, text: 'Analysis' },
          { icon: <FaMicroscope />, text: 'Gap Detection' },
          { icon: <FaPen />, text: 'Drafting' },
          { icon: <FaEye />, text: 'Peer Review' },
          { icon: <FaFileExport />, text: 'Export' }
        ].map(f => (
          <span
            key={f.text}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full font-mono text-[11px] border"
            style={{
              background: 'var(--idle-chip-bg)',
              borderColor: 'var(--border)',
              color: 'var(--text-m)',
            }}
          >
            {f.icon} {f.text}
          </span>
        ))}
      </div>
    </div>
  )
}
