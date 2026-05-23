import React, { useState } from 'react'
import { useApp } from '../../context/AppContext'
import { useProject } from '../../hooks/useProject'

export default function HumanReviewBanner() {
  const { humanReview, clearHumanReview, pushChat } = useApp()
  const { approve } = useProject()
  const [showFeedback, setShowFeedback] = useState(false)
  const [feedback, setFeedback] = useState('')
  const [loading, setLoading] = useState(false)
  const [exiting, setExiting] = useState(false)

  if (!humanReview) return null

  const dismiss = (action) => {
    setExiting(true)
    setTimeout(() => {
      clearHumanReview()
      setExiting(false)
      setShowFeedback(false)
      setFeedback('')
    }, 300)
    pushChat('user', action)
  }

  const handleApprove = async () => {
    setLoading(true)
    try {
      await approve(feedback)
      dismiss('✅ Approved checkpoint: ' + humanReview.reviewer)
    } catch (_) {
    } finally {
      setLoading(false)
    }
  }

  const handleRefuse = () => {
    dismiss('🚫 Refused checkpoint: ' + humanReview.reviewer)
  }

  const handleFeedback = () => {
    if (!showFeedback) {
      setShowFeedback(true)
      return
    }
    dismiss(`💬 Feedback provided: ${feedback}`)
  }

  return (
    <div
      id="human-review-banner"
      className={`mx-5 mb-4 review-banner overflow-hidden flex-shrink-0
        ${exiting ? 'animate-slide-up' : 'animate-slide-down'}`}
      role="alert"
      aria-live="assertive"
    >
      {/* Warning Stripe */}
      <div
        className="h-1 w-full"
        style={{
          background: 'linear-gradient(90deg, #f59e0b, #2563eb, #f59e0b)',
          backgroundSize: '200% 100%',
          animation: 'shimmer 2s linear infinite',
        }}
      />

      <div className="px-5 py-4">
        {/* Header */}
        <div className="flex items-start gap-3 mb-4">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center text-xl flex-shrink-0 mt-0.5"
            style={{
              background: 'rgba(245,158,11,0.15)',
              border: '1px solid rgba(245,158,11,0.3)',
            }}
          >
            ⚠️
          </div>
          <div className="flex-1">
            <div className="text-[13px] font-semibold text-[#fbbf24] mb-1">
              Action Required
            </div>
            <p className="text-[12.5px] text-[#e0f2fe] leading-relaxed">
              {humanReview.message}
            </p>
            <div className="mt-1 font-mono text-[10px] text-[#4a6fa5]">
              Checkpoint · {humanReview.reviewer || 'Agent'}
            </div>
          </div>
        </div>

        {/* Feedback textarea */}
        {showFeedback && (
          <div className="mb-3 animate-fade-in-up">
            <textarea
              id="human-review-feedback"
              className="field-input text-[12.5px]"
              rows={3}
              placeholder="Provide your feedback or clarification…"
              value={feedback}
              onChange={e => setFeedback(e.target.value)}
              autoFocus
            />
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex items-center gap-2 flex-wrap">
          <button
            id="human-review-approve-btn"
            onClick={handleApprove}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-white text-[12.5px] font-semibold transition-all hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50"
            style={{
              background: 'linear-gradient(135deg, #2563eb, #1d4ed8)',
              boxShadow: '0 4px 16px rgba(37,99,235,0.3)',
            }}
          >
            {loading ? <Spinner /> : '✓'} Approve
          </button>

          <button
            id="human-review-refuse-btn"
            onClick={handleRefuse}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-[12.5px] font-semibold border border-red-700/40 text-red-400 hover:bg-red-500/10 hover:border-red-500 transition-all"
          >
            🚫 Refuse
          </button>

          <button
            id="human-review-feedback-btn"
            onClick={handleFeedback}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-[12.5px] font-medium border border-[#1a3a6e] text-[#93c5fd] hover:text-[#e0f2fe] hover:border-[#2563eb] hover:bg-[#2563eb]/10 transition-all"
          >
            💬 {showFeedback ? 'Send Feedback' : 'Provide Feedback'}
          </button>

          <button
            id="human-review-dismiss-btn"
            onClick={() => dismiss('Dismissed checkpoint')}
            className="ml-auto text-[#1e3a5f] hover:text-[#4a6fa5] transition-colors text-xs font-mono"
          >
            dismiss
          </button>
        </div>
      </div>
    </div>
  )
}

function Spinner() {
  return <span className="w-3.5 h-3.5 rounded-full border-2 border-transparent border-t-white animate-spin inline-block" />
}
