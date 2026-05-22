import { ShieldCheck, Play, AlertTriangle, CheckCircle2, XCircle, TrendingUp, BookOpen, Beaker, Code2 } from 'lucide-react';

const reviewers = [
  {
    name: 'Reviewer A — Methodology',
    avatar: '🧠',
    focus: 'Theoretical soundness, formal proofs, and methodological rigor.',
    scores: [
      { label: 'Methodology', score: null },
      { label: 'Novelty',     score: null },
      { label: 'Clarity',     score: null },
    ],
  },
  {
    name: 'Reviewer B — Empirical',
    avatar: '🔬',
    focus: 'Datasets, experimental design, baselines, and reproducibility.',
    scores: [
      { label: 'Reproducibility', score: null },
      { label: 'Empirical Rigor', score: null },
      { label: 'Statistical Validity', score: null },
    ],
  },
  {
    name: 'Reviewer C — Presentation',
    avatar: '📝',
    focus: 'Writing quality, structure, figures, and citation coverage.',
    scores: [
      { label: 'Writing Quality', score: null },
      { label: 'Figure Quality',  score: null },
      { label: 'Citations',       score: null },
    ],
  },
];

export default function ReviewSimulator() {
  return (
    <div className="space-y-8">
      <div className="flex items-start justify-between animate-fade-in-up">
        <div>
          <h1 className="text-4xl font-serif text-ink-300 mb-2">Review Simulator</h1>
          <p className="text-ink-50 max-w-xl">
            Simulate a multi-reviewer peer review process. AI agents score your draft across
            methodology, empirics, and presentation quality.
          </p>
        </div>
        <button className="btn-magenta">
          <Play className="w-4 h-4" />
          Run Simulation
        </button>
      </div>

      {/* Reviewer Panels */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {reviewers.map((rev, i) => (
          <div key={i} className={`clay-card p-6 animate-fade-in-up stagger-${i + 1}`}>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-magenta-50 to-plum-50 flex items-center justify-center text-lg">
                {rev.avatar}
              </div>
              <div>
                <h3 className="text-sm font-semibold text-ink-200">{rev.name}</h3>
              </div>
            </div>

            <p className="text-xs text-ink-50 mb-5 leading-relaxed">{rev.focus}</p>

            <div className="space-y-3">
              {rev.scores.map((s) => (
                <div key={s.label} className="flex items-center justify-between">
                  <span className="text-sm text-ink-100">{s.label}</span>
                  <span className="text-sm font-mono text-ink-50">—/10</span>
                </div>
              ))}
            </div>

            <div className="mt-5 pt-4 border-t border-cream-300/50">
              <p className="text-xs text-ink-50 italic opacity-60">
                Awaiting simulation results…
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* Issues Panel */}
      <div className="clay-card p-8 animate-fade-in-up stagger-4">
        <div className="flex items-center gap-3 mb-6">
          <AlertTriangle className="w-5 h-5 text-amber-500" />
          <h2 className="text-xl font-serif text-ink-300">Issue Tracker</h2>
        </div>
        <div className="text-center py-10 text-ink-50 opacity-50">
          <ShieldCheck className="w-10 h-10 mx-auto mb-3 opacity-20" />
          <p className="text-sm">Run the review simulation to populate issue reports.</p>
        </div>
      </div>
    </div>
  );
}
