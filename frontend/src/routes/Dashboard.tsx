import {
  Database,
  FileText,
  BrainCircuit,
  Activity,
  ArrowRight,
  CheckCircle2,
  Loader2,
  Circle,
  TrendingUp,
} from 'lucide-react';

const pipelineSteps = [
  {
    title: 'Discovery & Collection',
    desc: '14 papers harvested from ArXiv, Semantic Scholar, and OpenAlex.',
    status: 'complete' as const,
    progress: 100,
  },
  {
    title: 'Document Intelligence',
    desc: 'OCR, layout analysis, and multi-modal extraction running…',
    status: 'active' as const,
    progress: 68,
  },
  {
    title: 'Multimodal RAG Indexing',
    desc: 'Embed text, figures, tables, and code with BGE-M3 + SigLIP.',
    status: 'pending' as const,
    progress: 0,
  },
  {
    title: 'Gap Analysis & Knowledge Graph',
    desc: 'Identify novelty gaps and build citation network.',
    status: 'pending' as const,
    progress: 0,
  },
  {
    title: 'Draft Generation & Review',
    desc: 'Produce IEEE-structured paper and simulate peer review.',
    status: 'pending' as const,
    progress: 0,
  },
];

function StatusIcon({ status }: { status: 'complete' | 'active' | 'pending' }) {
  if (status === 'complete')
    return <CheckCircle2 className="w-5 h-5 text-emerald-500" />;
  if (status === 'active')
    return <Loader2 className="w-5 h-5 text-magenta-500 animate-spin" />;
  return <Circle className="w-5 h-5 text-cream-400" />;
}

export default function Dashboard() {
  return (
    <div className="space-y-10">
      {/* ── Page Header ── */}
      <div className="animate-fade-in-up">
        <h1 className="text-4xl font-serif text-ink-300 mb-2">Research Overview</h1>
        <p className="text-ink-50 text-base max-w-xl">
          Real-time telemetry of your autonomous research pipeline. Monitor agents, track progress,
          and drill into each stage.
        </p>
      </div>

      {/* ── Metric Cards (Claymorphism) ── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Card 1 */}
        <div className="clay-card p-6 animate-fade-in-up stagger-1">
          <div className="flex items-start justify-between mb-5">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-magenta-50 to-magenta-100 flex items-center justify-center">
              <Database className="w-6 h-6 text-magenta-500" />
            </div>
            <span className="badge badge-active">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              ACTIVE
            </span>
          </div>
          <p className="text-xs font-semibold uppercase tracking-widest text-ink-50 mb-1">Current Project</p>
          <h3 className="text-2xl font-serif text-ink-300 mb-1">Multimodal RAG</h3>
          <p className="text-xs text-ink-50 opacity-60">Last updated: Just now</p>
        </div>

        {/* Card 2 */}
        <div className="clay-card p-6 animate-fade-in-up stagger-2">
          <div className="flex items-start justify-between mb-5">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-plum-50 to-plum-100 flex items-center justify-center">
              <FileText className="w-6 h-6 text-plum-400" />
            </div>
            <span className="badge badge-progress">
              <TrendingUp className="w-3 h-3" />
              +3 today
            </span>
          </div>
          <p className="text-xs font-semibold uppercase tracking-widest text-ink-50 mb-1">Documents Parsed</p>
          <div className="flex items-baseline gap-2">
            <h3 className="text-4xl font-serif text-ink-300">14</h3>
            <span className="text-sm text-ink-50">papers</span>
          </div>
          <div className="progress-track mt-4">
            <div className="progress-fill" style={{ width: '70%' }} />
          </div>
        </div>

        {/* Card 3 */}
        <div className="clay-card p-6 animate-fade-in-up stagger-3">
          <div className="flex items-start justify-between mb-5">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-magenta-50 to-plum-50 flex items-center justify-center">
              <BrainCircuit className="w-6 h-6 text-magenta-500" />
            </div>
          </div>
          <p className="text-xs font-semibold uppercase tracking-widest text-ink-50 mb-1">Current Phase</p>
          <h3 className="text-2xl font-serif text-gradient-magenta">Document Intelligence</h3>
          <p className="text-xs text-ink-50 opacity-60 mt-1">Extracting figures, tables, equations…</p>
        </div>
      </div>

      {/* ── Pipeline Timeline ── */}
      <div className="clay-card p-8 animate-fade-in-up stagger-4">
        <div className="flex items-center gap-3 mb-8">
          <Activity className="w-5 h-5 text-magenta-500" />
          <h2 className="text-2xl font-serif text-ink-300">Pipeline Telemetry</h2>
        </div>

        <div className="space-y-0">
          {pipelineSteps.map((step, i) => (
            <div key={i} className="relative flex gap-4 group">
              {/* Vertical line */}
              {i < pipelineSteps.length - 1 && (
                <div
                  className="absolute left-[9px] top-7 w-0.5 h-full"
                  style={{
                    background:
                      step.status === 'complete'
                        ? '#10b981'
                        : step.status === 'active'
                        ? 'linear-gradient(to bottom, #D80073, #E5DCC5)'
                        : '#E5DCC5',
                  }}
                />
              )}

              {/* Icon */}
              <div className="relative z-10 mt-0.5 flex-shrink-0">
                <StatusIcon status={step.status} />
              </div>

              {/* Content */}
              <div
                className={`flex-1 pb-8 transition-opacity duration-300 ${
                  step.status === 'pending' ? 'opacity-50' : ''
                }`}
              >
                <div className="flex items-center gap-3 mb-1">
                  <h4 className="font-semibold text-ink-200">{step.title}</h4>
                  {step.status === 'complete' && (
                    <span className="badge badge-active text-[10px]">Done</span>
                  )}
                  {step.status === 'active' && (
                    <span className="badge badge-progress text-[10px]">{step.progress}%</span>
                  )}
                  {step.status === 'pending' && (
                    <span className="badge badge-pending text-[10px]">Pending</span>
                  )}
                </div>
                <p className="text-sm text-ink-50">{step.desc}</p>
                {step.status === 'active' && (
                  <div className="progress-track mt-3 max-w-xs">
                    <div className="progress-fill" style={{ width: `${step.progress}%` }} />
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Quick Actions ── */}
      <div className="flex gap-4 animate-fade-in-up stagger-5">
        <button className="btn-magenta">
          Continue Pipeline
          <ArrowRight className="w-4 h-4" />
        </button>
        <button className="btn-outline">View All Papers</button>
        <button className="btn-outline">Open RAG Console</button>
      </div>
    </div>
  );
}
