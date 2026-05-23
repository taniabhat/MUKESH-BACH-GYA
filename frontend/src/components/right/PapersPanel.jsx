import React, { useState, useEffect } from 'react'
import { useApp } from '../../context/AppContext'
import { useProject } from '../../hooks/useProject'

export default function PapersPanel() {
  const { currentProject } = useApp()
  const { fetchPapers } = useProject()
  const [papers, setPapers] = useState([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)

  const should = currentProject && ['analyzing','idle','approved','drafting','refining','humanizing','reviewing','complete'].includes(currentProject.status)

  useEffect(() => {
    if (!should) return
    setLoading(true)
    fetchPapers(currentProject.id)
      .then(data => { setPapers(data.items || []); setTotal(data.total || 0) })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [currentProject?.id, currentProject?.status])

  if (!should) return null

  return (
    <SectionCard id="papers-panel" title="Discovered Papers" badge={loading ? '…' : `${total} papers`}>
      {loading ? <ShimmerList /> : papers.length === 0 ? <Empty icon="📭" text="No papers discovered yet" /> : (
        <div className="flex flex-col divide-y divide-[#0c2040]">
          {papers.slice(0, 30).map((p, i) => {
            const authors = Array.isArray(p.authors)
              ? p.authors.slice(0, 2).map(a => typeof a === 'string' ? a : a.name || '').join(', ')
              : ''
            const score = p.relevance_score ? (p.relevance_score * 100).toFixed(0) + '%' : ''
            return (
              <div key={p.id || i} className="flex gap-3 py-3">
                <div className="font-mono text-[15px] text-[#1e3a5f] min-w-[26px] pt-0.5">{String(i + 1).padStart(2, '0')}</div>
                <div className="flex-1 min-w-0">
                  <div className="text-[12.5px] text-[#e0f2fe] font-medium leading-snug mb-0.5">{p.title || 'Untitled'}</div>
                  <div className="font-mono text-[10px] text-[#4a6fa5]">
                    {authors}{p.year ? ` · ${p.year}` : ''}{p.doi ? ` · ${p.doi}` : ''}
                  </div>
                  {p.abstract && (
                    <div className="text-[11px] text-[#4a6fa5] mt-1 leading-relaxed line-clamp-2">
                      {p.abstract.slice(0, 120)}…
                    </div>
                  )}
                </div>
                {score && <div className="font-mono text-[11px] text-[#38bdf8] flex-shrink-0">{score}</div>}
              </div>
            )
          })}
        </div>
      )}
    </SectionCard>
  )
}

// ── Gaps Panel ────────────────────────────────────────────
export function GapsPanel() {
  const { currentProject } = useApp()
  const { fetchGaps } = useProject()
  const [gaps, setGaps] = useState([])
  const [loading, setLoading] = useState(false)

  const should = currentProject && ['approved','drafting','refining','humanizing','reviewing','complete'].includes(currentProject.status)

  useEffect(() => {
    if (!should) return
    setLoading(true)
    fetchGaps(currentProject.id)
      .then(setGaps)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [currentProject?.id, currentProject?.status])

  if (!should) return null

  return (
    <SectionCard id="gaps-panel" title="Research Gaps" badge={loading ? '…' : `${gaps.length} gaps`}>
      {loading ? <ShimmerList /> : gaps.length === 0 ? <Empty icon="🔬" text="No gaps identified yet" /> : (
        <div className="flex flex-col gap-3">
          {gaps.map((g, i) => {
            const sevCls = g.severity === 'high' ? 'sev-high' : g.severity === 'medium' ? 'sev-medium' : 'sev-low'
            const contribs = (g.suggested_contributions || []).slice(0, 3)
            return (
              <div key={g.id || i} className="gap-card">
                <span className={`inline-block font-mono text-[9px] px-2 py-0.5 rounded-full mb-2 ${sevCls}`}>
                  {(g.severity || '').toUpperCase()}
                </span>
                <h4 className="text-[12.5px] text-[#e0f2fe] font-medium mb-1">{g.title || g.id || 'Research Gap'}</h4>
                <p className="text-[11.5px] text-[#4a6fa5] leading-relaxed">{g.novelty_opportunity || ''}</p>
                {contribs.length > 0 && (
                  <ul className="mt-2 pl-4 text-[11px] text-[#4a6fa5] space-y-0.5 list-disc">
                    {contribs.map((c, j) => <li key={j}>{c}</li>)}
                  </ul>
                )}
              </div>
            )
          })}
        </div>
      )}
    </SectionCard>
  )
}

// ── Draft Panel ───────────────────────────────────────────
export function DraftPanel() {
  const { currentProject } = useApp()
  const { fetchDraft } = useProject()
  const [draft, setDraft] = useState(null)
  const [activeTab, setActiveTab] = useState(null)
  const [loading, setLoading] = useState(false)

  const should = currentProject && ['refining','humanizing','reviewing','complete'].includes(currentProject.status)

  useEffect(() => {
    if (!should) return
    setLoading(true)
    fetchDraft(currentProject.id)
      .then(d => { setDraft(d); if (d?.sections) setActiveTab(Object.keys(d.sections)[0]) })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [currentProject?.id, currentProject?.status])

  if (!should) return null

  const sections = draft?.sections || {}
  const keys = Object.keys(sections)

  return (
    <SectionCard
      id="draft-panel"
      title="Paper Draft"
      badge={draft ? `v${draft.version} · ${draft.status}` : loading ? '…' : '—'}
    >
      {loading ? <ShimmerList /> : !keys.length ? <Empty icon="✍️" text="Draft not yet generated" /> : (
        <>
          <div className="flex gap-0 border-b border-[#0c2040] mb-4 overflow-x-auto">
            {keys.map(k => (
              <button
                key={k}
                onClick={() => setActiveTab(k)}
                className={`tab-item ${activeTab === k ? 'active' : ''}`}
              >
                {k}
              </button>
            ))}
          </div>
          <pre className="text-[12.5px] text-[#93c5fd] leading-relaxed whitespace-pre-wrap font-sans max-h-[400px] overflow-y-auto">
            {sections[activeTab] || ''}
          </pre>
        </>
      )}
    </SectionCard>
  )
}

// ── Citations Panel ───────────────────────────────────────
export function CitationsPanel() {
  const { currentProject } = useApp()
  const { fetchCitations } = useProject()
  const [citations, setCitations] = useState([])
  const [loading, setLoading] = useState(false)

  const should = currentProject && ['reviewing','complete'].includes(currentProject.status)

  useEffect(() => {
    if (!should) return
    setLoading(true)
    fetchCitations(currentProject.id)
      .then(setCitations)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [currentProject?.id, currentProject?.status])

  if (!should) return null

  return (
    <SectionCard id="citations-panel" title="Citations" badge={loading ? '…' : `${citations.length} citations`}>
      {loading ? <ShimmerList /> : citations.length === 0 ? <Empty icon="📚" text="No citations yet" /> : (
        <div className="flex flex-col divide-y divide-[#0c2040]">
          {citations.map((c, i) => {
            const dotCls = c.validation_status === 'verified' ? 'bg-[#10b981]' : c.validation_status === 'warning' ? 'bg-[#f59e0b]' : 'bg-[#ef4444]'
            return (
              <div key={i} className="flex gap-2.5 items-start py-2.5">
                <div className={`w-2 h-2 rounded-full mt-1.5 flex-shrink-0 ${dotCls}`} title={c.validation_status} />
                <div className="font-mono text-[11px] text-[#4a6fa5] flex-1 leading-relaxed">
                  {(c.bibtex || '').slice(0, 200)}{(c.bibtex || '').length > 200 ? '…' : ''}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </SectionCard>
  )
}

// ── Assets Panel ──────────────────────────────────────────
export function AssetsPanel() {
  const { currentProject } = useApp()
  const { fetchAssets, genCode, genDiag } = useProject()
  const [assets, setAssets] = useState([])
  const [loading, setLoading] = useState(false)

  const should = currentProject?.status === 'complete'

  useEffect(() => {
    if (!should) return
    setLoading(true)
    fetchAssets(currentProject.id)
      .then(setAssets)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [currentProject?.id, currentProject?.status])

  if (!should) return null

  return (
    <SectionCard
      id="assets-panel"
      title="Generated Assets"
      headerRight={
        <div className="flex gap-2">
          <GhostBtn onClick={genCode}>⚙️ Code</GhostBtn>
          <GhostBtn onClick={genDiag}>📊 Diagrams</GhostBtn>
        </div>
      }
    >
      {loading ? <ShimmerList /> : assets.length === 0 ? <Empty icon="⚙️" text="Generate code or diagrams above" /> : (
        <div className="grid grid-cols-2 gap-2.5">
          {assets.map((a, i) => (
            <div key={i} className="p-3 rounded-xl border border-[#1a3a6e] bg-[#0c1f3a]/50">
              <div className="font-mono text-[9px] text-[#38bdf8] uppercase tracking-wider mb-1">{a.asset_type}</div>
              <div className="text-[11.5px] text-[#e0f2fe] break-all">{a.file_path ? a.file_path.split('/').pop() : a.id}</div>
              <div className="font-mono text-[9px] text-[#1e3a5f] mt-1">{new Date(a.created_at).toLocaleDateString()}</div>
            </div>
          ))}
        </div>
      )}
    </SectionCard>
  )
}

// ── Export Panel ──────────────────────────────────────────
export function ExportPanel() {
  const { currentProject } = useApp()
  const { exportPdf, downloadExport } = useProject()
  const [building, setBuilding] = useState(null)

  const should = currentProject?.status === 'complete'
  if (!should) return null

  const doExport = async (fmt) => {
    setBuilding(fmt)
    try { await exportPdf(fmt) } finally { setBuilding(null) }
  }

  return (
    <SectionCard id="export-panel" title="Export Paper">
      <p className="text-[12px] text-[#4a6fa5] mb-4">Export in IEEE two-column PDF, LaTeX source, or DOCX.</p>
      <div className="flex gap-2.5 flex-wrap mb-4">
        {['pdf','tex','docx'].map(fmt => (
          <button
            key={fmt}
            id={`export-build-${fmt}`}
            onClick={() => doExport(fmt)}
            disabled={!!building}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-[12.5px] font-medium text-white transition-all disabled:opacity-50 hover:scale-[1.02]"
            style={{ background: 'linear-gradient(135deg, #2563eb, #1d4ed8)', boxShadow: '0 4px 16px rgba(37,99,235,0.25)' }}
          >
            {building === fmt ? <Spin /> : '🚀'} {fmt === 'pdf' ? 'Build PDF' : fmt === 'tex' ? 'LaTeX' : 'DOCX'}
          </button>
        ))}
      </div>
      <div className="border-t border-[#0c2040] pt-4">
        <p className="text-[11px] text-[#4a6fa5] mb-3">Download previously built:</p>
        <div className="flex gap-2 flex-wrap">
          {['pdf','tex','docx'].map(fmt => (
            <button
              key={fmt}
              id={`export-dl-${fmt}`}
              onClick={() => downloadExport(fmt)}
              className="flex items-center gap-1.5 px-3 py-2 rounded-lg border border-[#1a3a6e] text-[11.5px] text-[#93c5fd] hover:text-[#e0f2fe] hover:border-[#2563eb] transition-all"
            >
              ⬇ {fmt.toUpperCase()}
            </button>
          ))}
        </div>
      </div>
    </SectionCard>
  )
}

// ── Shared Utilities ──────────────────────────────────────
function SectionCard({ id, title, badge, headerRight, children }) {
  return (
    <div id={id} className="mx-5 mb-4 clay-card overflow-hidden animate-fade-in-up">
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-[#0c2040]">
        <h3 className="font-display text-[14px] text-[#e0f2fe] font-semibold">{title}</h3>
        {badge && <span className="font-mono text-[10px] text-[#4a6fa5]">{badge}</span>}
        {headerRight}
      </div>
      <div className="px-5 py-4">{children}</div>
    </div>
  )
}

function Empty({ icon, text }) {
  return (
    <div className="empty-state">
      <div className="text-3xl mb-3 opacity-40">{icon}</div>
      <div>{text}</div>
    </div>
  )
}

function ShimmerList() {
  return (
    <div className="flex flex-col gap-3">
      {[0, 1, 2].map(i => (
        <div key={i}>
          <div className="shimmer-bar rounded mb-2" />
          <div className="shimmer-bar rounded" style={{ width: '60%' }} />
        </div>
      ))}
    </div>
  )
}

function GhostBtn({ children, onClick }) {
  return (
    <button
      onClick={onClick}
      className="px-2.5 py-1.5 rounded-lg border border-[#1a3a6e] text-[11px] text-[#4a6fa5] hover:text-[#e0f2fe] hover:border-[#2563eb] transition-all"
    >
      {children}
    </button>
  )
}

function Spin() {
  return <span className="w-3.5 h-3.5 rounded-full border-2 border-transparent border-t-white animate-spin inline-block" />
}
