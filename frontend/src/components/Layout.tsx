import { useState, useRef, useCallback, useEffect } from 'react';
import { Outlet, NavLink } from 'react-router-dom';
import {
  LayoutDashboard, Compass, FileText, MessageSquareText,
  Network, PenTool, ShieldCheck, Sparkles, Bot, User,
  AlertTriangle, CheckCircle2, XCircle, MessageCircle,
  ChevronLeft, ChevronRight, GripVertical, PanelLeftOpen, PanelLeftClose,
} from 'lucide-react';

/* ─── Navigation items for thin sidebar ─── */
const navItems = [
  { path: '/dashboard', icon: LayoutDashboard, tooltip: 'Dashboard' },
  { path: '/discovery', icon: Compass,           tooltip: 'Discovery' },
  { path: '/documents', icon: FileText,          tooltip: 'Documents' },
  { path: '/rag',       icon: MessageSquareText, tooltip: 'RAG Console' },
  { path: '/graph',     icon: Network,           tooltip: 'Knowledge Graph' },
  { path: '/draft',     icon: PenTool,           tooltip: 'Draft Editor' },
  { path: '/review',    icon: ShieldCheck,       tooltip: 'Review Sim' },
];

/* ─── Mock chat messages ─── */
const mockMessages = [
  {
    role: 'user' as const,
    content: 'Generate a comprehensive IEEE research paper exploring how multimodal RAG improves autonomous synthesis. Focus on BGE-M3 embeddings.',
  },
  {
    role: 'agent' as const,
    content: 'Understood. Initializing the Discovery agent to search ArXiv for recent papers on Multimodal RAG and BGE-M3.',
    action: 'Task queued: Literature Discovery',
    actionIcon: Compass,
  },
];

/* ─── Human-in-the-Loop Banner component ─── */
interface HitlBannerProps {
  message: string;
  onApprove: () => void;
  onRefuse: () => void;
  onFeedback: () => void;
}

function HitlBanner({ message, onApprove, onRefuse, onFeedback }: HitlBannerProps) {
  return (
    <div className="absolute top-5 left-1/2 -translate-x-1/2 z-50 w-[90%] max-w-2xl animate-fade-in-up pointer-events-auto">
      <div className="clay-card !rounded-2xl overflow-hidden shadow-2xl">
        <div className="h-1 bg-gradient-to-r from-amber-400 to-orange-400" />
        <div className="p-4 flex flex-col sm:flex-row gap-4 items-start sm:items-center">
          <div className="w-10 h-10 rounded-full bg-amber-100 flex items-center justify-center shrink-0">
            <AlertTriangle className="w-5 h-5 text-amber-600" />
          </div>
          <div className="flex-1 min-w-0">
            <h4 className="text-sm font-bold text-ink-300">⚠️ Action Required: Agent Checkpoint</h4>
            <p className="text-xs text-ink-100 mt-0.5 leading-relaxed">{message}</p>
          </div>
          <div className="flex gap-2 shrink-0">
            <button onClick={onApprove} className="flex items-center gap-1 px-3 py-1.5 bg-emerald-50 text-emerald-600 rounded-lg text-xs font-semibold hover:bg-emerald-100 transition-colors border border-emerald-200/60">
              <CheckCircle2 className="w-3.5 h-3.5" /> Approve
            </button>
            <button onClick={onRefuse} className="flex items-center gap-1 px-3 py-1.5 bg-red-50 text-red-500 rounded-lg text-xs font-semibold hover:bg-red-100 transition-colors border border-red-200/60">
              <XCircle className="w-3.5 h-3.5" /> Refuse
            </button>
            <button onClick={onFeedback} className="flex items-center gap-1 px-3 py-1.5 bg-blue-50 text-blue-500 rounded-lg text-xs font-semibold hover:bg-blue-100 transition-colors border border-blue-200/60">
              <MessageCircle className="w-3.5 h-3.5" /> Feedback
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ─── Main Layout ─── */
export default function Layout() {
  // Global icon sidebar expanded state
  const [sidebarExpanded, setSidebarExpanded] = useState(false);

  // Chat driver panel open/closed state
  const [chatOpen, setChatOpen] = useState(true);

  // Resizable left panel width (in px)
  const [chatWidth, setChatWidth] = useState(380);
  const isResizing = useRef(false);
  const startX = useRef(0);
  const startWidth = useRef(0);
  const containerRef = useRef<HTMLDivElement>(null);

  // Human-in-the-loop banner state — null = hidden, string = message to show
  const [hitlMessage, setHitlMessage] = useState<string | null>(null);

  /* ── Drag-to-resize handlers ── */
  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isResizing.current = true;
    startX.current = e.clientX;
    startWidth.current = chatWidth;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, [chatWidth]);

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!isResizing.current) return;
      const delta = e.clientX - startX.current;
      const newWidth = Math.min(Math.max(startWidth.current + delta, 260), 600);
      setChatWidth(newWidth);
    };
    const onMouseUp = () => {
      if (!isResizing.current) return;
      isResizing.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, []);

  /* ── Demo: simulate an agent checkpoint after 8 seconds ── */
  // In production, this would be triggered by a WebSocket event from the backend
  // Uncomment this to test the banner:
  // useEffect(() => {
  //   const t = setTimeout(() => {
  //     setHitlMessage("Reviewer B requests clarification on the 'Dataset Empiros' baseline parameters before continuing the empirical validation stage.");
  //   }, 8000);
  //   return () => clearTimeout(t);
  // }, []);

  const dismissHitl = () => setHitlMessage(null);

  return (
    <div ref={containerRef} className="relative w-screen h-screen overflow-hidden flex bg-cream-100">

      {/* ── Animated Background Blobs ── */}
      <div className="bg-blobs">
        <div className="blob blob-1" />
        <div className="blob blob-2" />
        <div className="blob blob-3" />
      </div>

      {/* ══════════════════════════════════════════════
           COLUMN 1 — Retractable Global Sidebar
         ══════════════════════════════════════════════ */}
      <aside
        className="relative z-20 glass flex flex-col py-5 border-r border-glass-border shrink-0 h-screen transition-all duration-300 ease-in-out overflow-hidden"
        style={{ width: sidebarExpanded ? '210px' : '68px' }}
      >
        {/* Logo + Toggle Row */}
        <div className="flex items-center gap-3 px-3 mb-6 shrink-0">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-magenta-500 to-plum-300 flex items-center justify-center shadow-md shrink-0">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          {/* Brand text — visible only when expanded */}
          <div
            className="overflow-hidden transition-all duration-300"
            style={{ width: sidebarExpanded ? '120px' : '0px', opacity: sidebarExpanded ? 1 : 0 }}
          >
            <p className="text-sm font-serif text-ink-300 whitespace-nowrap leading-tight">ResearchOS</p>
            <p className="text-[9px] font-mono text-ink-50 uppercase tracking-widest whitespace-nowrap">Workspace</p>
          </div>
        </div>

        {/* Nav items */}
        <nav className="flex flex-col gap-1 flex-1 w-full px-2 overflow-hidden">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                title={!sidebarExpanded ? item.tooltip : undefined}
                className={({ isActive }) =>
                  `relative group flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200
                  ${isActive
                    ? 'bg-magenta-50 text-magenta-600 shadow-sm'
                    : 'text-ink-100 hover:bg-white/50 hover:text-magenta-500 hover:translate-x-0.5'}`
                }
              >
                {({ isActive }) => (
                  <>
                    <Icon className={`w-5 h-5 shrink-0 transition-transform duration-200 ${isActive ? 'scale-110' : 'group-hover:scale-105'}`} />
                    {/* Label — slides in when expanded */}
                    <span
                      className="text-sm font-medium whitespace-nowrap overflow-hidden transition-all duration-300"
                      style={{ width: sidebarExpanded ? '130px' : '0px', opacity: sidebarExpanded ? 1 : 0 }}
                    >
                      {item.tooltip}
                    </span>
                    {/* Active dot */}
                    {isActive && (
                      <span className="absolute right-2 w-1.5 h-1.5 rounded-full bg-magenta-500 animate-pulse-glow shrink-0" />
                    )}
                    {/* Hover tooltip — only when collapsed */}
                    {!sidebarExpanded && (
                      <span className="absolute left-full ml-3 px-2 py-1 rounded-md bg-ink-300 text-white text-xs font-medium opacity-0 group-hover:opacity-100 transition-opacity duration-200 whitespace-nowrap pointer-events-none shadow-lg z-50">
                        {item.tooltip}
                      </span>
                    )}
                  </>
                )}
              </NavLink>
            );
          })}
        </nav>

        {/* Expand / Collapse toggle at the bottom */}
        <button
          onClick={() => setSidebarExpanded((e) => !e)}
          className="mx-2 mt-4 flex items-center gap-3 px-3 py-2.5 rounded-xl text-ink-50 hover:bg-white/40 hover:text-magenta-500 transition-all duration-200 group"
          title={sidebarExpanded ? 'Collapse sidebar' : 'Expand sidebar'}
        >
          {sidebarExpanded
            ? <PanelLeftClose className="w-5 h-5 shrink-0" />
            : <PanelLeftOpen  className="w-5 h-5 shrink-0" />}
          <span
            className="text-xs font-medium whitespace-nowrap overflow-hidden transition-all duration-300"
            style={{ width: sidebarExpanded ? '100px' : '0px', opacity: sidebarExpanded ? 1 : 0 }}
          >
            Collapse
          </span>
        </button>
      </aside>

      {/* ══════════════════════════════════════════════
           COLUMN 2 — Chat / Driver Panel (collapsible + resizable)
         ══════════════════════════════════════════════ */}
      <section
        className="relative z-10 flex flex-col glass-subtle border-r border-glass-border h-screen shrink-0 transition-all duration-300 ease-in-out overflow-hidden"
        style={{ width: chatOpen ? `${chatWidth}px` : '0px' }}
      >
        {/* Inner wrapper so content doesn't break when collapsed */}
        <div
          className="flex flex-col h-full"
          style={{ width: `${chatWidth}px`, minWidth: `${chatWidth}px` }}
        >
          {/* Header */}
          <header className="px-5 py-4 border-b border-white/20 bg-white/20 backdrop-blur-sm shrink-0 flex items-center justify-between">
            <div>
              <h2 className="text-base font-serif text-ink-300 leading-tight">ResearchOS</h2>
              <span className="text-[10px] font-mono text-ink-50 tracking-widest uppercase">Agentic Driver</span>
            </div>
          </header>

          {/* Chat History */}
          <div className="flex-1 overflow-y-auto p-5 space-y-5">
            {mockMessages.map((msg, i) => (
              <div key={i} className="flex gap-3">
                <div className={`w-7 h-7 rounded-full flex items-center justify-center shrink-0 mt-0.5 ${msg.role === 'user' ? 'bg-cream-300' : 'bg-magenta-100'}`}>
                  {msg.role === 'user'
                    ? <User className="w-3.5 h-3.5 text-ink-200" />
                    : <Bot className="w-3.5 h-3.5 text-magenta-600" />}
                </div>
                <div className={`p-3 rounded-2xl rounded-tl-sm shadow-sm border text-sm text-ink-200 leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-white/60 border-white/50'
                    : 'bg-magenta-50/50 border-magenta-100/50'
                }`}>
                  <p>{msg.content}</p>
                  {msg.action && msg.actionIcon && (
                    <div className="flex items-center gap-2 mt-2 p-2 bg-white/60 rounded-lg border border-white/60 text-xs">
                      <msg.actionIcon className="w-3.5 h-3.5 text-magenta-500 shrink-0" />
                      <span className="font-medium text-ink-200">{msg.action}</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Prompt Input */}
          <div className="p-4 bg-white/25 backdrop-blur-md border-t border-white/30 shrink-0">
            <div className="clay-card !p-4 space-y-3">
              <div>
                <label className="block text-[9px] font-bold uppercase tracking-widest text-ink-100 mb-1 ml-1">Topic / Paper Draft</label>
                <input
                  type="text"
                  placeholder="e.g. Multimodal Agentic RAG"
                  className="search-input !py-2 !text-sm !pl-4 !rounded-xl"
                />
              </div>
              <div>
                <label className="block text-[9px] font-bold uppercase tracking-widest text-ink-100 mb-1 ml-1">Research Goals & Guidelines</label>
                <textarea
                  rows={2}
                  placeholder="How should the agent structure the paper?"
                  className="search-input resize-none !py-2 !text-sm !pl-4 !rounded-xl"
                />
              </div>
              <button className="btn-magenta w-full justify-center !py-2.5 !text-sm">
                <Sparkles className="w-4 h-4" />
                Initialize Agent Pipeline
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════
           Drag Handle + Toggle Button
         ══════════════════════════════════════════════ */}
      <div className="relative z-30 flex items-center shrink-0">
        {/* Drag to resize (only visible when panel is open) */}
        {chatOpen && (
          <div
            onMouseDown={onMouseDown}
            className="w-1.5 h-full cursor-col-resize flex items-center justify-center group bg-transparent hover:bg-magenta-200/30 transition-colors duration-200"
            title="Drag to resize"
          >
            <GripVertical className="w-3 h-8 text-cream-400 group-hover:text-magenta-400 transition-colors" />
          </div>
        )}

        {/* Toggle Button */}
        <button
          onClick={() => setChatOpen((o) => !o)}
          title={chatOpen ? 'Collapse chat panel' : 'Open chat panel'}
          className="absolute -left-3 top-1/2 -translate-y-1/2 w-6 h-12 rounded-full glass border border-glass-border shadow-md flex items-center justify-center text-ink-100 hover:text-magenta-500 hover:border-magenta-300 transition-all duration-200 z-40"
        >
          {chatOpen ? <ChevronLeft className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        </button>
      </div>

      {/* ══════════════════════════════════════════════
           COLUMN 3 — Right Workspace Canvas (flex-1)
         ══════════════════════════════════════════════ */}
      <main className="relative z-10 flex-1 flex flex-col h-screen overflow-hidden bg-white/10 backdrop-blur-sm min-w-0">

        {/* HITL Banner — only mounts when hitlMessage is set */}
        {hitlMessage && (
          <HitlBanner
            message={hitlMessage}
            onApprove={dismissHitl}
            onRefuse={dismissHitl}
            onFeedback={dismissHitl}
          />
        )}

        {/* Live Telemetry Header */}
        <header className="px-6 py-3.5 border-b border-glass-border glass-subtle flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2.5">
            <div className="w-2 h-2 rounded-full bg-magenta-500 animate-pulse-glow" />
            <span className="text-sm text-ink-200">
              Agent active:{' '}
              <span className="text-magenta-600 font-semibold">Reviewing methodology…</span>
            </span>
          </div>
          <div className="flex items-center gap-3">
            {/* Demo: trigger HITL banner */}
            <button
              onClick={() =>
                setHitlMessage(
                  "Reviewer B requests clarification on the 'Dataset Empiros' baseline parameters before continuing the empirical validation stage."
                )
              }
              className="text-[11px] font-mono px-3 py-1 rounded-full bg-amber-50 border border-amber-200 text-amber-600 hover:bg-amber-100 transition-colors"
            >
              ⚠ Simulate Checkpoint
            </button>
            <div className="text-xs font-mono text-ink-50 px-3 py-1 rounded-full bg-white/40 border border-white/60">
              Memory: 42% · Tokens: 12k/128k
            </div>
          </div>
        </header>

        {/* Dynamic Workspace (current route content) */}
        <div className="flex-1 overflow-y-auto p-8">
          <div className="max-w-5xl mx-auto">
            <Outlet />
          </div>
        </div>
      </main>
    </div>
  );
}
