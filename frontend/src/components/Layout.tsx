import { Outlet, NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Compass,
  FileText,
  MessageSquareText,
  Network,
  PenTool,
  ShieldCheck,
  Download,
  Sparkles,
  Search,
} from 'lucide-react';

const navItems = [
  { path: '/dashboard',  label: 'Dashboard',       icon: LayoutDashboard },
  { path: '/discovery',  label: 'Discovery',        icon: Compass },
  { path: '/documents',  label: 'Documents',        icon: FileText },
  { path: '/rag',        label: 'RAG Console',      icon: MessageSquareText },
  { path: '/graph',      label: 'Knowledge Graph',  icon: Network },
  { path: '/draft',      label: 'Draft Editor',     icon: PenTool },
  { path: '/review',     label: 'Review Sim',       icon: ShieldCheck },
];

export default function Layout() {
  return (
    <div className="relative min-h-screen">
      {/* ── Animated Background Blobs ── */}
      <div className="bg-blobs">
        <div className="blob blob-1" />
        <div className="blob blob-2" />
        <div className="blob blob-3" />
      </div>

      {/* ── App Shell ── */}
      <div className="relative z-10 flex min-h-screen">

        {/* ── Glassmorphism Sidebar ── */}
        <aside className="fixed top-0 left-0 bottom-0 w-[260px] glass flex flex-col py-6 px-4 z-20 animate-slide-left">
          {/* Logo */}
          <div className="flex items-center gap-3 px-3 mb-8">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-magenta-500 to-plum-300 flex items-center justify-center shadow-lg">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-ink-300 leading-tight font-serif">ResearchOS</h1>
              <span className="text-[10px] font-mono text-ink-50 tracking-widest uppercase">v2.0 · Agentic</span>
            </div>
          </div>

          {/* Nav Items */}
          <nav className="flex-1 space-y-1">
            <p className="px-3 mb-3 text-[10px] font-semibold uppercase tracking-[0.2em] text-ink-50 opacity-60">
              Workspace
            </p>
            {navItems.map((item, i) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) => {
                    const base = `
                      flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium
                      transition-all duration-300 group relative
                      animate-fade-in-up stagger-${i + 1}
                    `;
                    return isActive
                      ? `${base} bg-magenta-50 text-magenta-600 shadow-sm border border-magenta-100`
                      : `${base} text-ink-100 hover:text-magenta-500 hover:bg-white/40 hover:translate-x-1`;
                  }}
                >
                  {({ isActive }) => (
                    <>
                      <Icon
                        className={`w-[18px] h-[18px] transition-transform duration-300 group-hover:scale-110 ${
                          isActive ? 'text-magenta-500' : ''
                        }`}
                      />
                      <span>{item.label}</span>
                      {isActive && (
                        <div className="absolute right-3 w-1.5 h-1.5 rounded-full bg-magenta-500 animate-pulse-glow" />
                      )}
                    </>
                  )}
                </NavLink>
              );
            })}
          </nav>

          {/* System Status Footer */}
          <div className="px-3 pt-4 border-t border-cream-300/50">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.6)]" />
              <span className="text-xs font-mono text-ink-50">System Online</span>
            </div>
          </div>
        </aside>

        {/* ── Main Content ── */}
        <main className="ml-[260px] flex-1 flex flex-col min-h-screen">
          {/* Top Bar */}
          <header className="sticky top-0 z-10 px-8 py-4 flex items-center justify-between glass-subtle">
            {/* Search Bar */}
            <div className="relative flex-1 max-w-2xl">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-ink-50 opacity-50" />
              <input
                type="text"
                placeholder="Search papers, methods, datasets, or ask a question…"
                className="search-input"
              />
            </div>

            {/* Export Button */}
            <button className="btn-magenta ml-6 animate-pulse-glow">
              <Download className="w-4 h-4" />
              Export Final
            </button>
          </header>

          {/* Page Content */}
          <div className="flex-1 px-8 py-8 overflow-auto">
            <div className="max-w-6xl mx-auto">
              <Outlet />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
