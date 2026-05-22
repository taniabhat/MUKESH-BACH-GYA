import { Network, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';

export default function KnowledgeGraph() {
  return (
    <div className="space-y-8">
      <div className="animate-fade-in-up">
        <h1 className="text-4xl font-serif text-ink-300 mb-2">Knowledge Graph</h1>
        <p className="text-ink-50 max-w-xl">
          Visualize the citation network, method relationships, dataset lineage, and
          research gap topology extracted from your corpus.
        </p>
      </div>

      {/* Graph Canvas */}
      <div className="clay-card p-0 overflow-hidden animate-fade-in-up stagger-1" style={{ minHeight: '500px' }}>
        {/* Toolbar */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-cream-300/50">
          <div className="flex items-center gap-2">
            <Network className="w-4 h-4 text-magenta-500" />
            <span className="text-sm font-medium text-ink-200">Graph Explorer</span>
          </div>
          <div className="flex gap-1">
            {[ZoomIn, ZoomOut, Maximize2].map((Icon, i) => (
              <button
                key={i}
                className="p-2 rounded-lg hover:bg-cream-200 transition-colors text-ink-50 hover:text-magenta-500"
              >
                <Icon className="w-4 h-4" />
              </button>
            ))}
          </div>
        </div>

        {/* Graph Placeholder */}
        <div className="flex flex-col items-center justify-center py-24 text-ink-50 opacity-50">
          <Network className="w-16 h-16 mb-4 opacity-20" />
          <p className="text-sm">Run the analysis pipeline to populate the knowledge graph.</p>
          <p className="text-xs mt-1 opacity-60">Papers → Methods → Datasets → Gaps</p>
        </div>
      </div>

      {/* Legend */}
      <div className="flex gap-6 animate-fade-in-up stagger-2">
        {[
          { label: 'Paper', color: 'bg-magenta-500' },
          { label: 'Method', color: 'bg-plum-400' },
          { label: 'Dataset', color: 'bg-emerald-500' },
          { label: 'Gap', color: 'bg-amber-500' },
        ].map((item) => (
          <div key={item.label} className="flex items-center gap-2 text-xs text-ink-50">
            <div className={`w-3 h-3 rounded-full ${item.color}`} />
            {item.label}
          </div>
        ))}
      </div>
    </div>
  );
}
