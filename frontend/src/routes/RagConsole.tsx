import { useState } from 'react';
import { Send, BookOpen, Image, Table2, Code2, Sparkles } from 'lucide-react';

export default function RagConsole() {
  const [query, setQuery] = useState('');

  return (
    <div className="space-y-8">
      <div className="animate-fade-in-up">
        <h1 className="text-4xl font-serif text-ink-300 mb-2">RAG Console</h1>
        <p className="text-ink-50 max-w-xl">
          Query the multimodal knowledge base. Hybrid retrieval fuses dense, sparse, and
          cross-encoder signals across text, figures, tables, and code.
        </p>
      </div>

      {/* Query Panel */}
      <div className="clay-card p-6 animate-fade-in-up stagger-1">
        <div className="flex gap-3">
          <div className="relative flex-1">
            <Sparkles className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-magenta-400" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="What attention mechanism does the proposed system use?"
              className="search-input !text-sm"
            />
          </div>
          <button className="btn-magenta flex-shrink-0">
            <Send className="w-4 h-4" />
            Query
          </button>
        </div>

        {/* Modality Filters */}
        <div className="flex gap-2 mt-4">
          {[
            { icon: BookOpen, label: 'Text' },
            { icon: Image, label: 'Figures' },
            { icon: Table2, label: 'Tables' },
            { icon: Code2, label: 'Code' },
          ].map((m) => {
            const Icon = m.icon;
            return (
              <button
                key={m.label}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
                  bg-cream-200/50 text-ink-100 border border-cream-300 hover:border-magenta-300
                  hover:text-magenta-500 hover:bg-magenta-50 transition-all duration-200"
              >
                <Icon className="w-3.5 h-3.5" />
                {m.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Results Placeholder */}
      <div className="clay-card p-10 text-center animate-fade-in-up stagger-2">
        <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-magenta-50 to-plum-50 flex items-center justify-center">
          <Sparkles className="w-7 h-7 text-magenta-400 opacity-50" />
        </div>
        <h3 className="text-lg font-serif text-ink-200 mb-1">No results yet</h3>
        <p className="text-sm text-ink-50 opacity-60">
          Enter a query above to retrieve relevant passages, figures, and tables.
        </p>
      </div>
    </div>
  );
}
