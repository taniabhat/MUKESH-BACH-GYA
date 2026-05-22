import { useState } from 'react';
import { Compass, ArrowRight, Globe, BookOpen, Database as DbIcon, Loader2 } from 'lucide-react';

const sources = [
  { name: 'Semantic Scholar', icon: BookOpen, color: 'text-magenta-500', bg: 'bg-magenta-50' },
  { name: 'ArXiv',           icon: Globe,    color: 'text-plum-400',    bg: 'bg-plum-50' },
  { name: 'OpenAlex',        icon: DbIcon,   color: 'text-emerald-600', bg: 'bg-emerald-50' },
];

export default function Discovery() {
  const [query, setQuery] = useState('');
  const [searching, setSearching] = useState(false);

  return (
    <div className="space-y-8">
      <div className="animate-fade-in-up">
        <h1 className="text-4xl font-serif text-ink-300 mb-2">Discovery & Collection</h1>
        <p className="text-ink-50 max-w-xl">
          Define your research hypothesis and let agents discover relevant literature
          from academic databases.
        </p>
      </div>

      {/* Hypothesis Input */}
      <div className="clay-card p-8 animate-fade-in-up stagger-1">
        <label className="block text-xs font-semibold uppercase tracking-widest text-ink-50 mb-3">
          Research Hypothesis
        </label>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          rows={4}
          placeholder="e.g., How can multimodal retrieval-augmented generation improve the quality of autonomously generated research papers?"
          className="search-input resize-none !pl-5"
          style={{ paddingLeft: '20px' }}
        />
        <div className="mt-6 flex items-center justify-between">
          <div className="flex gap-3">
            {sources.map((s) => {
              const Icon = s.icon;
              return (
                <div
                  key={s.name}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium ${s.bg} ${s.color} border border-white/50`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {s.name}
                </div>
              );
            })}
          </div>
          <button
            className="btn-magenta"
            onClick={() => setSearching(true)}
          >
            {searching ? <Loader2 className="w-4 h-4 animate-spin" /> : <Compass className="w-4 h-4" />}
            Search Literature
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Placeholder Results */}
      <div className="clay-card p-8 animate-fade-in-up stagger-2">
        <h3 className="text-lg font-serif text-ink-200 mb-4">Discovered Papers</h3>
        <div className="text-center py-12 text-ink-50 opacity-50">
          <Compass className="w-10 h-10 mx-auto mb-3 opacity-30" />
          <p>Enter a hypothesis above and run discovery to find papers.</p>
        </div>
      </div>
    </div>
  );
}
