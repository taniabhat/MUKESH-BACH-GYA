import { useState, useCallback } from 'react';
import { FileText, Upload, File, CheckCircle2, Image, Table2, Code2 } from 'lucide-react';

export default function Documents() {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  return (
    <div className="space-y-8">
      <div className="animate-fade-in-up">
        <h1 className="text-4xl font-serif text-ink-300 mb-2">Document Intelligence</h1>
        <p className="text-ink-50 max-w-xl">
          Upload research papers and watch the extraction pipeline parse sections, figures,
          tables, equations, and code blocks automatically.
        </p>
      </div>

      {/* Upload Zone */}
      <div
        className={`clay-card p-0 overflow-hidden animate-fade-in-up stagger-1 transition-all duration-300 ${
          isDragOver ? 'ring-2 ring-magenta-300 scale-[1.01]' : ''
        }`}
        onDragOver={(e) => { handleDrag(e); setIsDragOver(true); }}
        onDragLeave={(e) => { handleDrag(e); setIsDragOver(false); }}
        onDrop={(e) => { handleDrag(e); setIsDragOver(false); }}
      >
        <div className="p-12 text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-magenta-50 to-plum-50 flex items-center justify-center">
            <Upload className={`w-7 h-7 transition-colors ${isDragOver ? 'text-magenta-500' : 'text-ink-50'}`} />
          </div>
          <h3 className="text-lg font-semibold text-ink-200 mb-1">
            {isDragOver ? 'Drop files here…' : 'Drag & drop research papers'}
          </h3>
          <p className="text-sm text-ink-50 mb-6">PDF, DOCX, or LaTeX — up to 50 MB each</p>
          <button className="btn-outline">
            <File className="w-4 h-4" />
            Browse Files
          </button>
        </div>
      </div>

      {/* Extraction Capabilities */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 animate-fade-in-up stagger-2">
        {[
          { icon: FileText, label: 'Sections & Text', count: '—' },
          { icon: Image,    label: 'Figures',         count: '—' },
          { icon: Table2,   label: 'Tables',          count: '—' },
          { icon: Code2,    label: 'Code Blocks',     count: '—' },
        ].map((cap) => {
          const Icon = cap.icon;
          return (
            <div key={cap.label} className="clay-card p-5 text-center">
              <Icon className="w-6 h-6 mx-auto mb-2 text-magenta-400" />
              <p className="text-sm font-medium text-ink-200">{cap.label}</p>
              <p className="text-2xl font-serif text-ink-300 mt-1">{cap.count}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
