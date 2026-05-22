import { useEditor, EditorContent } from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import Placeholder from '@tiptap/extension-placeholder';
import { Bold, Italic, Strikethrough, List, ListOrdered, Undo, Redo, Sparkles, Save } from 'lucide-react';

export default function DraftEditor() {
  const editor = useEditor({
    extensions: [
      StarterKit,
      Placeholder.configure({
        placeholder: 'Begin writing your research paper…',
      }),
    ],
    content: '',
    editorProps: {
      attributes: {
        class: 'focus:outline-none min-h-[400px] text-ink-200 leading-relaxed',
      },
    },
  });

  const toolbarButtons = [
    { icon: Bold,          action: () => editor?.chain().focus().toggleBold().run(),          active: editor?.isActive('bold') },
    { icon: Italic,        action: () => editor?.chain().focus().toggleItalic().run(),        active: editor?.isActive('italic') },
    { icon: Strikethrough, action: () => editor?.chain().focus().toggleStrike().run(),        active: editor?.isActive('strike') },
    { icon: List,          action: () => editor?.chain().focus().toggleBulletList().run(),    active: editor?.isActive('bulletList') },
    { icon: ListOrdered,   action: () => editor?.chain().focus().toggleOrderedList().run(),   active: editor?.isActive('orderedList') },
    null, // divider
    { icon: Undo,          action: () => editor?.chain().focus().undo().run() },
    { icon: Redo,          action: () => editor?.chain().focus().redo().run() },
  ];

  return (
    <div className="space-y-8">
      <div className="flex items-start justify-between animate-fade-in-up">
        <div>
          <h1 className="text-4xl font-serif text-ink-300 mb-2">Draft Editor</h1>
          <p className="text-ink-50 max-w-xl">
            Compose your IEEE-structured paper with AI-assisted section generation,
            inline citation insertion, and real-time review feedback.
          </p>
        </div>
        <div className="flex gap-2">
          <button className="btn-outline text-sm">
            <Sparkles className="w-4 h-4" />
            Generate Section
          </button>
          <button className="btn-magenta text-sm">
            <Save className="w-4 h-4" />
            Save Draft
          </button>
        </div>
      </div>

      {/* Editor Card */}
      <div className="clay-card p-0 overflow-hidden animate-fade-in-up stagger-1">
        {/* Toolbar */}
        <div className="flex items-center gap-1 px-4 py-2.5 border-b border-cream-300/50 bg-cream-200/30">
          {toolbarButtons.map((btn, i) => {
            if (!btn) {
              return <div key={i} className="w-px h-5 bg-cream-400 mx-1" />;
            }
            const Icon = btn.icon;
            return (
              <button
                key={i}
                onClick={btn.action}
                className={`p-2 rounded-lg transition-all duration-200 ${
                  btn.active
                    ? 'bg-magenta-100 text-magenta-600'
                    : 'text-ink-50 hover:bg-cream-200 hover:text-ink-200'
                }`}
              >
                <Icon className="w-4 h-4" />
              </button>
            );
          })}
        </div>

        {/* Writing Surface */}
        <div className="px-12 py-10 bg-white min-h-[500px] font-serif text-lg">
          <EditorContent editor={editor} />
        </div>
      </div>
    </div>
  );
}
