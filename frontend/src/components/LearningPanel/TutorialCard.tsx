import { Lightbulb, X } from 'lucide-react';

interface TutorialCardProps {
  tutorial: {
    title: string;
    explanation: string;
    example?: string;
    strategies?: string[];
  };
  onClose: () => void;
}

export default function TutorialCard({ tutorial, onClose }: TutorialCardProps) {
  return (
    <div className="bg-card border rounded-xl p-4 shadow-lg animate-in slide-in-from-right-4">
      <div className="flex items-start justify-between mb-2">
        <h4 className="font-bold flex items-center gap-2">
          <Lightbulb className="w-4 h-4 text-yellow-500" />
          {tutorial.title}
        </h4>
        <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
          <X className="w-4 h-4" />
        </button>
      </div>
      
      <p className="text-sm text-muted-foreground mb-3 leading-relaxed">
        {tutorial.explanation}
      </p>

      {tutorial.example && (
        <div className="bg-muted/50 p-3 rounded-lg text-xs mb-3">
          <span className="font-semibold block mb-1">Example:</span>
          {tutorial.example}
        </div>
      )}

      {tutorial.strategies && (
        <div className="space-y-1">
          <span className="text-xs font-semibold">Common Strategies:</span>
          <ul className="text-xs list-disc pl-4 text-muted-foreground">
            {tutorial.strategies.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
