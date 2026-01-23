import { useEffect, useState } from 'react';
import { History, Clock } from 'lucide-react';
import { api } from '../../utils/api';
import { cn } from '../../utils/cn';

interface HistoryTimelineProps {
  datasetId: string;
  className?: string;
}

export default function HistoryTimeline({ datasetId, className }: HistoryTimelineProps) {
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchHistory();
  }, [datasetId]);

  const fetchHistory = async () => {
    try {
        const data = await api.getHistory(datasetId);
        // Sort by newest first
        setHistory(data.reverse());
    } catch (err) {
        console.error("Failed to load history", err);
    } finally {
        setLoading(false);
    }
  };

  if (loading) return <div className="p-6 text-center text-muted-foreground animate-pulse">Loading history...</div>;
  
  if (history.length === 0) {
      return (
        <div className={cn("bg-card border rounded-xl p-6", className)}>
            <h3 className="text-lg font-bold flex items-center gap-2 mb-4">
                <History className="w-5 h-5 text-primary" />
                Version History
            </h3>
            <div className="text-center text-muted-foreground py-8 text-sm">
                No history recorded for this dataset yet.
            </div>
        </div>
      );
  }

  return (
    <div className={cn("bg-card border rounded-xl p-6", className)}>
      <h3 className="text-lg font-bold flex items-center gap-2 mb-4">
        <History className="w-5 h-5 text-primary" />
        Version History
      </h3>

      <div className="space-y-6 relative ml-2 border-l border-border pl-6">
        {history.map((entry, index) => (
            <div key={entry.id || index} className="relative">
                {/* Dot */}
                <div className="absolute -left-[29px] top-1 w-3 h-3 rounded-full bg-primary ring-4 ring-background" />
                
                <div className="flex flex-col">
                    <span className="font-semibold text-sm">{entry.action}</span>
                    <span className="text-xs text-muted-foreground flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {new Date(entry.timestamp).toLocaleString()}
                    </span>
                    
                    {entry.details && Object.keys(entry.details).length > 0 && (
                        <div className="mt-2 bg-muted/50 p-2 rounded text-xs space-y-1">
                            {Object.entries(entry.details).map(([key, value]) => (
                                <div key={key} className="flex justify-between gap-4">
                                    <span className="text-muted-foreground capitalize">{key.replace(/_/g, ' ')}:</span>
                                    <span className="font-mono">{String(value)}</span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        ))}
      </div>
    </div>
  );
}
