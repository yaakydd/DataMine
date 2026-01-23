import { useState } from 'react';
import { Eraser, BarChart3, Sparkles, Loader2 } from 'lucide-react';
import { api } from '../../utils/api';
import { cn } from '../../utils/cn';

interface QuickActionsPanelProps {
  datasetId: string;
  onActionComplete: (result: any) => void;
}

export default function QuickActionsPanel({ datasetId, onActionComplete }: QuickActionsPanelProps) {
  const [loading, setLoading] = useState<string | null>(null);

  const handleAction = async (action: string) => {
    try {
      setLoading(action);
      let result;
      
      if (action === 'clean') {
        result = await api.quickClean(datasetId);
      } else if (action === 'summary') {
        result = await api.getQuickSummary(datasetId);
      }
      
      onActionComplete({ action, result });
    } catch (err) {
      console.error(err);
      alert("Action failed: " + (err as Error).message);
    } finally {
      setLoading(null);
    }
  };

  return (
    <div className="bg-card border rounded-xl p-6 shadow-sm space-y-4">
      <h3 className="text-xl font-bold flex items-center gap-2">
        <Sparkles className="w-5 h-5 text-yellow-500" />
        Quick Actions
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ActionButton 
          icon={Eraser} 
          label="Auto-Clean Data" 
          description="Remove duplicates & empty rows"
          onClick={() => handleAction('clean')}
          isLoading={loading === 'clean'}
          color="text-blue-500"
          bg="bg-blue-500/10"
        />
        <ActionButton 
            icon={BarChart3} 
            label="Generate Summary" 
            description="Get key statistics instantly"
            onClick={() => handleAction('summary')}
            isLoading={loading === 'summary'}
            color="text-purple-500"
            bg="bg-purple-500/10"
        />
      </div>
    </div>
  );
}

function ActionButton({ icon: Icon, label, description, onClick, isLoading, color, bg }: any) {
    return (
        <button 
            onClick={onClick}
            disabled={isLoading}
            className={cn(
                "flex items-start gap-3 p-4 rounded-lg border hover:bg-muted/50 transition-all text-left",
                isLoading && "opacity-70 cursor-not-allowed"
            )}
        >
            <div className={cn("p-2 rounded-md", bg, color)}>
                {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Icon className="w-5 h-5" />}
            </div>
            <div>
                <span className="font-semibold block">{label}</span>
                <span className="text-xs text-muted-foreground">{description}</span>
            </div>
        </button>
    )
}
