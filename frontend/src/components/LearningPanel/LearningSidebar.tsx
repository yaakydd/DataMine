import { useState, useEffect } from 'react';
import { BookOpen, ChevronRight, GraduationCap } from 'lucide-react';
import { cn } from '../../utils/cn';
import { api } from '../../utils/api';
import TutorialCard from './TutorialCard';

interface LearningSidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  currentContext?: string; // e.g., 'missing_values', 'outliers'
}

export default function LearningSidebar({ isOpen, onToggle, currentContext }: LearningSidebarProps) {
  const [activeTutorial, setActiveTutorial] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  // Auto-fetch tutorial when context changes
  useEffect(() => {
    if (currentContext && isOpen) {
        fetchTutorial(currentContext);
    }
  }, [currentContext, isOpen]);

  const fetchTutorial = async (concept: string) => {
    try {
        setLoading(true);
        const data = await api.getTutorial(concept);
        setActiveTutorial(data);
    } catch (err) {
        console.error("Failed to fetch tutorial", err);
    } finally {
        setLoading(false);
    }
  };

  return (
    <aside 
      className={cn(
        "fixed right-0 top-0 h-full bg-card border-l border-border transition-all duration-300 z-30 shadow-xl flex flex-col",
        isOpen ? "w-80 translate-x-0" : "w-12 translate-x-full"
      )}
    >
        {/* Toggle Button */}
        <button 
            onClick={onToggle}
            className="absolute -left-12 top-4 bg-card border border-border p-2 rounded-l-lg shadow-md hover:bg-muted transition-colors"
        >
            {isOpen ? <ChevronRight className="w-5 h-5" /> : <BookOpen className="w-5 h-5 text-primary" />}
        </button>

        {isOpen && (
            <div className="flex-1 flex flex-col p-4 overflow-hidden">
                <div className="flex items-center gap-2 mb-6 border-b pb-4">
                    <GraduationCap className="w-6 h-6 text-primary" />
                    <h2 className="font-bold text-lg">Learning Goals</h2>
                </div>

                <div className="flex-1 overflow-y-auto space-y-4">
                    <p className="text-sm text-muted-foreground">
                        As you work with your data, I'll explain key concepts here.
                    </p>

                    {loading && (
                        <div className="animate-pulse space-y-3">
                            <div className="h-24 bg-muted rounded-lg w-full"></div>
                        </div>
                    )}

                    {!loading && activeTutorial && (
                        <TutorialCard 
                            tutorial={activeTutorial} 
                            onClose={() => setActiveTutorial(null)} 
                        />
                    )}

                    {/* Default content if no active tutorial */}
                    {!loading && !activeTutorial && (
                        <div className="space-y-2">
                           <h3 className="font-semibold text-sm">Recommended Topics</h3>
                           <button onClick={() => fetchTutorial('outliers')} className="w-full text-left text-sm p-3 rounded hover:bg-muted border border-transparent hover:border-border transition-all">
                                📉 Understanding Outliers
                           </button>
                           <button onClick={() => fetchTutorial('missing_values')} className="w-full text-left text-sm p-3 rounded hover:bg-muted border border-transparent hover:border-border transition-all">
                                🔍 Handling Missing Data
                           </button>
                        </div>
                    )}
                </div>
            </div>
        )}
    </aside>
  );
}
