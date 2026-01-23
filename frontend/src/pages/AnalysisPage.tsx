import { useEffect, useState, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Loader2 } from 'lucide-react';
import AnalysisDashboard from '../components/Visualization/AnalysisDashboard';
import QuickActionsPanel from '../components/QuickActions/QuickActionsPanel';
import HistoryTimeline from '../components/History/HistoryTimeline';
import { api } from '../utils/api';

export default function AnalysisPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [analysisData, setAnalysisData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const dataset = location.state?.dataset;

  const fetchAnalysis = useCallback(async () => {
    if (!dataset?.dataset_id) return;
    try {
      setLoading(true);
      const data = await api.analyzeDataset(dataset.dataset_id);
      setAnalysisData(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [dataset?.dataset_id]);

  useEffect(() => {
    if (!dataset?.dataset_id) {
       setLoading(false);
       return;
    }
    fetchAnalysis();
  }, [dataset, fetchAnalysis]);

  const handleActionComplete = (result: any) => {
      alert(result.result.message || "Action completed!");
      if (result.action === 'clean') {
          // Re-analyze data to update charts
          fetchAnalysis();
      }
  };

  if (!dataset?.dataset_id) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4 text-center">
        <h2 className="text-2xl font-bold">No Dataset Selected</h2>
        <p className="text-muted-foreground">Please upload a dataset to begin analysis.</p>
        <button 
          onClick={() => navigate('/upload')}
          className="px-6 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
        >
          Go to Upload
        </button>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
        <Loader2 className="w-12 h-12 animate-spin text-primary" />
        <h2 className="text-xl font-medium">Analyzing your data...</h2>
        <p className="text-muted-foreground text-sm max-w-md text-center">
          Our AI is checking for patterns, calculating statistics, and generating visualizations.
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 border border-destructive/20 bg-destructive/5 rounded-xl text-center space-y-4">
        <h2 className="text-xl font-bold text-destructive">Analysis Failed</h2>
        <p className="text-muted-foreground">{error}</p>
        <button 
           onClick={() => window.location.reload()}
           className="px-4 py-2 bg-background border hover:bg-muted rounded-lg transition-colors"
        >
          Try Again
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div className="flex items-center justify-between">
        <div>
           <h1 className="text-3xl font-bold tracking-tight">Analysis Results</h1>
           <p className="text-muted-foreground">{dataset.info.filename} • {dataset.info.size}</p>
        </div>
      </div>

      <QuickActionsPanel 
        datasetId={dataset.dataset_id} 
        onActionComplete={handleActionComplete} 
      />

      {analysisData && (
        <AnalysisDashboard 
          report={analysisData.report} 
          visualizations={analysisData.visualizations.visualization_data} 
        />
      )}

      <HistoryTimeline datasetId={dataset.dataset_id} />
    </div>
  );
}
