import { useState, useEffect } from 'react';
import { FolderOpen, Plus, Search } from 'lucide-react';
import { api } from '../../utils/api';
import DatasetItem from './DatasetItem';
import { cn } from '../../utils/cn';

interface WorkspaceSidebarProps {
    className?: string;
}

export default function WorkspaceSidebar({ className }: WorkspaceSidebarProps) {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDatasets();
  }, []);

  const fetchDatasets = async () => {
    try {
        const data = await api.listDatasets();
        setDatasets(data);
    } catch (err) {
        console.error("Failed to load datasets", err);
    } finally {
        setLoading(false);
    }
  };

  return (
    <aside className={cn("w-64 bg-card border-r flex flex-col h-full", className)}>
        <div className="p-4 border-b space-y-4">
            <div className="flex items-center justify-between">
                <h2 className="font-bold flex items-center gap-2">
                    <FolderOpen className="w-5 h-5 text-primary" />
                    Workspace
                </h2>
                <button className="p-1 hover:bg-muted rounded">
                    <Plus className="w-4 h-4" />
                </button>
            </div>
            
            <div className="relative">
                <Search className="absolute left-2 top-2.5 w-4 h-4 text-muted-foreground" />
                <input 
                    type="text" 
                    placeholder="Search data..." 
                    className="w-full bg-muted/50 rounded-md pl-8 pr-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
                />
            </div>
        </div>

        <div className="flex-1 overflow-y-auto p-2 space-y-1">
            {loading && <div className="text-center p-4 text-xs text-muted-foreground">Loading...</div>}
            
            {!loading && datasets.length === 0 && (
                <div className="text-center p-8 text-muted-foreground">
                    <p className="text-sm">No datasets yet.</p>
                    <p className="text-xs mt-1">Upload one to get started.</p>
                </div>
            )}

            {datasets.map((d) => (
                <DatasetItem key={d.dataset_id} dataset={d} />
            ))}
        </div>
        
        <div className="p-4 border-t text-xs text-center text-muted-foreground">
            {datasets.length} Active Datasets
        </div>
    </aside>
  );
}
