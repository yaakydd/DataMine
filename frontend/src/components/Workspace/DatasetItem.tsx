import { FileSpreadsheet, Trash2 } from 'lucide-react';
import { cn } from '../../utils/cn';
import { useNavigate } from 'react-router-dom';

interface DatasetItemProps {
  dataset: {
    dataset_id: string;
    filename: string;
    uploaded_at: string;
    size_mb: number;
    rows: number;
  };
  isActive?: boolean;
}

export default function DatasetItem({ dataset, isActive }: DatasetItemProps) {
  const navigate = useNavigate();

  const handleClick = () => {
      navigate('/analysis', { state: { dataset: { dataset_id: dataset.dataset_id, info: { filename: dataset.filename, size: `${dataset.size_mb.toFixed(2)} MB` } } } }); 
  };

  return (
    <div 
        onClick={handleClick}
        className={cn(
            "group flex items-center gap-3 p-2 rounded-lg cursor-pointer transition-colors border border-transparent",
            isActive ? "bg-primary/10 border-primary/20" : "hover:bg-muted"
        )}
    >
      <div className="bg-green-500/10 p-2 rounded text-green-600">
        <FileSpreadsheet className="w-4 h-4" />
      </div>
      <div className="flex-1 min-w-0">
        <h4 className="font-medium text-sm truncate" title={dataset.filename}>
            {dataset.filename}
        </h4>
        <p className="text-xs text-muted-foreground truncate">
            {(dataset.size_mb).toFixed(2)} MB • {dataset.rows} rows
        </p>
      </div>
      <button className="opacity-0 group-hover:opacity-100 p-1 hover:text-destructive transition-opacity">
          <Trash2 className="w-4 h-4" />
      </button>
    </div>
  );
}
