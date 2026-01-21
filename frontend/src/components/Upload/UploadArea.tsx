import { useState, useRef } from 'react';
import { UploadCloud, File, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';
import { cn } from '../../utils/cn';

interface UploadResponse {
  success: boolean;
  message: string;
  dataset_id: string;
  info: any;
  explanation: string;
}

interface UploadAreaProps {
  onUploadSuccess: (data: UploadResponse) => void;
}

export default function UploadArea({ onUploadSuccess }: UploadAreaProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const validateAndSetFile = (selectedFile: File) => {
    setError(null);
    const validTypes = [
      'text/csv', 
      'application/vnd.ms-excel', 
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
      'application/json'
    ];
    
    // Simple extension check as fallback
    const validExtensions = ['.csv', '.xlsx', '.xls', '.json', '.parquet'];
    const fileExtension = '.' + selectedFile.name.split('.').pop()?.toLowerCase();

    if (!validTypes.includes(selectedFile.type) && !validExtensions.includes(fileExtension)) {
      setError("Unsupported file format. Please upload CSV, Excel, JSON or Parquet.");
      return;
    }

    setFile(selectedFile);
  };

  const handleUpload = async () => {
    if (!file) return;

    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail?.explanation || "Upload failed");
      }

      onUploadSuccess(data);
    } catch (err: any) {
      setError(err.message || "Failed to upload file");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="w-full max-w-xl mx-auto space-y-4">
      <div 
        className={cn(
          "relative border-2 border-dashed rounded-xl p-8 transition-all duration-200 ease-in-out flex flex-col items-center justify-center gap-4 text-center cursor-pointer",
          isDragging ? "border-primary bg-primary/5 scale-[1.02]" : "border-muted-foreground/25 hover:border-primary/50 hover:bg-muted/50",
          file ? "bg-muted/30 border-primary/50" : ""
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input 
          type="file" 
          ref={fileInputRef} 
          className="hidden" 
          accept=".csv,.xlsx,.xls,.json,.parquet"
          onChange={handleFileSelect}
        />
        
        {file ? (
          <div className="flex flex-col items-center animate-in fade-in zoom-in duration-300">
            <div className="p-4 rounded-full bg-primary/10 text-primary mb-2">
              <File className="w-8 h-8" />
            </div>
            <p className="font-medium text-lg">{file.name}</p>
            <p className="text-sm text-muted-foreground">{(file.size / 1024).toFixed(1)} KB</p>
            <button 
              onClick={(e) => { e.stopPropagation(); setFile(null); }}
              className="mt-4 text-xs text-destructive hover:underline"
            >
              Remove file
            </button>
          </div>
        ) : (
          <>
            <div className="p-4 rounded-full bg-muted text-muted-foreground">
              <UploadCloud className="w-8 h-8" />
            </div>
            <div>
              <p className="font-semibold text-lg">Click to upload or drag and drop</p>
              <p className="text-sm text-muted-foreground mt-1">
                CSV, Excel, JSON or Parquet (max 50MB)
              </p>
            </div>
          </>
        )}
      </div>

      {error && (
        <div className="p-4 rounded-lg bg-destructive/10 text-destructive flex items-center gap-3 animate-in slide-in-from-top-2">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <p className="text-sm font-medium">{error}</p>
        </div>
      )}

      {file && !isUploading && (
        <button
          onClick={handleUpload}
          className="w-full py-4 rounded-xl bg-primary text-primary-foreground font-bold hover:bg-primary/90 transition-all flex items-center justify-center gap-2 shadow-lg shadow-primary/20"
        >
          <CheckCircle2 className="w-5 h-5" />
          Analyze Data
        </button>
      )}

      {isUploading && (
        <div className="w-full py-4 rounded-xl bg-muted text-muted-foreground font-bold flex items-center justify-center gap-2 cursor-wait">
          <Loader2 className="w-5 h-5 animate-spin" />
          Uploading & Analyzing...
        </div>
      )}
    </div>
  );
}
