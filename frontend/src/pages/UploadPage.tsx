import { useNavigate } from 'react-router-dom';
import UploadArea from '../components/Upload/UploadArea';

export default function UploadPage() {
  const navigate = useNavigate();

  const handleUploadSuccess = (data: any) => {
    // Navigate to analysis page with the uploaded data
    navigate('/analysis', { state: { dataset: data } });
  };

  return (
    <div className="max-w-4xl mx-auto space-y-12">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-extrabold tracking-tight bg-gradient-to-r from-white to-white/60 bg-clip-text text-transparent">
          Upload Your Dataset
        </h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          We'll automatically clean your data, find hidden patterns, and generate beautiful visualizations. 
          Just drop your file below.
        </p>
      </div>

      <UploadArea onUploadSuccess={handleUploadSuccess} />
      
      <div className="grid md:grid-cols-3 gap-6 pt-12">
          <div className="text-center space-y-2">
              <div className="w-12 h-12 mx-auto rounded-full bg-blue-500/10 flex items-center justify-center text-blue-400 font-bold text-xl">1</div>
              <h3 className="font-bold">Upload</h3>
              <p className="text-sm text-muted-foreground">Drag & drop your CSV or Excel file</p>
          </div>
          <div className="text-center space-y-2">
              <div className="w-12 h-12 mx-auto rounded-full bg-purple-500/10 flex items-center justify-center text-purple-400 font-bold text-xl">2</div>
              <h3 className="font-bold">Process</h3>
              <p className="text-sm text-muted-foreground">We clean and analyze automatically</p>
          </div>
          <div className="text-center space-y-2">
              <div className="w-12 h-12 mx-auto rounded-full bg-green-500/10 flex items-center justify-center text-green-400 font-bold text-xl">3</div>
              <h3 className="font-bold">Discover</h3>
              <p className="text-sm text-muted-foreground">Get insights in plain English</p>
          </div>
      </div>
    </div>
  );
}
