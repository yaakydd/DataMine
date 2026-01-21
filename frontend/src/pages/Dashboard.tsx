export default function Dashboard() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
      <p className="text-muted-foreground">Welcome to FairData. Manage your datasets and view insights here.</p>
      
      <div className="grid gap-4 md:grid-cols-3">
          <div className="p-6 rounded-xl border bg-card text-card-foreground shadow">
              <div className="text-2xl font-bold">0</div>
              <p className="text-xs text-muted-foreground">Uploaded Datasets</p>
          </div>
           {/* Add more stats placeholders */}
      </div>
    </div>
  );
}
