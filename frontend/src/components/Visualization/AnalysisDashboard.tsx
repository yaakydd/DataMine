import { useState } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import { ThumbsUp, TrendingUp } from 'lucide-react';
import { cn } from '../../utils/cn';

interface AnalysisDashboardProps {
  report: any;
  visualizations: any;
}

export default function AnalysisDashboard({ report, visualizations }: AnalysisDashboardProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'quality' | 'columns' | 'charts'>('overview');

  const qualityScore = report.data_quality_score?.overall_score || 0;
  const qualityGrade = report.data_quality_score?.grade || 'N/A';

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      {/* Tab Navigation */}
      <div className="flex space-x-1 rounded-xl bg-muted p-1">
        {['overview', 'quality', 'charts'].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab as any)}
            className={cn(
              "w-full rounded-lg py-2.5 text-sm font-medium leading-5 transition-all duration-200",
              activeTab === tab
                ? "bg-background text-primary shadow"
                : "text-muted-foreground hover:bg-background/50 hover:text-foreground"
            )}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Content Area */}
      <div className="min-h-[400px]">
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Quick Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard label="Rows" value={report.dataset_overview.total_rows} />
              <StatCard label="Columns" value={report.dataset_overview.total_columns} />
              <StatCard label="Numeric" value={report.dataset_overview.numeric_columns} />
              <StatCard label="Categorical" value={report.dataset_overview.categorical_columns} />
            </div>

            {/* Plain English Explanation */}
            <div className="bg-card border rounded-xl p-6 shadow-sm space-y-4">
              <h3 className="text-xl font-bold flex items-center gap-2">
                <ThumbsUp className="w-5 h-5 text-primary" />
                What is this dataset?
              </h3>
              <p className="text-muted-foreground leading-relaxed whitespace-pre-line">
                {report.dataset_overview.explanation}
              </p>
            </div>

            {/* Key Insights */}
            {report.insights && report.insights.length > 0 && (
              <div className="bg-card border rounded-xl p-6 shadow-sm space-y-4">
                <h3 className="text-xl font-bold flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-green-500" />
                  Key Insights
                </h3>
                <div className="grid gap-4">
                  {report.insights.map((insight: any, idx: number) => (
                    <div key={idx} className={cn("p-4 rounded-lg border-l-4", 
                      insight.type === 'warning' ? "bg-yellow-500/10 border-yellow-500" : "bg-blue-500/10 border-blue-500"
                    )}>
                      <h4 className="font-bold text-sm">{insight.title}</h4>
                      <p className="text-sm mt-1">{insight.message}</p>
                      <p className="text-xs text-muted-foreground mt-2">{insight.explanation}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'quality' && (
          <div className="space-y-8">
             <div className="flex items-center justify-center py-8">
                <div className="relative w-48 h-48 flex items-center justify-center">
                    <svg className="w-full h-full transform -rotate-90">
                        <circle cx="96" cy="96" r="88" stroke="currentColor" strokeWidth="12" fill="transparent" className="text-muted" />
                        <circle cx="96" cy="96" r="88" stroke="currentColor" strokeWidth="12" fill="transparent" 
                            strokeDasharray={2 * Math.PI * 88}
                            strokeDashoffset={2 * Math.PI * 88 * (1 - qualityScore / 100)}
                            className={cn(
                                qualityScore > 80 ? "text-green-500" : qualityScore > 60 ? "text-yellow-500" : "text-red-500",
                                "transition-all duration-1000 ease-out"
                            )}
                         />
                    </svg>
                    <div className="absolute flex flex-col items-center">
                        <span className="text-5xl font-bold">{qualityScore}</span>
                        <span className="text-sm font-medium text-muted-foreground">Grade: {qualityGrade}</span>
                    </div>
                </div>
             </div>

             <div className="bg-muted/30 rounded-xl p-6 whitespace-pre-line leading-relaxed">
                 {report.data_quality_score.explanation}
             </div>
          </div>
        )}

        {activeTab === 'charts' && (
          <div className="space-y-8">
            {/* Histograms */}
            {visualizations?.histograms && Object.keys(visualizations.histograms).length > 0 && (
              <div>
                <h3 className="text-lg font-bold mb-4">Distributions (Histograms)</h3>
                <div className="grid md:grid-cols-2 gap-6">
                  {Object.entries(visualizations.histograms).map(([col, data]: [string, any]) => (
                     <ChartCard key={col} title={`Distribution of ${col}`}>
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={data.counts.map((count: number, i: number) => ({
                                bin: data.bin_edges[i]?.toFixed(1),
                                count
                            }))}>
                                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                                <XAxis dataKey="bin" tick={{fontSize: 12}} />
                                <YAxis />
                                <Tooltip cursor={{fill: 'transparent'}} contentStyle={{ borderRadius: '8px', border: 'none', background: '#333', color: '#fff' }} />
                                <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                     </ChartCard>
                  ))}
                </div>
              </div>
            )}
             
            {/* Bar Charts for Categorical */}
            {visualizations?.bar_charts && Object.keys(visualizations.bar_charts).length > 0 && (
               <div>
                <h3 className="text-lg font-bold mb-4 mt-8">Categories (Top 10)</h3>
                <div className="grid md:grid-cols-2 gap-6">
                  {Object.entries(visualizations.bar_charts).map(([col, data]: [string, any]) => (
                     <ChartCard key={col} title={`Counts for ${col}`}>
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart layout="vertical" data={data.categories.map((cat: string, i: number) => ({
                                category: cat,
                                count: data.counts[i]
                            }))}>
                                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                                <XAxis type="number" />
                                <YAxis dataKey="category" type="category" width={100} tick={{fontSize: 11}} />
                                <Tooltip cursor={{fill: 'transparent'}} contentStyle={{ borderRadius: '8px', border: 'none', background: '#333', color: '#fff' }} />
                                <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                     </ChartCard>
                  ))}
                </div>
               </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string, value: string | number }) {
  return (
    <div className="bg-card border rounded-xl p-4 shadow-sm">
      <p className="text-xs font-medium text-muted-foreground uppercase">{label}</p>
      <p className="text-2xl font-bold mt-1">{value}</p>
    </div>
  );
}

function ChartCard({ title, children }: { title: string, children: React.ReactNode }) {
    return (
        <div className="h-80 bg-card border rounded-xl p-4 flex flex-col">
            <h4 className="text-sm font-medium mb-4 text-muted-foreground">{title}</h4>
            <div className="flex-1 min-h-0">
                {children}
            </div>
        </div>
    )
}
