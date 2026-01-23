const API_BASE = '/api';

export const api = {
  // Datasets
  uploadDataset: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail?.explanation || data.detail || 'Upload failed');
    return data;
  },

  analyzeDataset: async (id: string) => {
    const res = await fetch(`${API_BASE}/analyze/${id}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Analysis failed');
    return data;
  },

  // Learning
  getTutorial: async (concept: string) => {
    const res = await fetch(`${API_BASE}/learning/tutorial/${concept}`);
    return res.json();
  },

  // Workspaces
  createWorkspace: async (name: string) => {
    const res = await fetch(`${API_BASE}/workspaces?name=${name}`, { method: 'POST' });
    return res.json();
  },

  // History
  getHistory: async (datasetId: string) => {
    const res = await fetch(`${API_BASE}/history/${datasetId}`);
    return res.json();
  },

  // Quick Actions
  quickClean: async (datasetId: string) => {
    const res = await fetch(`${API_BASE}/quick-actions/auto-clean/${datasetId}`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Auto-clean failed');
    return data;
  },

  getQuickSummary: async (datasetId: string) => {
    const res = await fetch(`${API_BASE}/quick-actions/summary/${datasetId}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Summary generation failed');
    return data;
  },

  // Workspace / Datasets
  listDatasets: async () => {
    const res = await fetch(`${API_BASE}/datasets/`);
    return res.json();
  },


};
