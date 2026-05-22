import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface Project {
  id: string;
  title: string;
  research_idea: string;
  status: string;
  created_at: string;
}

export const researchApi = {
  getProjects: async (): Promise<Project[]> => {
    const { data } = await api.get('/projects');
    return data;
  },
  createProject: async (payload: { title: string; research_idea: string }): Promise<Project> => {
    const { data } = await api.post('/projects', payload);
    return data;
  },
  triggerDiscovery: async (projectId: string) => {
    const { data } = await api.post(`/projects/${projectId}/discover`);
    return data;
  },
  // Add other endpoints matching the FastAPI backend here...
};

export default api;
