import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Layout from './components/Layout';
import Dashboard from './routes/Dashboard';
import Discovery from './routes/Discovery';
import Documents from './routes/Documents';
import RagConsole from './routes/RagConsole';
import KnowledgeGraph from './routes/KnowledgeGraph';
import DraftEditor from './routes/DraftEditor';
import ReviewSimulator from './routes/ReviewSimulator';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Navigate to="/dashboard" replace />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="discovery" element={<Discovery />} />
            <Route path="documents" element={<Documents />} />
            <Route path="rag" element={<RagConsole />} />
            <Route path="graph" element={<KnowledgeGraph />} />
            <Route path="draft" element={<DraftEditor />} />
            <Route path="review" element={<ReviewSimulator />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
