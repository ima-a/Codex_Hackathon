import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navigation from './components/Navigation';
import Landing from './pages/Landing';
import Dashboard from './pages/Dashboard';
import InventoryManager from './pages/InventoryManager';
import AgentBehavior from './pages/AgentBehavior';
import Reconciliation from './pages/Reconciliation';

function App() {
    return (
        <Router>
            <div className="min-h-screen bg-gray-50">
                <Navigation />
                <Routes>
                    <Route path="/" element={<Landing />} />
                    <Route path="/inventory-manager" element={<InventoryManager />} />
                    <Route path="/dashboard" element={<Dashboard />} />
                    <Route path="/agents" element={<AgentBehavior />} />
                    <Route path="/reconciliation" element={<Reconciliation />} />
                </Routes>
            </div>
        </Router>
    );
}

export default App;
