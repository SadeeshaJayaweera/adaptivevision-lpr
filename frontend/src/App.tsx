import React, { useState, useEffect } from 'react';

// Basic Mock Dashboard Interface
function App() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/api/health')
      .then(res => res.json())
      .then(data => setHealth(data))
      .catch(err => console.error(err));
  }, []);

  return (
    <div style={{ padding: '2rem', fontFamily: 'sans-serif', backgroundColor: '#1e1e1e', color: '#fff', minHeight: '100vh' }}>
      <h1>AdaptiveVision-LPR Dashboard</h1>
      <p>Research-grade, uncertainty-aware AI vehicle license plate recognition platform.</p>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginTop: '2rem' }}>
        <div style={{ border: '1px solid #333', padding: '1rem', borderRadius: '8px' }}>
          <h2>Live Cameras</h2>
          <p>Camera 01: 🟢 Active</p>
          <div style={{ backgroundColor: '#333', padding: '1rem', borderRadius: '4px' }}>
             <p>Vehicle #184</p>
             <h3 style={{ color: '#4ade80' }}>WP CAA-1234</h3>
             <p>Confidence: 94.7%</p>
             <p>Condition: Heavy Rain + Low Light</p>
             <p>Occlusion: 8%</p>
             <p>Decision: <strong>ACCEPT</strong></p>
          </div>
        </div>

        <div style={{ border: '1px solid #333', padding: '1rem', borderRadius: '8px' }}>
          <h2>System Telemetry</h2>
          <p>Backend Status: {health ? health.status : 'Disconnected'}</p>
          <p>API Version: 1.0.0</p>
          <p>Active Nodes: 1 (CPU/MPS)</p>
          <p>Disaster Mode: 🔴 Disabled</p>
        </div>
      </div>
    </div>
  );
}

export default App;
