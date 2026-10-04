import React, { useState } from 'react';
import './index.css';

// Mock Data representing the Event Model
const mockAcceptEvent = {
  id: "evt-8f92a",
  camera: "CAM-01 (Highway North)",
  plate: "WP CAA-1234",
  confidence: 0.94,
  status: "ACCEPT",
  conditions: ["HEAVY_RAIN", "LOW_LIGHT"],
  occlusion: 0.08,
  recoverability: 0.96,
  frames: 23,
  pipeline: ["LOW_LIGHT_ENHANCEMENT", "RAIN_ARTIFACT_REDUCTION", "PERSPECTIVE_RECTIFICATION"]
};

const mockUnknownEvent = {
  id: "evt-3b11c",
  camera: "CAM-04 (Junction West)",
  plate: "WP C??-1???",
  confidence: 0.21,
  status: "UNKNOWN",
  reason: "PHYSICAL_OCCLUSION",
  conditions: ["MUD_SPLATTER"],
  occlusion: 0.72,
  recoverability: 0.15,
  frames: 14,
  pipeline: ["PERSPECTIVE_RECTIFICATION", "SUPER_RESOLUTION"]
};

const mockReprocessEvent = {
  id: "evt-9x22d",
  camera: "CAM-02 (Toll Booth A)",
  plate: "CP AB-5582",
  confidence: 0.71,
  status: "REPROCESS",
  reason: "UNCERTAIN_EVIDENCE",
  conditions: ["GLARE", "MOTION_BLUR"],
  occlusion: 0.25,
  recoverability: 0.85,
  frames: 5,
  pipeline: ["GLARE_SUPPRESSION", "MOTION_DEBLUR"]
};

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  
  return (
    <div className="dashboard-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          AdaptiveVision <span style={{color: 'var(--primary)'}}>LPR</span>
        </div>
        
        <nav style={{marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem'}}>
          <div className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>
            <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"></path></svg>
            Live Dashboard
          </div>
          <div className={`nav-item ${activeTab === 'history' ? 'active' : ''}`} onClick={() => setActiveTab('history')}>
            <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
            Detection History
          </div>
          <div className={`nav-item ${activeTab === 'health' ? 'active' : ''}`} onClick={() => setActiveTab('health')}>
            <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
            Camera Health
          </div>
          <div className={`nav-item ${activeTab === 'disaster' ? 'active' : ''}`} onClick={() => setActiveTab('disaster')}>
            <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
            Disaster Mode
          </div>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="header">
          <div>
            <h1 style={{fontSize: '2rem'}}>System Overview</h1>
            <p className="metric-label">Research-grade Uncertainty-Aware Perception</p>
          </div>
          <div style={{display: 'flex', gap: '1rem'}}>
            <div className="status-badge status-danger">Disaster Mode: ACTIVE</div>
            <div className="status-badge status-accept">System Health: 98%</div>
          </div>
        </header>

        {activeTab === 'dashboard' && (
          <div className="animate-slide-in">
            <div className="grid-3">
              {/* ACCEPT EVENT */}
              <div className="glass-panel" style={{padding: '1.5rem'}}>
                <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                  <h3 style={{color: 'var(--text-muted)'}}>{mockAcceptEvent.camera}</h3>
                  <span className="status-badge status-accept">ACCEPT</span>
                </div>
                
                <div className="plate-display" style={{color: 'var(--success)'}}>
                  {mockAcceptEvent.plate}
                </div>
                
                <div style={{marginBottom: '1rem'}}>
                  {mockAcceptEvent.conditions.map(c => <span key={c} className="condition-tag">{c}</span>)}
                </div>

                <div className="metric-row">
                  <span className="metric-label">Confidence</span>
                  <span className="metric-value">{(mockAcceptEvent.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="progress-bar-bg">
                  <div className="progress-bar-fill" style={{width: `${mockAcceptEvent.confidence * 100}%`, background: 'var(--success)'}}></div>
                </div>

                <div className="metric-row" style={{marginTop: '1rem'}}>
                  <span className="metric-label">Recoverability</span>
                  <span className="metric-value">{(mockAcceptEvent.recoverability * 100).toFixed(1)}%</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Frames Used</span>
                  <span className="metric-value">{mockAcceptEvent.frames}</span>
                </div>
              </div>

              {/* UNKNOWN EVENT */}
              <div className="glass-panel" style={{padding: '1.5rem'}}>
                <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                  <h3 style={{color: 'var(--text-muted)'}}>{mockUnknownEvent.camera}</h3>
                  <span className="status-badge status-unknown">UNKNOWN</span>
                </div>
                
                <div className="plate-display">
                  <span className="plate-char">WP C</span>
                  <span className="plate-char unknown">??</span>
                  <span className="plate-char">-1</span>
                  <span className="plate-char unknown">???</span>
                </div>
                
                <div style={{marginBottom: '1rem'}}>
                  {mockUnknownEvent.conditions.map(c => <span key={c} className="condition-tag">{c}</span>)}
                </div>

                <div className="metric-row">
                  <span className="metric-label">Reason</span>
                  <span className="metric-value" style={{color: 'var(--unknown)'}}>{mockUnknownEvent.reason}</span>
                </div>
                
                <div className="metric-row" style={{marginTop: '1rem'}}>
                  <span className="metric-label">Occlusion</span>
                  <span className="metric-value">{(mockUnknownEvent.occlusion * 100).toFixed(1)}%</span>
                </div>
                <div className="progress-bar-bg">
                  <div className="progress-bar-fill" style={{width: `${mockUnknownEvent.occlusion * 100}%`, background: 'var(--unknown)'}}></div>
                </div>
                
                <div className="metric-row" style={{marginTop: '1rem'}}>
                  <span className="metric-label">Recommendation</span>
                  <span className="metric-value">Await additional frame</span>
                </div>
              </div>

              {/* REPROCESS EVENT */}
              <div className="glass-panel" style={{padding: '1.5rem'}}>
                <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                  <h3 style={{color: 'var(--text-muted)'}}>{mockReprocessEvent.camera}</h3>
                  <span className="status-badge status-reprocess">REPROCESS</span>
                </div>
                
                <div className="plate-display">
                  <span className="plate-char">CP AB-55</span>
                  <span className="plate-char uncertain">8</span>
                  <span className="plate-char">2</span>
                </div>
                
                <div style={{marginBottom: '1rem'}}>
                  {mockReprocessEvent.conditions.map(c => <span key={c} className="condition-tag">{c}</span>)}
                </div>

                <div className="metric-row">
                  <span className="metric-label">Confidence</span>
                  <span className="metric-value">{(mockReprocessEvent.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="progress-bar-bg">
                  <div className="progress-bar-fill" style={{width: `${mockReprocessEvent.confidence * 100}%`, background: 'var(--warning)'}}></div>
                </div>

                <div style={{marginTop: '1.5rem', display: 'flex', gap: '0.5rem'}}>
                   <button className="btn" style={{flex: 1}}>Wait for Frames</button>
                   <button className="btn btn-secondary" style={{flex: 1}}>Force Accept</button>
                </div>
              </div>
            </div>

            {/* Reprocessing Trace UI */}
            <div className="grid-2">
              <div className="glass-panel" style={{padding: '1.5rem'}}>
                <h2>Reprocessing Trace</h2>
                <p className="metric-label" style={{marginBottom: '1.5rem'}}>Event {mockReprocessEvent.id}</p>
                
                <div className="timeline">
                  <div className="timeline-item">
                    <h4>Attempt 1: Original Frame</h4>
                    <p className="metric-label">Confidence: 42% (Insufficient)</p>
                  </div>
                  <div className="timeline-item">
                    <h4>Attempt 2: Glare Suppression</h4>
                    <p className="metric-label">Confidence: 63% (Insufficient)</p>
                  </div>
                  <div className="timeline-item">
                    <h4>Attempt 3: Motion Deblur + Super Resolution</h4>
                    <p className="metric-label">Confidence: 71% (Borderline)</p>
                  </div>
                  <div className="timeline-item" style={{opacity: 0.5}}>
                    <h4>Attempt 4: Temporal Fusion (Pending)</h4>
                    <p className="metric-label">Waiting for frames 6-15...</p>
                  </div>
                </div>
              </div>
              
              <div className="glass-panel" style={{padding: '1.5rem'}}>
                <h2>Camera Health</h2>
                <p className="metric-label" style={{marginBottom: '1.5rem'}}>Real-time Diagnostics</p>
                
                <div className="metric-row">
                  <span className="metric-label">CAM-01 (Highway North)</span>
                  <span className="status-badge status-accept" style={{padding: '0.25rem 0.75rem', fontSize: '0.7rem'}}>99%</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">CAM-02 (Toll Booth A)</span>
                  <span className="status-badge status-reprocess" style={{padding: '0.25rem 0.75rem', fontSize: '0.7rem'}}>74%</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">CAM-03 (Toll Booth B)</span>
                  <span className="status-badge status-accept" style={{padding: '0.25rem 0.75rem', fontSize: '0.7rem'}}>92%</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">CAM-04 (Junction West)</span>
                  <span className="status-badge status-danger" style={{padding: '0.25rem 0.75rem', fontSize: '0.7rem'}}>41%</span>
                </div>
                
                <div style={{marginTop: '2rem', padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '8px', border: '1px solid rgba(239, 68, 68, 0.3)'}}>
                  <h4 style={{color: 'var(--danger)', marginBottom: '0.5rem'}}>Warning: CAM-04</h4>
                  <p className="metric-label" style={{fontSize: '0.8rem'}}>High frame drop rate (32%) detected. Suspected network congestion or physical lens obstruction.</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
