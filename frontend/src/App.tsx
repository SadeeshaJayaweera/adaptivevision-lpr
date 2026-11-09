import React, { useState, useRef, useEffect, useCallback } from 'react';
import './index.css';

interface InferenceResult {
  event_id: string;
  plate: string;
  raw_ocr: string;
  status: 'ACCEPT' | 'REPROCESS' | 'UNKNOWN';
  confidence: number;
  format_valid: boolean;
  validation_reason: string | null;
  conditions: string[];
  condition_classification: {
    illumination: string;
    weather: string;
    motion: string;
    optics: string;
  };
  quality_metrics: {
    blur: number;
    brightness: number;
    glare: number;
    contrast: number;
    noise: number;
  };
  occlusion_analysis: {
    visibility: number;
    occlusion: number;
    occlusion_type: string;
    visible_character_regions: number;
  };
  recoverability: {
    class: string;
    confidence: number;
  };
  adaptive_routing: {
    pipeline_selected: string[];
    steps_executed: string[];
  };
  uncertainty: {
    epistemic: number;
    aleatoric: number;
  };
  original_crop_preview?: string;
  enhanced_plate_preview: string;
  sauvola_crop_preview?: string;
  annotated_frame_preview?: string;
  message: string;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'history' | 'health' | 'disaster'>('dashboard');
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentResult, setCurrentResult] = useState<InferenceResult | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [history, setHistory] = useState<Array<InferenceResult & { timestamp: string; source: string }>>([]);
  const [disasterMode, setDisasterMode] = useState(false);
  const [backendStatus, setBackendStatus] = useState<'online' | 'offline' | 'checking'>('checking');
  const [dragOver, setDragOver] = useState(false);
  const [selectedScenario, setSelectedScenario] = useState<string | null>(null);
  const [sliderPos, setSliderPos] = useState(50);
  const [viewMode, setViewMode] = useState<'slider' | 'side-by-side' | 'sauvola'>('slider');
  const [zoomLevel, setZoomLevel] = useState<'1x' | '2x' | '3x'>('1x');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const API_BASE = (typeof process !== 'undefined' && process.env?.REACT_APP_API_URL) || 'http://localhost:8000';

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (res.ok) {
        setBackendStatus('online');
      } else {
        setBackendStatus('offline');
      }
    } catch {
      setBackendStatus('offline');
    }
  }, [API_BASE]);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  const handleFileUpload = async (file: File) => {
    setIsProcessing(true);
    setPreviewUrl(URL.createObjectURL(file));

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/api/inference/simulate`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Inference API error: ${res.statusText}`);
      }

      const data: InferenceResult = await res.json();
      setCurrentResult(data);
      setHistory(prev => [
        {
          ...data,
          timestamp: new Date().toLocaleTimeString(),
          source: file.name
        },
        ...prev
      ]);
    } catch (err: any) {
      alert(`Error processing image: ${err.message}. Make sure backend is running on ${API_BASE}.`);
    } finally {
      setIsProcessing(false);
    }
  };

  // Helper to create synthetic test canvas images for one-click scenario testing
  const runScenarioTest = (scenario: 'rain' | 'low_light' | 'glare' | 'clean') => {
    setSelectedScenario(scenario);
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (scenario === 'glare') {
      // 1. Vehicle body - dark metallic gray
      ctx.fillStyle = '#1e2430';
      ctx.fillRect(0, 0, 640, 480);

      // Bumper contour lines
      ctx.strokeStyle = '#0f141d';
      ctx.lineWidth = 6;
      ctx.beginPath();
      ctx.moveTo(0, 380);
      ctx.lineTo(640, 380);
      ctx.stroke();

      // Plate mounting recess
      ctx.fillStyle = '#0b0f17';
      ctx.fillRect(160, 200, 320, 110);

      // Plate background (white reflective)
      ctx.fillStyle = '#f8fafc';
      ctx.fillRect(170, 210, 300, 90);
      ctx.strokeStyle = '#111827';
      ctx.lineWidth = 4;
      ctx.strokeRect(170, 210, 300, 90);

      // Sri Lankan Plate text WP CAA-1234
      ctx.fillStyle = '#0f172a';
      ctx.font = 'bold 36px monospace';
      ctx.fillText('WP CAA-1234', 190, 268);

      // Severe Specular Glare over plate characters
      const glareGrad = ctx.createRadialGradient(310, 255, 5, 310, 255, 85);
      glareGrad.addColorStop(0, 'rgba(255, 255, 255, 0.98)');
      glareGrad.addColorStop(0.35, 'rgba(255, 255, 255, 0.90)');
      glareGrad.addColorStop(0.7, 'rgba(255, 255, 255, 0.40)');
      glareGrad.addColorStop(1, 'transparent');
      ctx.fillStyle = glareGrad;
      ctx.fillRect(0, 0, 640, 480);

    } else if (scenario === 'low_light') {
      // Low light / night scene
      ctx.fillStyle = '#05070a';
      ctx.fillRect(0, 0, 640, 480);
      
      // Underexposed plate
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(170, 210, 300, 90);
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 3;
      ctx.strokeRect(170, 210, 300, 90);

      ctx.fillStyle = '#475569';
      ctx.font = 'bold 36px monospace';
      ctx.fillText('SP GA-7890', 195, 268);

    } else if (scenario === 'rain') {
      // Rain & motion blur
      ctx.fillStyle = '#1f2937';
      ctx.fillRect(0, 0, 640, 480);

      ctx.fillStyle = '#f1f5f9';
      ctx.fillRect(170, 210, 300, 90);
      ctx.strokeStyle = '#111827';
      ctx.lineWidth = 4;
      ctx.strokeRect(170, 210, 300, 90);

      ctx.fillStyle = '#111827';
      ctx.font = 'bold 36px monospace';
      ctx.fillText('CP AB-5582', 195, 268);

      // Rain streak overlay
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
      ctx.lineWidth = 2;
      for (let i = 0; i < 70; i++) {
        const x = Math.random() * 640;
        const y = Math.random() * 480;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + 12, y + 28);
        ctx.stroke();
      }

    } else {
      // Clean daylight reference
      ctx.fillStyle = '#374151';
      ctx.fillRect(0, 0, 640, 480);

      ctx.fillStyle = '#f8fafc';
      ctx.fillRect(170, 210, 300, 90);
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 4;
      ctx.strokeRect(170, 210, 300, 90);

      ctx.fillStyle = '#000000';
      ctx.font = 'bold 38px monospace';
      ctx.fillText('WP CAA-5678', 190, 270);
    }

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `${scenario}_test.jpg`, { type: 'image/jpeg' });
        handleFileUpload(file);
      }
    }, 'image/jpeg');
  };

  const downloadImage = (dataUrl: string, filename: string) => {
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getStatusClass = (status?: string) => {
    switch (status) {
      case 'ACCEPT': return 'status-accept';
      case 'REPROCESS': return 'status-reprocess';
      case 'UNKNOWN': return 'status-unknown';
      default: return 'status-accept';
    }
  };

  return (
    <div className="dashboard-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          AdaptiveVision <span style={{ color: 'var(--primary)' }}>LPR</span>
        </div>

        <nav style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <div
            className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"></path></svg>
            Live ML Studio
          </div>
          <div
            className={`nav-item ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
            Detection History ({history.length})
          </div>
          <div
            className={`nav-item ${activeTab === 'health' ? 'active' : ''}`}
            onClick={() => setActiveTab('health')}
          >
            <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
            Camera Telemetry
          </div>
          <div
            className={`nav-item ${activeTab === 'disaster' ? 'active' : ''}`}
            onClick={() => setActiveTab('disaster')}
          >
            <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
            Disaster Resilience
          </div>
        </nav>

        <div style={{ marginTop: 'auto', padding: '1rem', background: 'rgba(255,255,255,0.03)', borderRadius: '12px' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>BACKEND API</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: backendStatus === 'online' ? 'var(--success)' : 'var(--danger)' }}></span>
            <span>{backendStatus === 'online' ? 'Online (FastAPI Port 8000)' : 'Reconnecting...'}</span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="header">
          <div>
            <h1 style={{ fontSize: '2rem' }}>
              {activeTab === 'dashboard' && 'Live Inference & Adaptive ML'}
              {activeTab === 'history' && 'Processed Detection History'}
              {activeTab === 'health' && 'Camera Stream Telemetry'}
              {activeTab === 'disaster' && 'Disaster Mode Architecture'}
            </h1>
            <p className="metric-label">Uncertainty-Aware Vehicle License Plate Recognition & Glare Restoration Pipeline</p>
          </div>
          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
            <div
              className={`status-badge ${disasterMode ? 'status-danger' : 'status-accept'}`}
              style={{ cursor: 'pointer' }}
              onClick={() => setDisasterMode(!disasterMode)}
            >
              Disaster Mode: {disasterMode ? 'ACTIVE' : 'STANDBY'}
            </div>
            <div className="status-badge status-accept">Backend: {backendStatus.toUpperCase()}</div>
          </div>
        </header>

        {/* TAB 1: DASHBOARD / ML STUDIO */}
        {activeTab === 'dashboard' && (
          <div className="animate-slide-in">
            {/* Image Upload Zone */}
            <div className="glass-panel" style={{ padding: '2rem', marginBottom: '2rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '1rem' }}>
                <div>
                  <h2>Upload Vehicle / Plate Image</h2>
                  <p className="metric-label">Assesses specular glare, executes adaptive ML enhancement filters, and recognizes plate numbers</p>
                </div>
                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                  <button
                    className={`btn ${selectedScenario === 'glare' ? '' : 'btn-secondary'}`}
                    style={{ borderColor: 'rgba(245, 158, 11, 0.4)' }}
                    onClick={() => runScenarioTest('glare')}
                  >
                    ☀️ Test Specular Glare
                  </button>
                  <button
                    className={`btn ${selectedScenario === 'rain' ? '' : 'btn-secondary'}`}
                    onClick={() => runScenarioTest('rain')}
                  >
                    🌧️ Test Rain & Blur
                  </button>
                  <button
                    className={`btn ${selectedScenario === 'low_light' ? '' : 'btn-secondary'}`}
                    onClick={() => runScenarioTest('low_light')}
                  >
                    🌙 Test Low Light
                  </button>
                  <button
                    className={`btn ${selectedScenario === 'clean' ? '' : 'btn-secondary'}`}
                    onClick={() => runScenarioTest('clean')}
                  >
                    🚗 Test Clean Day
                  </button>
                </div>
              </div>

              <div
                style={{
                  border: `2px dashed ${dragOver ? 'var(--primary)' : 'rgba(255, 255, 255, 0.2)'}`,
                  borderRadius: '16px',
                  padding: '2.5rem',
                  textAlign: 'center',
                  background: dragOver ? 'rgba(59, 130, 246, 0.08)' : 'rgba(0, 0, 0, 0.2)',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragOver(false);
                  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                    handleFileUpload(e.dataTransfer.files[0]);
                  }
                }}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  accept="image/*"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      handleFileUpload(e.target.files[0]);
                    }
                  }}
                />
                <svg width="48" height="48" fill="none" stroke="var(--primary)" strokeWidth="1.5" viewBox="0 0 24 24" style={{ margin: '0 auto 1rem auto' }}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                </svg>
                <h3 style={{ marginBottom: '0.5rem' }}>
                  {isProcessing ? '⚡ Optimizing Image with Machine Learning Pipeline...' : 'Click or Drag & Drop Vehicle Photo Here'}
                </h3>
                <p className="metric-label">Upload any car photo or plate image with glare, blur, or darkness (Supports JPEG, PNG, WEBP)</p>
              </div>
            </div>

            {/* DEDICATED ML PLATE RESTORATION & CLARITY SHOWCASE */}
            {currentResult && (
              <div className="ml-showcase-panel">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem', marginBottom: '1rem' }}>
                  <div>
                    <h2 style={{ fontSize: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      🔬 Machine Learning Plate Optimization & Restoration Lab
                    </h2>
                    <p className="metric-label">
                      End-to-end verified computer vision & deep learning pipeline: Glare Inpainting, Denoising, Contrast Normalization & Super-Resolution
                    </p>
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
                    <span className={`status-badge ${getStatusClass(currentResult.status)}`}>
                      Decision: {currentResult.status}
                    </span>
                    {currentResult.enhanced_plate_preview && (
                      <button
                        className="btn"
                        style={{ fontSize: '0.82rem', padding: '0.35rem 0.8rem' }}
                        onClick={() => downloadImage(currentResult.enhanced_plate_preview, `enhanced_plate_${currentResult.plate || 'restored'}.png`)}
                      >
                        ⬇ Download Plate
                      </button>
                    )}
                    {currentResult.sauvola_crop_preview && (
                      <button
                        className="btn btn-secondary"
                        style={{ fontSize: '0.82rem', padding: '0.35rem 0.8rem' }}
                        onClick={() => downloadImage(currentResult.sauvola_crop_preview!, `sauvola_mask_${currentResult.plate || 'pixels'}.png`)}
                      >
                        ⬇ Download Binary Mask
                      </button>
                    )}
                  </div>
                </div>

                {/* View Mode & Zoom Navigation Bar */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', margin: '1rem 0 1.25rem 0', flexWrap: 'wrap', gap: '0.75rem', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '0.75rem' }}>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button
                      className={`view-mode-pill ${viewMode === 'slider' ? 'active' : ''}`}
                      onClick={() => setViewMode('slider')}
                    >
                      ↔️ A/B Comparison Slider
                    </button>
                    <button
                      className={`view-mode-pill ${viewMode === 'side-by-side' ? 'active' : ''}`}
                      onClick={() => setViewMode('side-by-side')}
                    >
                      📊 3-Panel Inspection
                    </button>
                    <button
                      className={`view-mode-pill ${viewMode === 'sauvola' ? 'active' : ''}`}
                      onClick={() => setViewMode('sauvola')}
                    >
                      🔬 Sauvola Binary Pixels
                    </button>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Pixel Zoom:</span>
                    {(['1x', '2x', '3x'] as const).map(z => (
                      <button
                        key={z}
                        className={`view-mode-pill ${zoomLevel === z ? 'active' : ''}`}
                        style={{ padding: '0.25rem 0.6rem', fontSize: '0.72rem' }}
                        onClick={() => setZoomLevel(z)}
                      >
                        {z}
                      </button>
                    ))}
                  </div>
                </div>

                {/* VIEW MODE 1: Interactive Drag A/B Split Comparison Slider */}
                {viewMode === 'slider' && (
                  <div style={{ padding: '0.5rem 0' }}>
                    <div
                      className="interactive-slider-box"
                      style={{
                        transform: zoomLevel === '2x' ? 'scale(1.15)' : zoomLevel === '3x' ? 'scale(1.3)' : 'scale(1)',
                        transformOrigin: 'center center',
                        transition: 'transform 0.2s',
                        height: '210px'
                      }}
                    >
                      {/* Layer 1: Enhanced (Full background) */}
                      <div className="slider-layer">
                        <img src={currentResult.enhanced_plate_preview} alt="ML-Enhanced Plate" />
                        <span className="preview-badge badge-enhanced" style={{ right: 10, left: 'auto' }}>
                          ML-OPTIMIZED (SHARP PIXELS)
                        </span>
                      </div>

                      {/* Layer 2: Raw / Degraded (Clipped to sliderPos %) */}
                      <div
                        className="slider-layer"
                        style={{
                          clipPath: `polygon(0 0, ${sliderPos}% 0, ${sliderPos}% 100%, 0 100%)`
                        }}
                      >
                        <img
                          src={currentResult.original_crop_preview || previewUrl || ''}
                          alt="Raw Degraded Sensor"
                        />
                        <span className="preview-badge badge-degraded">
                          RAW SENSOR DEGRADED / BLURRED
                        </span>
                      </div>

                      {/* Slider Divider Line */}
                      <div className="slider-handle-line" style={{ left: `${sliderPos}%` }}>
                        <div className="slider-handle-circle">↔</div>
                      </div>

                      {/* Range Input for dragging */}
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={sliderPos}
                        onChange={(e) => setSliderPos(Number(e.target.value))}
                        className="slider-range-input"
                        title="Drag left/right to compare raw vs ML enhanced"
                      />
                    </div>
                    <div style={{ textAlign: 'center', fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                      Drag slider left/right to compare <strong>Raw Degraded Sensor</strong> (Left) vs <strong>ML-Reconstructed Pixels</strong> (Right)
                    </div>
                  </div>
                )}

                {/* VIEW MODE 2: 3-Panel Visual Comparison Grid */}
                {viewMode === 'side-by-side' && (
                  <div className="ml-comparison-grid">
                    {/* Panel 1: Vehicle Context & Localization */}
                    <div className="ml-preview-card vehicle-context">
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                        <strong style={{ fontSize: '0.9rem', color: '#93c5fd' }}>1. Vehicle Localization</strong>
                        <span className="metric-label" style={{ fontSize: '0.75rem' }}>Full Scene</span>
                      </div>
                      <div className="preview-image-container">
                        <span className="preview-badge badge-context">DETECTED BOUNDING BOX</span>
                        <img
                          src={currentResult.annotated_frame_preview || previewUrl || ''}
                          alt="Vehicle Localization"
                        />
                      </div>
                      <div className="metric-row" style={{ marginTop: '0.75rem' }}>
                        <span className="metric-label">Localization Method</span>
                        <span className="metric-value" style={{ fontSize: '0.8rem' }}>Adaptive Morphological Contour</span>
                      </div>
                    </div>

                    {/* Panel 2: Raw Sensor Crop (Degraded) */}
                    <div className="ml-preview-card raw-degraded">
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                        <strong style={{ fontSize: '0.9rem', color: '#fca5a5' }}>2. Raw Sensor Crop (Degraded)</strong>
                        <span className="metric-label" style={{ fontSize: '0.75rem' }}>Camera Sensor</span>
                      </div>
                      <div className="preview-image-container">
                        <span className="preview-badge badge-degraded">RAW DEGRADATION / GLARE</span>
                        <img
                          src={currentResult.original_crop_preview || previewUrl || ''}
                          alt="Raw Degraded Plate"
                        />
                      </div>
                      <div className="metric-row" style={{ marginTop: '0.75rem' }}>
                        <span className="metric-label">Detected Glare Index</span>
                        <span className="metric-value" style={{ fontSize: '0.8rem', color: currentResult.quality_metrics.glare > 0.05 ? 'var(--warning)' : 'var(--text-muted)' }}>
                          {(currentResult.quality_metrics.glare * 100).toFixed(1)}% {currentResult.quality_metrics.glare > 0.05 ? '(Glare Active)' : ''}
                        </span>
                      </div>
                    </div>

                    {/* Panel 3: Machine Learning Enhanced Plate (Crisp & Readable) */}
                    <div className="ml-preview-card enhanced">
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                        <strong style={{ fontSize: '0.9rem', color: '#6ee7b7' }}>3. ML-Optimized Plate (Crisp Letters)</strong>
                        <span className="metric-label" style={{ fontSize: '0.75rem' }}>Enhanced Output</span>
                      </div>
                      <div className="preview-image-container">
                        <span className="preview-badge badge-enhanced">ML INPAINTED & RESTORED</span>
                        <img
                          src={currentResult.enhanced_plate_preview}
                          alt="Enhanced Number Plate"
                        />
                      </div>
                      <div className="metric-row" style={{ marginTop: '0.75rem' }}>
                        <span className="metric-label">OCR Recognition Score</span>
                        <span className="metric-value" style={{ fontSize: '0.8rem', color: 'var(--success)' }}>
                          {(currentResult.confidence * 100).toFixed(1)}% Confidence
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {/* VIEW MODE 3: Sauvola Local Binary Character Mask */}
                {viewMode === 'sauvola' && (
                  <div className="ml-comparison-grid">
                    <div className="ml-preview-card enhanced">
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                        <strong style={{ fontSize: '0.9rem', color: '#6ee7b7' }}>Reconstructed Color Plate</strong>
                        <span className="metric-label" style={{ fontSize: '0.75rem' }}>Deblurred & Super-Resolved</span>
                      </div>
                      <div className="preview-image-container" style={{ height: '180px' }}>
                        <span className="preview-badge badge-enhanced">WIENER + CLAHE + INPAINT</span>
                        <img
                          src={currentResult.enhanced_plate_preview}
                          alt="Enhanced RGB"
                          style={{
                            transform: zoomLevel === '2x' ? 'scale(1.4)' : zoomLevel === '3x' ? 'scale(1.8)' : 'scale(1)',
                            transition: 'transform 0.2s'
                          }}
                        />
                      </div>
                    </div>

                    <div className="ml-preview-card" style={{ border: '1px solid rgba(139, 92, 246, 0.4)', background: 'rgba(76, 29, 149, 0.15)' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                        <strong style={{ fontSize: '0.9rem', color: '#c4b5fd' }}>Sauvola Binary Character Pixels</strong>
                        <span className="metric-label" style={{ fontSize: '0.75rem' }}>Local Window Binarization</span>
                      </div>
                      <div className="preview-image-container" style={{ height: '180px', background: '#fff' }}>
                        <span className="preview-badge" style={{ background: 'rgba(124, 58, 237, 0.9)', color: 'white' }}>
                          ISOLATED CHARACTER STROKES
                        </span>
                        <img
                          src={currentResult.sauvola_crop_preview || currentResult.enhanced_plate_preview}
                          alt="Sauvola Binary Mask"
                          style={{
                            transform: zoomLevel === '2x' ? 'scale(1.4)' : zoomLevel === '3x' ? 'scale(1.8)' : 'scale(1)',
                            transition: 'transform 0.2s',
                            filter: 'contrast(1.2)'
                          }}
                        />
                      </div>
                    </div>
                  </div>
                )}

                {/* Filter Pipeline Badges */}
                <div style={{ marginTop: '1.5rem', background: 'rgba(0,0,0,0.25)', padding: '1rem', borderRadius: '12px' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem', fontWeight: 600 }}>
                    ACTIVE COMPUTER VISION & ML OPTIMIZATION FILTERS EXECUTED:
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                    {currentResult.adaptive_routing.steps_executed.length > 0 ? (
                      currentResult.adaptive_routing.steps_executed.map((step) => {
                        const isGlare = step.includes('GLARE');
                        return (
                          <span
                            key={step}
                            className={`filter-chip ${isGlare ? 'active-glare' : ''}`}
                          >
                            {isGlare ? '☀️ ' : '✓ '}
                            {step.replace(/_/g, ' ')}
                          </span>
                        );
                      })
                    ) : (
                      <span className="filter-chip">✓ STANDARD PASS (No Degraded Conditions Detected)</span>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Live Inference Output Section */}
            {currentResult && (
              <div className="grid-3" style={{ marginBottom: '2rem' }}>
                {/* Result Plate Card */}
                <div className="glass-panel" style={{ padding: '1.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h3 style={{ color: 'var(--text-muted)' }}>Event {currentResult.event_id}</h3>
                    <span className={`status-badge ${getStatusClass(currentResult.status)}`}>
                      {currentResult.status}
                    </span>
                  </div>

                  <div className="plate-display" style={{ color: currentResult.status === 'ACCEPT' ? 'var(--success)' : currentResult.status === 'REPROCESS' ? 'var(--warning)' : 'var(--unknown)' }}>
                    {currentResult.plate}
                  </div>

                  <div style={{ marginBottom: '1rem' }}>
                    {currentResult.conditions.map(c => (
                      <span key={c} className="condition-tag">{c}</span>
                    ))}
                  </div>

                  <div className="metric-row">
                    <span className="metric-label">Recognition Confidence</span>
                    <span className="metric-value">{(currentResult.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="progress-bar-bg">
                    <div className="progress-bar-fill" style={{ width: `${currentResult.confidence * 100}%`, background: 'var(--primary)' }}></div>
                  </div>

                  <div className="metric-row" style={{ marginTop: '1rem' }}>
                    <span className="metric-label">Sri Lankan Format Valid</span>
                    <span className="metric-value" style={{ color: currentResult.format_valid ? 'var(--success)' : 'var(--warning)' }}>
                      {currentResult.format_valid ? 'VALID SYNTAX' : 'INVALID PATTERN'}
                    </span>
                  </div>
                  {currentResult.validation_reason && (
                    <div className="metric-row">
                      <span className="metric-label">Validation Reason</span>
                      <span className="metric-value" style={{ fontSize: '0.8rem', color: currentResult.format_valid ? 'var(--success)' : 'var(--warning)' }}>
                        {currentResult.validation_reason}
                      </span>
                    </div>
                  )}
                </div>

                {/* Quality & Occlusion Metrics */}
                <div className="glass-panel" style={{ padding: '1.5rem' }}>
                  <h3 style={{ color: 'var(--text-muted)', marginBottom: '1rem' }}>Quality & Occlusion Metrics</h3>

                  <div className="metric-row">
                    <span className="metric-label">Blur Sharpness Score</span>
                    <span className="metric-value">{currentResult.quality_metrics.blur.toFixed(1)}</span>
                  </div>
                  <div className="metric-row">
                    <span className="metric-label">Brightness Metric</span>
                    <span className="metric-value">{(currentResult.quality_metrics.brightness * 100).toFixed(0)}%</span>
                  </div>
                  <div className="metric-row">
                    <span className="metric-label">Contrast Standard Dev</span>
                    <span className="metric-value">{currentResult.quality_metrics.contrast.toFixed(1)}</span>
                  </div>
                  <div className="metric-row">
                    <span className="metric-label">Glare Saturation</span>
                    <span className="metric-value">{(currentResult.quality_metrics.glare * 100).toFixed(1)}%</span>
                  </div>

                  <div className="metric-row" style={{ marginTop: '0.75rem' }}>
                    <span className="metric-label">Plate Occlusion</span>
                    <span className="metric-value">{(currentResult.occlusion_analysis.occlusion * 100).toFixed(1)}%</span>
                  </div>
                  <div className="progress-bar-bg">
                    <div className="progress-bar-fill" style={{ width: `${currentResult.occlusion_analysis.occlusion * 100}%`, background: 'var(--unknown)' }}></div>
                  </div>

                  <div className="metric-row" style={{ marginTop: '1rem' }}>
                    <span className="metric-label">Recoverability Class</span>
                    <span className="metric-value">{currentResult.recoverability.class} ({(currentResult.recoverability.confidence * 100).toFixed(0)}%)</span>
                  </div>
                </div>

                {/* Uncertainty & Epistemic Decision Engine */}
                <div className="glass-panel" style={{ padding: '1.5rem' }}>
                  <h3 style={{ color: 'var(--text-muted)', marginBottom: '1rem' }}>Uncertainty & Decision Engine</h3>

                  <div className="metric-row">
                    <span className="metric-label">Aleatoric (Data / Noise) Uncertainty</span>
                    <span className="metric-value">{(currentResult.uncertainty.aleatoric * 100).toFixed(1)}%</span>
                  </div>
                  <div className="progress-bar-bg">
                    <div className="progress-bar-fill" style={{ width: `${currentResult.uncertainty.aleatoric * 100}%`, background: 'var(--warning)' }}></div>
                  </div>

                  <div className="metric-row" style={{ marginTop: '1rem' }}>
                    <span className="metric-label">Epistemic (Model Ambiguity) Uncertainty</span>
                    <span className="metric-value">{(currentResult.uncertainty.epistemic * 100).toFixed(1)}%</span>
                  </div>
                  <div className="progress-bar-bg">
                    <div className="progress-bar-fill" style={{ width: `${currentResult.uncertainty.epistemic * 100}%`, background: 'var(--primary)' }}></div>
                  </div>

                  <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'rgba(0,0,0,0.2)', borderRadius: '8px' }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>RAW OCR INFERENCE:</div>
                    <code style={{ fontSize: '0.9rem', color: '#93c5fd' }}>{currentResult.raw_ocr || 'N/A'}</code>
                  </div>
                </div>
              </div>
            )}

            {/* Reprocessing Trace UI */}
            <div className="grid-2">
              <div className="glass-panel" style={{ padding: '1.5rem' }}>
                <h2>Adaptive Routing Trace</h2>
                <p className="metric-label" style={{ marginBottom: '1.5rem' }}>Multi-stage computer vision & deep learning pipeline</p>

                <div className="timeline">
                  <div className="timeline-item">
                    <h4>Stage 1: Environmental Condition Classifier</h4>
                    <p className="metric-label">Extracts HSV/LAB specular glare masks, Laplacian variance blur, and luminosity histograms.</p>
                  </div>
                  <div className="timeline-item">
                    <h4>Stage 2: Plate Visibility & Occlusion Analysis</h4>
                    <p className="metric-label">Morphological gradient segmentation, spatial character zone density, and recoverability score.</p>
                  </div>
                  <div className="timeline-item">
                    <h4>Stage 3: Adaptive Filter Graph Execution</h4>
                    <p className="metric-label">Telea specular glare inpainting, dual TopHat/BlackHat contrast recovery, CLAHE, and 2x bicubic super-resolution.</p>
                  </div>
                  <div className="timeline-item">
                    <h4>Stage 4: Optical Character Recognition & Sri Lankan Syntax Engine</h4>
                    <p className="metric-label">EasyOCR deep CRAFT+CRNN character extraction with alphanumeric confusion correction (e.g. WP CAA-1234).</p>
                  </div>
                  <div className="timeline-item">
                    <h4>Stage 5: Epistemic & Aleatoric Uncertainty Decision</h4>
                    <p className="metric-label">Resolves ACCEPT (high-confidence), REPROCESS (recoverable ambiguity), or UNKNOWN (insufficient evidence).</p>
                  </div>
                </div>
              </div>

              <div className="glass-panel" style={{ padding: '1.5rem' }}>
                <h2>Camera Health Monitor</h2>
                <p className="metric-label" style={{ marginBottom: '1.5rem' }}>Active Stream Telemetry</p>

                <div className="metric-row">
                  <span className="metric-label">CAM-01 (Highway North)</span>
                  <span className="status-badge status-accept" style={{ padding: '0.25rem 0.75rem', fontSize: '0.7rem' }}>99% HEALTH</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">CAM-02 (Toll Booth A)</span>
                  <span className="status-badge status-reprocess" style={{ padding: '0.25rem 0.75rem', fontSize: '0.7rem' }}>74% HEALTH</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">CAM-03 (Toll Booth B)</span>
                  <span className="status-badge status-accept" style={{ padding: '0.25rem 0.75rem', fontSize: '0.7rem' }}>92% HEALTH</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">CAM-04 (Junction West)</span>
                  <span className="status-badge status-danger" style={{ padding: '0.25rem 0.75rem', fontSize: '0.7rem' }}>41% HEALTH</span>
                </div>

                <div style={{ marginTop: '2rem', padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '8px', border: '1px solid rgba(239, 68, 68, 0.3)' }}>
                  <h4 style={{ color: 'var(--danger)', marginBottom: '0.5rem' }}>Telemetry Notice: CAM-04</h4>
                  <p className="metric-label" style={{ fontSize: '0.8rem' }}>Frame drop rate exceeds 30%. Degraded-stream policy prioritizing keyframes.</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: DETECTION HISTORY */}
        {activeTab === 'history' && (
          <div className="animate-slide-in glass-panel" style={{ padding: '2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <h2>Recognition Event Log</h2>
              <button className="btn btn-secondary" onClick={() => setHistory([])}>Clear History</button>
            </div>

            {history.length === 0 ? (
              <p className="metric-label" style={{ textAlign: 'center', padding: '3rem' }}>
                No events recorded yet. Upload an image or run scenario tests in the Live ML Studio.
              </p>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', color: 'var(--text-muted)' }}>
                      <th style={{ padding: '1rem' }}>Time</th>
                      <th style={{ padding: '1rem' }}>Source</th>
                      <th style={{ padding: '1rem' }}>Predicted Plate</th>
                      <th style={{ padding: '1rem' }}>Decision</th>
                      <th style={{ padding: '1rem' }}>Confidence</th>
                      <th style={{ padding: '1rem' }}>Conditions</th>
                      <th style={{ padding: '1rem' }}>Pipeline</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((item, idx) => (
                      <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                        <td style={{ padding: '1rem' }} className="metric-label">{item.timestamp}</td>
                        <td style={{ padding: '1rem' }}>{item.source}</td>
                        <td style={{ padding: '1rem', fontFamily: 'monospace', fontWeight: 'bold' }}>{item.plate}</td>
                        <td style={{ padding: '1rem' }}>
                          <span className={`status-badge ${getStatusClass(item.status)}`} style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}>
                            {item.status}
                          </span>
                        </td>
                        <td style={{ padding: '1rem' }}>{(item.confidence * 100).toFixed(1)}%</td>
                        <td style={{ padding: '1rem' }}>
                          {item.conditions.map(c => <span key={c} className="condition-tag" style={{ fontSize: '0.65rem' }}>{c}</span>)}
                        </td>
                        <td style={{ padding: '1rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                          {item.adaptive_routing.steps_executed.join(' → ') || 'Direct Pass'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* TAB 3: CAMERA TELEMETRY */}
        {activeTab === 'health' && (
          <div className="animate-slide-in grid-2">
            <div className="glass-panel" style={{ padding: '2rem' }}>
              <h2>Active Camera Stream Nodes</h2>
              <p className="metric-label" style={{ marginBottom: '2rem' }}>Telemetry collected from RTSP and video streaming feeds</p>

              {[
                { id: 'CAM-01', name: 'Highway North - Km 14', fps: 29.8, drops: '0.2%', health: 99, status: 'NORMAL' },
                { id: 'CAM-02', name: 'Toll Booth A - Inbound', fps: 24.1, drops: '4.8%', health: 74, status: 'RECOVERY' },
                { id: 'CAM-03', name: 'Toll Booth B - Outbound', fps: 30.0, drops: '0.1%', health: 92, status: 'NORMAL' },
                { id: 'CAM-04', name: 'Junction West - Adverse Feed', fps: 14.2, drops: '32.1%', health: 41, status: 'DEGRADED' },
              ].map(cam => (
                <div key={cam.id} style={{ marginBottom: '1.5rem', padding: '1rem', background: 'rgba(0,0,0,0.2)', borderRadius: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                    <strong>{cam.id}: {cam.name}</strong>
                    <span className={`status-badge ${cam.health > 80 ? 'status-accept' : cam.health > 60 ? 'status-reprocess' : 'status-danger'}`}>
                      {cam.status}
                    </span>
                  </div>
                  <div className="metric-row">
                    <span className="metric-label">Frame Rate (FPS)</span>
                    <span className="metric-value">{cam.fps}</span>
                  </div>
                  <div className="metric-row">
                    <span className="metric-label">Frame Drop Rate</span>
                    <span className="metric-value">{cam.drops}</span>
                  </div>
                  <div className="metric-row">
                    <span className="metric-label">Health Metric</span>
                    <span className="metric-value">{cam.health}%</span>
                  </div>
                  <div className="progress-bar-bg" style={{ marginTop: '0.5rem' }}>
                    <div className="progress-bar-fill" style={{ width: `${cam.health}%`, background: cam.health > 80 ? 'var(--success)' : cam.health > 60 ? 'var(--warning)' : 'var(--danger)' }}></div>
                  </div>
                </div>
              ))}
            </div>

            <div className="glass-panel" style={{ padding: '2rem' }}>
              <h2>Degraded-Stream Mitigation</h2>
              <p className="metric-label" style={{ marginBottom: '1.5rem' }}>Adaptive response to stream degradation</p>

              <div className="timeline">
                <div className="timeline-item">
                  <h4>Jitter Buffering</h4>
                  <p className="metric-label">Compensates for network latency spikes up to 400ms without discarding vehicle tracks.</p>
                </div>
                <div className="timeline-item">
                  <h4>Temporal Frame Prioritization</h4>
                  <p className="metric-label">Drops intermediate stationary frames when backlogs occur, reserving compute for clear plate crops.</p>
                </div>
                <div className="timeline-item">
                  <h4>Uncertainty-Gated Abstention</h4>
                  <p className="metric-label">When optical degradation renders evidence inconclusive, returns UNKNOWN rather than hallucinating plates.</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: DISASTER RESILIENCE */}
        {activeTab === 'disaster' && (
          <div className="animate-slide-in glass-panel" style={{ padding: '2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
              <div>
                <h2>Disaster Resilience & Edge Failover</h2>
                <p className="metric-label">Operating policies during severe weather, network disruption, or server outages</p>
              </div>
              <button
                className={`btn ${disasterMode ? 'btn-secondary' : ''}`}
                style={{ background: disasterMode ? 'var(--danger)' : 'var(--primary)' }}
                onClick={() => setDisasterMode(!disasterMode)}
              >
                {disasterMode ? 'Deactivate Disaster Mode' : 'Activate Disaster Mode'}
              </button>
            </div>

            <div className="grid-3" style={{ marginBottom: '2rem' }}>
              <div style={{ padding: '1.5rem', background: 'rgba(0,0,0,0.2)', borderRadius: '12px' }}>
                <h4 style={{ marginBottom: '0.5rem' }}>Local Edge Cache</h4>
                <p className="metric-label" style={{ marginBottom: '1rem' }}>SQLite zero-dependency persistence buffer</p>
                <div className="metric-value" style={{ color: 'var(--success)' }}>Active (SQLite fallback)</div>
              </div>
              <div style={{ padding: '1.5rem', background: 'rgba(0,0,0,0.2)', borderRadius: '12px' }}>
                <h4 style={{ marginBottom: '0.5rem' }}>Reduced Compute Mode</h4>
                <p className="metric-label" style={{ marginBottom: '1rem' }}>Lightweight filters during power/compute throttling</p>
                <div className="metric-value" style={{ color: disasterMode ? 'var(--danger)' : 'var(--text-muted)' }}>
                  {disasterMode ? 'ENFORCED (Minimal DAG)' : 'STANDBY (Full DAG)'}
                </div>
              </div>
              <div style={{ padding: '1.5rem', background: 'rgba(0,0,0,0.2)', borderRadius: '12px' }}>
                <h4 style={{ marginBottom: '0.5rem' }}>Safe Abstention Policy</h4>
                <p className="metric-label" style={{ marginBottom: '1rem' }}>Enforces UNKNOWN whenever confidence &lt; 0.85</p>
                <div className="metric-value" style={{ color: 'var(--unknown)' }}>Strict Mode Enabled</div>
              </div>
            </div>

            <div style={{ padding: '1.5rem', background: 'rgba(59, 130, 246, 0.08)', borderRadius: '12px', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
              <h4 style={{ color: 'var(--primary)', marginBottom: '0.5rem' }}>Architecture Safeguards</h4>
              <p className="metric-label">
                Under disaster mode, the pipeline bypasses optional GPU super-resolution and switches to deterministic CLAHE and edge-density recoverability heuristics, ensuring continuous operations even on edge micro-nodes with limited power or zero internet connectivity.
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
