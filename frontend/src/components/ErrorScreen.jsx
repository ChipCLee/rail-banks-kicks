import React from 'react';
import { AlertCircle, RotateCcw } from 'lucide-react';

export default function ErrorScreen({ message, onReset }) {
  return (
    <div className="error-box">
      <AlertCircle size={48} color="#ef4444" />
      <div>
        <h3 className="error-title">Analysis Error</h3>
        <p style={{ color: 'var(--text-muted)', marginTop: '8px', maxWidth: '500px' }}>
          {message || 'No valid shots found — every possible bank is blocked or misses all pockets.'}
        </p>
      </div>
      <button 
        className="btn-upload" 
        onClick={onReset}
        style={{ marginTop: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}
      >
        <RotateCcw size={18} />
        Try Another Photo
      </button>
    </div>
  );
}
