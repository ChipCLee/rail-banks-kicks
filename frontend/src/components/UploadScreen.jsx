import React, { useState, useRef } from 'react';
import { UploadCloud, AlertTriangle, Palette } from 'lucide-react';

export default function UploadScreen({ onFileSelected, error: externalError, defaultFeltColor = 'auto' }) {
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState(externalError || null);
  const [feltColor, setFeltColor] = useState(defaultFeltColor);
  const inputRef = useRef(null);

  const validateAndPass = (file) => {
    setError(null);
    if (!file) return;
    
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      setError('Invalid format. Please upload JPG, PNG, or WEBP images only.');
      return;
    }
    
    const maxBytes = 20 * 1024 * 1024;
    if (file.size > maxBytes) {
      setError('File size exceeds 20 MB limit.');
      return;
    }
    
    onFileSelected(file, feltColor);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndPass(e.dataTransfer.files[0]);
    }
  };

  const feltOptions = [
    { id: 'auto', label: 'Auto Detect', color: '#00DCFF', icon: '✨' },
    { id: 'blue', label: 'Simonis Blue', color: '#3B82F6', icon: '🟦' },
    { id: 'green', label: 'Green Felt', color: '#10B981', icon: '🟩' },
    { id: 'red', label: 'Red / Burgundy', color: '#EF4444', icon: '🟥' },
  ];

  return (
    <div className="upload-container">
      {/* Felt Color Selection Control */}
      <div style={{ background: 'var(--bg-card)', padding: '14px 18px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px', fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-main)' }}>
          <Palette size={18} color="var(--accent-cyan)" />
          <span>Select Table Felt Color (for higher CV accuracy):</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '8px' }}>
          {feltOptions.map((opt) => {
            const isSelected = feltColor === opt.id;
            return (
              <button
                key={opt.id}
                type="button"
                onClick={() => setFeltColor(opt.id)}
                style={{
                  padding: '8px 12px',
                  borderRadius: 'var(--radius-sm)',
                  border: isSelected ? `2px solid ${opt.color}` : '1px solid var(--border-color)',
                  background: isSelected ? 'rgba(30, 41, 59, 0.9)' : 'var(--bg-main)',
                  color: isSelected ? '#FFFFFF' : 'var(--text-muted)',
                  fontWeight: isSelected ? 600 : 400,
                  fontSize: '0.85rem',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px',
                  transition: 'all 0.2s ease',
                  boxShadow: isSelected ? `0 0 8px ${opt.color}40` : 'none',
                }}
              >
                <span>{opt.icon}</span>
                <span>{opt.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div 
        className={`dropzone ${dragActive ? 'active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <UploadCloud className="upload-icon" />
        <div>
          <h2>Upload Pool Table Photo</h2>
          <p style={{ color: 'var(--text-muted)', marginTop: '4px' }}>
            Hold phone overhead and capture full table. Drag & drop or tap below.
          </p>
        </div>
        <button type="button" className="btn-upload">Choose File</button>
        <input 
          ref={inputRef}
          type="file" 
          accept="image/jpeg,image/png,image/webp" 
          style={{ display: 'none' }}
          onChange={(e) => e.target.files?.[0] && validateAndPass(e.target.files[0])}
        />
      </div>

      {(error || externalError) && (
        <div style={{ marginTop: '16px', color: '#ef4444', display: 'flex', alignItems: 'center', gap: '8px', justifyContent: 'center' }}>
          <AlertTriangle size={18} />
          <span>{error || externalError}</span>
        </div>
      )}
    </div>
  );
}
