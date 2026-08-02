import React, { useState, useRef } from 'react';
import { UploadCloud, AlertTriangle } from 'lucide-react';

export default function UploadScreen({ onFileSelected, error: externalError }) {
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState(externalError || null);
  const inputRef = useRef(null);

  const validateAndPass = (file) => {
    setError(null);
    if (!file) return;
    
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/heic', 'image/heif'];
    if (!validTypes.includes(file.type)) {
      setError('Invalid format. Please upload JPG, PNG, WEBP, or HEIC images only.');
      return;
    }
    
    const maxBytes = 20 * 1024 * 1024;
    if (file.size > maxBytes) {
      setError('File size exceeds 20 MB limit.');
      return;
    }
    
    onFileSelected(file);
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

  return (
    <div className="upload-container">
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
          accept="image/jpeg,image/png,image/webp,image/heic,image/heif,.heic,.heif"
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
