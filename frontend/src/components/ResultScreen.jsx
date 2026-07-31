import React, { useState } from 'react';
import ShotList from './ShotList';
import { RotateCcw } from 'lucide-react';

export default function ResultScreen({ result, onReset }) {
  const [selectedIndex, setSelectedIndex] = useState(0);

  const directShots = result.direct_shots || [];
  const bankShots = result.bank_shots || [];
  const kickShots = result.kick_shots || [];

  return (
    <div className="result-layout">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2>Analysis Results</h2>
        <button 
          onClick={onReset}
          style={{
            background: 'transparent',
            border: '1px solid var(--border-color)',
            color: 'var(--text-main)',
            padding: '8px 16px',
            borderRadius: 'var(--radius-sm)',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
        >
          <RotateCcw size={16} />
          New Photo
        </button>
      </div>

      <div className="annotated-image-card">
        {result.annotated_image_b64 ? (
          <img 
            src={`data:image/jpeg;base64,${result.annotated_image_b64}`} 
            alt="Annotated Pool Table Top-Down View" 
          />
        ) : (
          <div style={{ padding: '40px', textAlign: 'center' }}>Top-down view unavailable</div>
        )}
      </div>

      <ShotList 
        directShots={directShots}
        bankShots={bankShots}
        kickShots={kickShots}
        selectedIndex={selectedIndex}
        onSelectShot={setSelectedIndex}
      />
    </div>
  );
}
