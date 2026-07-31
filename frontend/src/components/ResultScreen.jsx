import React, { useState } from 'react';
import ShotList from './ShotList';
import { RotateCcw, Target, AlertTriangle } from 'lucide-react';

export default function ResultScreen({ result, onReset, onEnterTeachMode }) {
  const [selectedIndex, setSelectedIndex] = useState(0);

  const directShots = result.direct_shots || [];
  const bankShots = result.bank_shots || [];
  const kickShots = result.kick_shots || [];
  const totalShots = directShots.length + bankShots.length + kickShots.length;

  return (
    <div className="result-layout">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
        <h2>Analysis Results</h2>
        <div style={{ display: 'flex', gap: '8px' }}>
          {onEnterTeachMode && (
            <button 
              onClick={onEnterTeachMode}
              style={{
                background: 'rgba(245, 158, 11, 0.15)',
                border: '1px solid var(--accent-yellow)',
                color: 'var(--accent-yellow)',
                padding: '8px 14px',
                borderRadius: 'var(--radius-sm)',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontSize: '0.85rem',
                fontWeight: 600,
              }}
            >
              <Target size={16} />
              Re-position Cue Ball
            </button>
          )}
          <button 
            onClick={onReset}
            style={{
              background: 'transparent',
              border: '1px solid var(--border-color)',
              color: 'var(--text-main)',
              padding: '8px 14px',
              borderRadius: 'var(--radius-sm)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '0.85rem',
            }}
          >
            <RotateCcw size={16} />
            New Photo
          </button>
        </div>
      </div>

      {totalShots === 0 && (
        <div 
          style={{
            background: 'rgba(245, 158, 11, 0.12)',
            border: '1px solid var(--accent-yellow)',
            borderRadius: 'var(--radius-md)',
            padding: '12px 16px',
            color: 'var(--text-main)',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            fontSize: '0.9rem',
          }}
        >
          <AlertTriangle size={22} color="var(--accent-yellow)" style={{ flexShrink: 0 }} />
          <div>
            <strong style={{ color: 'var(--accent-yellow)' }}>No valid shots found.</strong> Every possible path is obstructed or misses all pockets. Identified balls, pockets, and rail diamonds are rendered below.
          </div>
        </div>
      )}

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
