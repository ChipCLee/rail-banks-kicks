import React, { useState } from 'react';
import ShotList from './ShotList';
import TableDiagramCanvas from './TableDiagramCanvas';
import { RotateCcw, Target, AlertTriangle, Eye, Camera } from 'lucide-react';

export default function ResultScreen({ result, onReset, onEnterTeachMode }) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [activeView, setActiveView] = useState('cv_diagram'); // Default view is 'cv_diagram'

  const directShots = result.direct_shots || [];
  const bankShots = result.bank_shots || [];
  const kickShots = result.kick_shots || [];
  const totalShots = directShots.length + bankShots.length + kickShots.length;
  const ballsCount = result.balls?.length || 0;

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

      {/* View Switcher Tabs: 2D Diagram (Main) vs Original Picture */}
      <div style={{ display: 'flex', gap: '8px', background: 'var(--bg-card)', padding: '6px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
        <button
          onClick={() => setActiveView('cv_diagram')}
          style={{
            flex: 1,
            padding: '8px 14px',
            borderRadius: 'var(--radius-sm)',
            border: 'none',
            background: activeView === 'cv_diagram' ? 'var(--accent-cyan)' : 'transparent',
            color: activeView === 'cv_diagram' ? '#000' : 'var(--text-main)',
            fontWeight: 600,
            fontSize: '0.85rem',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '6px',
            transition: 'all 0.2s ease',
          }}
        >
          <Eye size={16} />
          2D Vector Diagram (Main Display)
        </button>

        <button
          onClick={() => setActiveView('picture')}
          style={{
            flex: 1,
            padding: '8px 14px',
            borderRadius: 'var(--radius-sm)',
            border: 'none',
            background: activeView === 'picture' ? 'var(--accent-green)' : 'transparent',
            color: activeView === 'picture' ? '#000' : 'var(--text-main)',
            fontWeight: 600,
            fontSize: '0.85rem',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '6px',
            transition: 'all 0.2s ease',
          }}
        >
          <Camera size={16} />
          Original Photo View
        </button>
      </div>

      {/* Line Color Legend Indicator */}
      <div style={{ display: 'flex', gap: '16px', fontSize: '0.85rem', background: 'rgba(15, 23, 42, 0.6)', padding: '8px 14px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#10B981', display: 'inline-block' }}></span>
          <span><strong>Green Line:</strong> Cue Ball → Target Ball</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#06B6D4', display: 'inline-block' }}></span>
          <span><strong>Blue Line:</strong> Target Ball → Target Pocket</span>
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
            <strong style={{ color: 'var(--accent-yellow)' }}>No valid shots found.</strong> Every possible path is obstructed or misses all pockets. Inspect the 2D Vector Diagram to verify detected ball positions.
          </div>
        </div>
      )}

      {/* Display: 2D Diagram Canvas or Original Photo Overlay */}
      {activeView === 'cv_diagram' ? (
        <TableDiagramCanvas result={result} selectedShotIndex={selectedIndex} />
      ) : (
        <div className="annotated-image-card">
          {result.annotated_image_b64 ? (
            <img 
              src={`data:image/jpeg;base64,${result.annotated_image_b64}`} 
              alt="Pool Table Top-Down Camera View" 
            />
          ) : (
            <div style={{ padding: '40px', textAlign: 'center' }}>Photo view unavailable</div>
          )}
        </div>
      )}

      <ShotList 
        directShots={directShots}
        bankShots={bankShots}
        kickShots={kickShots}
        selectedIndex={selectedIndex}
        onSelectShot={(idx) => {
          setSelectedIndex(idx);
        }}
      />
    </div>
  );
}
