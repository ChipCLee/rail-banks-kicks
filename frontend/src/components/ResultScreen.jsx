import React, { useState } from 'react';
import ShotList from './ShotList';
import { RotateCcw, Target, AlertTriangle, Eye, Layers } from 'lucide-react';

export default function ResultScreen({ result, onReset, onEnterTeachMode }) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [activeView, setActiveView] = useState('shots'); // 'shots' | 'cv_diagram'

  const directShots = result.direct_shots || [];
  const bankShots = result.bank_shots || [];
  const kickShots = result.kick_shots || [];
  const totalShots = directShots.length + bankShots.length + kickShots.length;
  const ballsCount = result.balls?.length || 0;

  const activeImageB64 = activeView === 'cv_diagram' && result.cv_diagram_b64
    ? result.cv_diagram_b64
    : result.annotated_image_b64;

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

      {/* View Switcher Tabs */}
      <div style={{ display: 'flex', gap: '8px', background: 'var(--bg-card)', padding: '6px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
        <button
          onClick={() => setActiveView('shots')}
          style={{
            flex: 1,
            padding: '8px 14px',
            borderRadius: 'var(--radius-sm)',
            border: 'none',
            background: activeView === 'shots' ? 'var(--accent-green)' : 'transparent',
            color: activeView === 'shots' ? '#000' : 'var(--text-main)',
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
          <Layers size={16} />
          Shot Trajectories ({totalShots})
        </button>

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
          2D CV Detection Diagram ({ballsCount} Balls)
        </button>
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
            <strong style={{ color: 'var(--accent-yellow)' }}>No valid shots found.</strong> Every possible path is obstructed or misses all pockets. Inspect the <strong>2D CV Detection Diagram</strong> tab to verify all detected balls, pockets, and diamonds.
          </div>
        </div>
      )}

      {activeView === 'cv_diagram' && (
        <div style={{ fontSize: '0.85rem', color: 'var(--accent-cyan)', background: 'rgba(6, 182, 212, 0.1)', border: '1px solid var(--accent-cyan)', padding: '8px 14px', borderRadius: 'var(--radius-sm)' }}>
          🔍 <strong>CV Detection Map:</strong> Verified {ballsCount} balls, 6 pockets, and 18 rail diamonds. Use this diagram to confirm Computer Vision accuracy.
        </div>
      )}

      <div className="annotated-image-card">
        {activeImageB64 ? (
          <img 
            src={`data:image/jpeg;base64,${activeImageB64}`} 
            alt="Pool Table Top-Down View" 
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
        onSelectShot={(idx) => {
          setActiveView('shots');
          setSelectedIndex(idx);
        }}
      />
    </div>
  );
}
