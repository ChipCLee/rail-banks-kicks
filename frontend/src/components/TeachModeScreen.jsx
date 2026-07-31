import React, { useRef } from 'react';
import { Target, HelpCircle, RotateCcw } from 'lucide-react';

export default function TeachModeScreen({ result, onSetCueBallLocation, onReset }) {
  const imgRef = useRef(null);

  const handleImageClick = (e) => {
    if (!imgRef.current || !result.table_dims_mm) return;
    const rect = imgRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const ratioX = Math.max(0, Math.min(1, clickX / rect.width));
    const ratioY = Math.max(0, Math.min(1, clickY / rect.height));

    const tableW = result.table_dims_mm.width;
    const tableH = result.table_dims_mm.height;

    // Convert pixel click to table mm (y=0 at bottom)
    const mmX = ratioX * tableW;
    const mmY = (1 - ratioY) * tableH;

    // Check if user tapped near an existing detected object ball
    const balls = result.balls || [];
    const tapRadius = 60; // mm
    const tappedBall = balls.find(b => Math.hypot(b.x - mmX, b.y - mmY) <= tapRadius);

    if (tappedBall) {
      onSetCueBallLocation({ manual_cue_ball_id: tappedBall.id });
    } else {
      onSetCueBallLocation({ manual_cue_x: mmX, manual_cue_y: mmY });
    }
  };

  return (
    <div className="teach-mode-container">
      <div className="teach-banner">
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Target size={28} color="var(--accent-yellow)" />
          <div>
            <h3 style={{ fontFamily: 'var(--font-heading)', color: 'var(--accent-yellow)' }}>
              Teach Mode — Select Cue Ball
            </h3>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginTop: '2px' }}>
              Cue ball was not automatically detected. Tap anywhere on the table to set the Cue Ball location, or tap an identified ball.
            </p>
          </div>
        </div>
        <button 
          onClick={onReset}
          style={{
            background: 'transparent',
            border: '1px solid var(--border-color)',
            color: 'var(--text-main)',
            padding: '6px 14px',
            borderRadius: 'var(--radius-sm)',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '0.85rem',
            whiteSpace: 'nowrap',
          }}
        >
          <RotateCcw size={14} />
          New Photo
        </button>
      </div>

      <div className="annotated-image-card" style={{ position: 'relative', cursor: 'crosshair', marginTop: '16px' }}>
        {result.annotated_image_b64 && (
          <img 
            ref={imgRef}
            src={`data:image/jpeg;base64,${result.annotated_image_b64}`} 
            alt="Table top-down view in Teach Mode"
            onClick={handleImageClick}
            style={{ width: '100%', height: 'auto', display: 'block' }}
          />
        )}
      </div>

      <div style={{ marginTop: '16px', display: 'flex', gap: '16px', flexWrap: 'wrap', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '(50,50,50)', border: '1px solid #fff' }}></span>
          <span>Pockets (TL, TR, ML, MR, BL, BR)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: '10px', height: '10px', transform: 'rotate(45deg)', background: 'var(--accent-cyan)' }}></span>
          <span>Rail Diamonds (0.5 – 4.0)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#ef4444' }}></span>
          <span>Identified Object Balls ({result.balls?.length || 0})</span>
        </div>
      </div>
    </div>
  );
}
