import React from 'react';
import { Target, CornerDownRight, Zap } from 'lucide-react';

export default function ShotList({ directShots = [], bankShots = [], kickShots = [], selectedIndex, onSelectShot }) {
  let currentIndex = 0;

  return (
    <div className="shot-list-container">
      {directShots.length > 0 && (
        <div className="shot-group">
          <div className="group-title">
            <Zap size={18} color="var(--accent-green)" />
            <span>Direct Shots ({directShots.length})</span>
          </div>
          {directShots.map((shot, idx) => {
            const globalIdx = currentIndex++;
            const isSelected = selectedIndex === globalIdx;
            return (
              <div 
                key={`direct-${idx}`}
                className={`shot-card ${isSelected ? 'active' : ''}`}
                onClick={() => onSelectShot(globalIdx)}
              >
                <div>
                  <strong style={{ textTransform: 'capitalize' }}>{shot.object_ball_label} ball</strong>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                    Direct into <strong>{shot.pocket_id}</strong> pocket
                  </div>
                </div>
                <span className="badge-ease" style={{ background: 'rgba(16, 185, 129, 0.2)', color: 'var(--accent-green)' }}>
                  Direct
                </span>
              </div>
            );
          })}
        </div>
      )}

      {bankShots.length > 0 && (
        <div className="shot-group">
          <div className="group-title">
            <CornerDownRight size={18} color="var(--accent-cyan)" />
            <span>Bank Shots ({bankShots.length})</span>
          </div>
          {bankShots.map((shot, idx) => {
            const globalIdx = currentIndex++;
            const isSelected = selectedIndex === globalIdx;
            return (
              <div 
                key={`bank-${idx}`}
                className={`shot-card ${isSelected ? 'active' : ''}`}
                onClick={() => onSelectShot(globalIdx)}
              >
                <div>
                  <strong style={{ textTransform: 'capitalize' }}>{shot.object_ball_label} ball</strong>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                    Via <strong>{shot.rail}</strong> rail → <strong>{shot.pocket_id}</strong> pocket
                  </div>
                  {shot.throw_correction_deg && (
                    <div style={{ fontSize: '0.75rem', color: 'var(--accent-purple)', marginTop: '2px' }}>
                      Throw correction: {shot.throw_correction_deg}° (Rebound: {shot.adjusted_rebound_angle_deg}°)
                    </div>
                  )}
                </div>
                <span className="badge-ease">
                  {shot.bank_angle_deg}° (Ease: {shot.ease_score})
                </span>
              </div>
            );
          })}
        </div>
      )}

      {kickShots.length > 0 && (
        <div className="shot-group">
          <div className="group-title">
            <Target size={18} color="var(--accent-purple)" />
            <span>Kick Shots ({kickShots.length})</span>
          </div>
          {kickShots.map((shot, idx) => {
            const globalIdx = currentIndex++;
            const isSelected = selectedIndex === globalIdx;
            return (
              <div 
                key={`kick-${idx}`}
                className={`shot-card ${isSelected ? 'active' : ''}`}
                onClick={() => onSelectShot(globalIdx)}
              >
                <div>
                  <strong style={{ textTransform: 'capitalize' }}>Cue → {shot.object_ball_label} ball</strong>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                    {shot.diamond_label} → <strong>{shot.pocket_id}</strong> pocket
                  </div>
                </div>
                <span className="badge-ease" style={{ background: 'rgba(168, 85, 247, 0.2)', color: 'var(--accent-purple)' }}>
                  Kick
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
