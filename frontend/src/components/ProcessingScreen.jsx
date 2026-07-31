import React from 'react';

export default function ProcessingScreen() {
  return (
    <div className="processing-box">
      <div className="spinner"></div>
      <div>
        <h3 style={{ fontFamily: 'var(--font-heading)' }}>Analyzing Table Geometry...</h3>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '6px' }}>
          Detecting table boundary, ball positions, and calculating bank trajectories.
        </p>
      </div>
    </div>
  );
}
