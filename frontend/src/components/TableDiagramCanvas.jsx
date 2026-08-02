import React, { useRef, useEffect } from 'react';

export default function TableDiagramCanvas({ result, selectedShotIndex }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !result) return;

    const ctx = canvas.getContext('2d');
    const dims = result.table_dims_mm || { width: 2540.0, height: 1270.0 };

    // Standardized horizontal landscape dimensions
    const pfWidthMm = dims.width;   // 2540.0 mm (Long dimension)
    const pfHeightMm = dims.height; // 1270.0 mm (Short dimension)

    // Base layout units (Standardized 2:1 horizontal aspect ratio)
    const margin = 46;
    const pfW = 920;
    const pfH = 460;

    canvas.width = pfW + 2 * margin;
    canvas.height = pfH + 2 * margin + 28; // +28 for top banner

    const pfX1 = margin;
    const pfY1 = margin + 28;
    const pfX2 = margin + pfW;
    const pfY2 = margin + 28 + pfH;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Draw Wood Cap Rail Border
    ctx.fillStyle = '#1B2333';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // 2. Draw Simonis 860 Tournament Blue Playfield
    ctx.fillStyle = '#0072CE';
    ctx.fillRect(pfX1, pfY1, pfW, pfH);
    ctx.strokeStyle = '#004B9A';
    ctx.lineWidth = 4;
    ctx.strokeRect(pfX1, pfY1, pfW, pfH);

    // Coordinate mapping (Table mm space -> Canvas px space: x horizontal, y vertical)
    const mmToCanvas = (xMm, yMm) => {
      const cX = pfX1 + (xMm / pfWidthMm) * pfW;
      const cY = pfY2 - (yMm / pfHeightMm) * pfH;
      return { x: cX, y: cY };
    };

    // 3. Draw Head String Line & Foot Spot
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();

    const hsStart = mmToCanvas(635.0, 0.0);
    const hsEnd = mmToCanvas(635.0, pfHeightMm);
    ctx.moveTo(hsStart.x, hsStart.y);
    ctx.lineTo(hsEnd.x, hsEnd.y);
    ctx.stroke();
    ctx.setLineDash([]); // Reset line dash

    // Foot spot
    const fsPt = mmToCanvas(1905.0, 635.0);
    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath();
    ctx.arc(fsPt.x, fsPt.y, 3.5, 0, 2 * Math.PI);
    ctx.fill();

    // 4. Draw Rail Diamonds
    const diamonds = result.diamonds || [];
    diamonds.forEach((d) => {
      const pt = mmToCanvas(d.x, d.y);
      const sz = 4;
      ctx.fillStyle = '#00DCFF';
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pt.x, pt.y - sz);
      ctx.lineTo(pt.x + sz, pt.y);
      ctx.lineTo(pt.x, pt.y + sz);
      ctx.lineTo(pt.x - sz, pt.y);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = '#FFFFFF';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      let textX = pt.x;
      let textY = pt.y;

      if (d.rail === 'TOP') textY -= 12;          // Top horizontal long rail
      else if (d.rail === 'BOTTOM') textY += 12;  // Bottom horizontal long rail
      else if (d.rail === 'LEFT') textX -= 14;    // Left vertical short rail
      else if (d.rail === 'RIGHT') textX += 14;   // Right vertical short rail

      ctx.fillText(String(d.number), textX, textY);
    });

    // 5. Draw 6 Pockets
    const pockets = result.pockets || [];
    pockets.forEach((pkt) => {
      const pt = mmToCanvas(pkt.x, pkt.y);
      ctx.fillStyle = '#111111';
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 14, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = '#E2E8F0';
      ctx.font = 'bold 9px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(pkt.id, pt.x, pt.y);
    });

    // All available shots
    const directShots = result.direct_shots || [];
    const bankShots = result.bank_shots || [];
    const kickShots = result.kick_shots || [];
    const allShots = [...directShots, ...bankShots, ...kickShots];

    const selectedShot = (selectedShotIndex !== undefined && selectedShotIndex !== null && allShots[selectedShotIndex])
      ? allShots[selectedShotIndex]
      : null;

    // Helper: Draw Arrow Line
    const drawArrowLine = (p1, p2, color, width = 3.5, dashed = false) => {
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = width;
      if (dashed) ctx.setLineDash([8, 6]);

      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
      if (dashed) ctx.setLineDash([]);

      // Draw Arrow Head at p2
      const angle = Math.atan2(p2.y - p1.y, p2.x - p1.x);
      const headLen = 12;
      ctx.beginPath();
      ctx.moveTo(p2.x, p2.y);
      ctx.lineTo(p2.x - headLen * Math.cos(angle - Math.PI / 6), p2.y - headLen * Math.sin(angle - Math.PI / 6));
      ctx.lineTo(p2.x - headLen * Math.cos(angle + Math.PI / 6), p2.y - headLen * Math.sin(angle + Math.PI / 6));
      ctx.closePath();
      ctx.fill();
    };

    // 6. Draw Selected Shot Trajectory Lines
    if (selectedShot && selectedShot.path && selectedShot.path.length >= 2) {
      const pathPts = selectedShot.path.map((pt) => mmToCanvas(pt.x, pt.y));

      // Highlight target pocket
      const targetPocket = pockets.find((p) => p.id === selectedShot.pocket_id);
      if (targetPocket) {
        const tPktPt = mmToCanvas(targetPocket.x, targetPocket.y);
        ctx.strokeStyle = '#D30094'; // Purple glow ring
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.arc(tPktPt.x, tPktPt.y, 20, 0, 2 * Math.PI);
        ctx.stroke();
      }

      if (selectedShot.shot_type === 'direct' && pathPts.length >= 3) {
        // Direct Shot:
        // Cue -> Target Ball: GREEN LINE (#10B981)
        drawArrowLine(pathPts[0], pathPts[1], '#10B981', 3.5);
        // Target Ball -> Pocket: BLUE LINE (#06B6D4)
        drawArrowLine(pathPts[1], pathPts[2], '#06B6D4', 3.5);
      } else if (selectedShot.shot_type === 'one_bank' && pathPts.length >= 4) {
        // Bank Shot:
        // Cue -> Target Ball: GREEN LINE (#10B981)
        drawArrowLine(pathPts[0], pathPts[1], '#10B981', 3.5);
        // Target Ball -> Rail: BLUE LINE (#06B6D4)
        drawArrowLine(pathPts[1], pathPts[2], '#06B6D4', 3.5, true);
        // Rail contact dot
        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.arc(pathPts[2].x, pathPts[2].y, 5, 0, 2 * Math.PI);
        ctx.fill();
        // Rail -> Pocket: BLUE LINE (#06B6D4)
        drawArrowLine(pathPts[2], pathPts[3], '#06B6D4', 3.5, true);
      } else if (selectedShot.shot_type === 'one_rail_kick' && pathPts.length >= 4) {
        // Kick Shot:
        // Cue -> Rail -> Target Ball: GREEN LINE (#10B981)
        drawArrowLine(pathPts[0], pathPts[1], '#10B981', 3.5);
        drawArrowLine(pathPts[1], pathPts[2], '#10B981', 3.5);
        // Target Ball -> Pocket: BLUE LINE (#06B6D4)
        drawArrowLine(pathPts[2], pathPts[3], '#06B6D4', 3.5, true);
      }
    }

    // 7. Draw Detected Balls
    const balls = result.balls || [];
    const ballR = 11;

    balls.forEach((ball) => {
      const pt = mmToCanvas(ball.x, ball.y);
      const isSelectedTarget = selectedShot && selectedShot.object_ball_id === ball.id;

      // Target Ball Highlight
      if (isSelectedTarget) {
        ctx.strokeStyle = '#00D7FF';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, ballR + 4, 0, 2 * Math.PI);
        ctx.stroke();
      }

      if (ball.label === 'cue') {
        ctx.fillStyle = '#FFFFFF';
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, ballR, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#000000';
        ctx.font = 'bold 9px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('CUE', pt.x, pt.y);
      } else {
        let color = '#EF4444'; // red
        if (ball.label.includes('blue')) color = '#3B82F6';
        if (ball.label.includes('orange')) color = '#F97316';
        if (ball.label.includes('yellow')) color = '#EAB308';
        if (ball.label.includes('green')) color = '#22C55E';
        if (ball.label.includes('purple')) color = '#A855F7';
        if (ball.label === 'eight') color = '#1E293B';

        ctx.fillStyle = color;
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, ballR, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#FFFFFF';
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(ball.label.replace('solid-', '').replace('stripe-', ''), pt.x, pt.y + ballR + 9);
      }
    });

    // 8. Top Banner Header
    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, canvas.width, 28);
    ctx.fillStyle = '#00DCFF';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(
      `2D Diagram (9ft Simonis 860 Blue) | ${balls.length} Balls | Long Rails: Top & Bottom (3 Pockets, 6 Diamonds) | Short Rails: Left & Right (2 Pockets, 3 Diamonds)`,
      12,
      14
    );
  }, [result, selectedShotIndex]);

  return (
    <div style={{ width: '100%', borderRadius: 'var(--radius-lg)', overflow: 'hidden', border: '1px solid var(--border-color)', background: '#1B2333' }}>
      <canvas
        ref={canvasRef}
        style={{
          width: '100%',
          height: 'auto',
          display: 'block',
        }}
      />
    </div>
  );
}
