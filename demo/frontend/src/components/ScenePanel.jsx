// ScenePanel.jsx — CV perception / detected objects display
import React from 'react';

const OBJECTS_DEFAULT = [
  { id: 'obj_0', label: 'red block',     color: '#EF4444', x: 0.32,  z: 0.21, conf: null },
  { id: 'obj_1', label: 'green cube',    color: '#22C55E', x: -0.18, z: 0.28, conf: null },
  { id: 'obj_2', label: 'blue cylinder', color: '#3B82F6', x: 0.05,  z: -0.22, conf: null },
  { id: 'obj_3', label: 'yellow sphere', color: '#EAB308', x: -0.28, z: 0.10, conf: null },
];

function MiniTopView({ objects, targetId }) {
  const size = 120;
  const scale = 160;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {/* Grid */}
      {[0, 1, 2, 3, 4].map(i => (
        <React.Fragment key={i}>
          <line x1={i * size / 4} y1={0} x2={i * size / 4} y2={size} stroke="#e8e8e0" strokeWidth="0.5" />
          <line x1={0} y1={i * size / 4} x2={size} y2={i * size / 4} stroke="#e8e8e0" strokeWidth="0.5" />
        </React.Fragment>
      ))}
      {/* Arm base */}
      <circle cx={size / 2} cy={size / 2} r={6} fill="#0A0A0A" />
      <circle cx={size / 2} cy={size / 2} r={10} fill="none" stroke="#0A0A0A" strokeWidth="0.5" strokeDasharray="2,2" />

      {/* Objects */}
      {objects.map((obj) => {
        const px = size / 2 + obj.x * scale * 0.3;
        const py = size / 2 - obj.z * scale * 0.3;
        const isTarget = obj.id === targetId;
        return (
          <g key={obj.id}>
            {isTarget && (
              <circle cx={px} cy={py} r={10} fill="none" stroke="var(--orange)" strokeWidth="1" opacity="0.6" />
            )}
            <rect
              x={px - 5}
              y={py - 5}
              width={10}
              height={10}
              fill={obj.color}
              stroke={isTarget ? 'var(--orange)' : 'none'}
              strokeWidth={isTarget ? 1.5 : 0}
            />
          </g>
        );
      })}

      {/* Axes labels */}
      <text x={size - 6} y={size / 2 + 3} fontSize={7} fill="#aaa" textAnchor="end">x</text>
      <text x={size / 2 + 3} y={8} fontSize={7} fill="#aaa">z</text>
    </svg>
  );
}

export default function ScenePanel({ cvResult }) {
  const objects = cvResult?.detectedObjects || OBJECTS_DEFAULT;
  const targetId = cvResult?.targetObject?.id || null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{
        padding: '10px 16px',
        borderBottom: 'var(--border)',
      }}>
        <div className="label" style={{ marginBottom: 1 }}>CV Module</div>
        <div className="display-md" style={{ fontSize: 15 }}>Scene Perception</div>
      </div>

      <div style={{ padding: '12px 16px', borderBottom: 'var(--border)', display: 'flex', gap: 16, alignItems: 'flex-start' }}>
        {/* Top-down view */}
        <div>
          <div className="label" style={{ marginBottom: 6, fontSize: 8 }}>TOP VIEW</div>
          <div style={{ border: 'var(--border)', background: '#FAFAF8' }}>
            <MiniTopView objects={objects} targetId={targetId} />
          </div>
        </div>

        {/* Object list */}
        <div style={{ flex: 1 }}>
          <div className="label" style={{ marginBottom: 6, fontSize: 8 }}>DETECTED OBJECTS</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
            {objects.map((obj) => {
              const isTarget = obj.id === targetId;
              return (
                <div key={obj.id} style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '4px 8px',
                  background: isTarget ? 'var(--orange-dim)' : 'transparent',
                  border: `1px solid ${isTarget ? 'var(--orange)' : 'var(--gray-light)'}`,
                  transition: 'all 0.2s',
                }}>
                  <div style={{ width: 10, height: 10, background: obj.color, flexShrink: 0 }} />
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, flex: 1 }}>
                    {obj.label}
                  </div>
                  {isTarget && (
                    <div style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: 8,
                      color: 'var(--orange)',
                      fontWeight: 600,
                      letterSpacing: '0.1em',
                    }}>TARGET</div>
                  )}
                  {obj.conf && (
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--gray-text)' }}>
                      {(obj.conf * 100).toFixed(0)}%
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Position data */}
      {cvResult && (
        <div style={{ padding: '10px 16px' }}>
          <div className="label" style={{ marginBottom: 6, fontSize: 8 }}>TARGET POSITION</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6 }}>
            {['x', 'y', 'z'].map((axis, i) => (
              <div key={axis} style={{
                background: 'var(--gray-bg)',
                border: 'var(--border)',
                padding: '6px 8px',
                textAlign: 'center',
              }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 8, color: 'var(--gray-text)', marginBottom: 2 }}>{axis.toUpperCase()}</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 600 }}>
                  {[cvResult.targetObject?.x, 0.14, cvResult.targetObject?.z][i]?.toFixed(2) || '0.00'}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
