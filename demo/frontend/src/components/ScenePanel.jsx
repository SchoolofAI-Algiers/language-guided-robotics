// ScenePanel.jsx — CV perception / detected objects display
// Phase 2: uses real detected objects from Flask/Vision pipeline
// Shows empty state when no instruction has been run yet
import React from 'react';

const COLOR_MAP = {
  red:    '#EF4444',
  green:  '#22C55E',
  blue:   '#3B82F6',
  yellow: '#EAB308',
  cyan:   '#06B6D4',
};

function MiniTopView({ objects }) {
  const size = 120;
  // table center in world coords is ~[0.5, 0.0]
  // map to SVG: arm base at left, table in center-right
  const worldToSvg = (x, z) => ({
    px: size * 0.15 + (x) * size * 0.55,
    py: size * 0.5  - (z) * size * 0.6,
  });

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {/* Grid */}
      {[0, 1, 2, 3, 4].map(i => (
        <React.Fragment key={i}>
          <line x1={i * size / 4} y1={0} x2={i * size / 4} y2={size} stroke="#e8e8e0" strokeWidth="0.5" />
          <line x1={0} y1={i * size / 4} x2={size} y2={i * size / 4} stroke="#e8e8e0" strokeWidth="0.5" />
        </React.Fragment>
      ))}

      {/* Arm base at origin [0,0] */}
      <circle cx={size * 0.15} cy={size * 0.5} r={5} fill="#0A0A0A" />
      <circle cx={size * 0.15} cy={size * 0.5} r={8} fill="none" stroke="#0A0A0A" strokeWidth="0.5" strokeDasharray="2,2" />

      {/* Table outline at ~[0.5, 0.0] ±0.22 */}
      {(() => {
        const tl = worldToSvg(0.5 - 0.22, -0.22);
        const br = worldToSvg(0.5 + 0.22,  0.22);
        return (
          <rect
            x={tl.px} y={tl.py}
            width={br.px - tl.px}
            height={br.py - tl.py}
            fill="rgba(0,0,0,0.03)"
            stroke="#cccccc"
            strokeWidth="0.8"
            strokeDasharray="3,2"
          />
        );
      })()}

      {/* Objects */}
      {objects.map((obj, i) => {
        const pos = obj.pos || [0, 0, 0];
        const { px, py } = worldToSvg(pos[0], pos[1]);
        const fill = COLOR_MAP[obj.color] || '#888';
        const isCircle = obj.shape === 'sphere';

        return (
          <g key={obj.id || i}>
            {isCircle
              ? <circle cx={px} cy={py} r={5} fill={fill} stroke="white" strokeWidth="0.5" />
              : <rect x={px - 5} y={py - 5} width={10} height={10} fill={fill} stroke="white" strokeWidth="0.5" />
            }
          </g>
        );
      })}

      {/* Axes labels */}
      <text x={size - 4} y={size * 0.5 + 3} fontSize={7} fill="#aaa" textAnchor="end">x</text>
      <text x={size * 0.15 + 3} y={8} fontSize={7} fill="#aaa">y</text>
    </svg>
  );
}

export default function ScenePanel({ cvResult }) {
  const isReal = cvResult && cvResult.detected_objects !== undefined;
  const isError = cvResult?.error;
  const objects = isReal ? cvResult.detected_objects : [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>

      {/* Header */}
      <div style={{
        padding: '10px 16px',
        borderBottom: 'var(--border)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <div>
          <div className="label" style={{ marginBottom: 1 }}>CV Module</div>
          <div className="display-md" style={{ fontSize: 15 }}>Scene Perception</div>
        </div>
        {isReal && (
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 9,
            color: '#22c55e',
            border: '1px solid #22c55e',
            padding: '2px 8px',
          }}>
            LIVE · {cvResult.model || 'ResNet18'}
          </div>
        )}
        {!isReal && !isError && (
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 9,
            color: 'var(--gray-text)',
            border: '1px solid var(--gray-light)',
            padding: '2px 8px',
          }}>
            AWAITING
          </div>
        )}
        {isError && (
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 9,
            color: '#ef4444',
            border: '1px solid #ef4444',
            padding: '2px 8px',
          }}>
            ERROR
          </div>
        )}
      </div>

      {/* Error message */}
      {isError && (
        <div style={{
          padding: '8px 16px',
          fontFamily: 'var(--font-mono)',
          fontSize: 10,
          color: '#ef4444',
          borderBottom: 'var(--border)',
        }}>
          {cvResult.error}
        </div>
      )}

      {/* Top view + object list */}
      <div style={{
        padding: '12px 16px',
        borderBottom: 'var(--border)',
        display: 'flex',
        gap: 10,
        alignItems: 'flex-start',
      }}>
        {/* Top-down view */}
        <div>
          <div className="label" style={{ marginBottom: 6, fontSize: 8 }}>TOP VIEW</div>
          <div style={{ border: 'var(--border)', background: '#FAFAF8' }}>
            <MiniTopView objects={objects} />
          </div>
        </div>

        {/* Object list */}
        <div style={{ flex: 1 }}>
          <div className="label" style={{ marginBottom: 6, fontSize: 8 }}>
            DETECTED OBJECTS {isReal ? `(${cvResult.n_visible} visible)` : ''}
          </div>

          {/* Empty state */}
          {objects.length === 0 && !isError && (
            <div style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 10,
              color: 'var(--gray-text)',
              padding: '8px 0',
              lineHeight: 1.6,
            }}>
              {isReal 
                ? '— no objects on table —'
                : '— run an instruction<br />to detect objects —'
              }
            </div>
          )}

          {/* Real objects */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
            {objects.map((obj, i) => {
              const colorHex = COLOR_MAP[obj.color] || '#888';
              const label = `${obj.color} ${obj.shape}`;

              return (
                <div key={obj.id || i} style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '4px 8px',
                  border: '1px solid var(--gray-light)',
                }}>
                  {/* Color + shape indicator */}
                  <div style={{
                    width: 10,
                    height: 10,
                    background: colorHex,
                    flexShrink: 0,
                    borderRadius: obj.shape === 'sphere' ? '50%' : obj.shape === 'cylinder' ? '2px' : 0,
                  }} />
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, flex: 1 }}>
                    {label}
                  </div>
                  {obj.pos && (
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 8, color: 'var(--gray-text)' }}>
                      ({obj.pos[0]?.toFixed(2)}, {obj.pos[1]?.toFixed(2)})
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Position data — only when real data available */}
      {isReal && objects.length > 0 && (
        <div style={{ padding: '10px 16px' }}>
          <div className="label" style={{ marginBottom: 6, fontSize: 8 }}>
            FIRST OBJECT POSITION (m)
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6 }}>
            {['x', 'y', 'z'].map((axis, i) => {
              const val = objects[0]?.pos?.[i] ?? 0;
              return (
                <div key={axis} style={{
                  background: 'var(--gray-bg)',
                  border: 'var(--border)',
                  padding: '6px 8px',
                  textAlign: 'center',
                }}>
                  <div style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: 8,
                    color: 'var(--gray-text)',
                    marginBottom: 2,
                  }}>
                    {axis.toUpperCase()}
                  </div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 600 }}>
                    {typeof val === 'number' ? val.toFixed(2) : '0.00'}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Empty position placeholder */}
      {!isReal && (
        <div style={{ padding: '10px 16px' }}>
          <div className="label" style={{ marginBottom: 6, fontSize: 8 }}>TARGET POSITION</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6 }}>
            {['X', 'Y', 'Z'].map((axis) => (
              <div key={axis} style={{
                background: 'var(--gray-bg)',
                border: 'var(--border)',
                padding: '6px 8px',
                textAlign: 'center',
              }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 8, color: 'var(--gray-text)', marginBottom: 2 }}>
                  {axis}
                </div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 600, color: 'var(--gray-mid)' }}>
                  —
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  );
}