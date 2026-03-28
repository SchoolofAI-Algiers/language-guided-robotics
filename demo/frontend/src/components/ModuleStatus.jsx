// ModuleStatus.jsx — Status cards for each team track
import React from 'react';

const MODULES = [
  {
    key: 'robotics',
    label: 'Robotics',
    track: 'M1=–M2',
    status: 'active',
    items: ['PyBullet env', '7-DOF wrapper', 'Gymnasium API', 'IK utilities'],
    phase: 'W1–2',
    completion: 35,
  },
  {
    key: 'vision',
    label: 'Vision / ML',
    track: 'M3–M4',
    status: 'active',
    items: ['CNN extractor', 'Object detection', 'Depth localization', 'Dataset pipeline'],
    phase: 'W3–4',
    completion: 20,
  },
  {
    key: 'nlp',
    label: 'NLP',
    track: 'M5–M6',
    status: 'active',
    items: ['sentence-transformers', 'Instruction dataset', 'Embedding cache', 'Similarity tests'],
    phase: 'W5–6',
    completion: 25,
  },
  {
    key: 'rl',
    label: 'RL',
    track: 'M7–M8',
    status: 'pending',
    items: ['PPO loop', 'Reward shaping', 'Curriculum', 'Eval scripts'],
    phase: 'W5–6',
    completion: 10,
  },
  {
    key: 'demo',
    label: 'Demo / Systems',
    track: 'M9–M10',
    status: 'active',
    items: ['Flask backend', 'React frontend', 'Video stream', 'Logging'],
    phase: 'W1–2',
    completion: 40,
  },
];

function ModuleCard({ mod }) {
  const isActive = mod.status === 'active';
  return (
    <div style={{
      background: 'var(--white)',
      border: 'var(--border)',
      padding: '14px 16px',
      position: 'relative',
      overflow: 'hidden',
    }}>
      {/* Orange left accent */}
      <div style={{
        position: 'absolute',
        left: 0,
        top: 0,
        bottom: 0,
        width: 3,
        background: isActive ? 'var(--orange)' : 'var(--gray-light)',
      }} />

      {/* Header row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
        <div>
          <div style={{
            fontFamily: 'var(--font-display)',
            fontWeight: 800,
            fontSize: 15,
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
          }}>{mod.label}</div>
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 9,
            color: 'var(--gray-text)',
            marginTop: 1,
          }}>{mod.track} · {mod.phase}</div>
        </div>
        <div style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 9,
          padding: '2px 7px',
          background: isActive ? 'var(--orange)' : 'var(--gray-light)',
          color: isActive ? 'white' : 'var(--gray-text)',
          letterSpacing: '0.1em',
        }}>
          {isActive ? 'ACTIVE' : 'PENDING'}
        </div>
      </div>

      {/* Progress bar */}
      <div style={{ marginBottom: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--gray-text)' }}>PROGRESS</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, fontWeight: 600 }}>{mod.completion}%</span>
        </div>
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{ width: `${mod.completion}%` }}
          />
        </div>
      </div>

      {/* Task list */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        {mod.items.map((item, i) => (
          <div key={i} style={{
            display: 'flex',
            alignItems: 'center',
            gap: 7,
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--gray-text)',
          }}>
            <span style={{ color: i < 2 ? 'var(--orange)' : 'var(--gray-mid)', fontSize: 8 }}>
              {i < 2 ? '✓' : '○'}
            </span>
            {item}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ModuleStatus() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{
        padding: '12px 20px',
        borderBottom: 'var(--border)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <div>
          <div className="label" style={{ marginBottom: 2 }}>Team Tracks</div>
          <div className="display-md" style={{ fontSize: 16 }}>Module Status</div>
        </div>
        <div style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 9,
          color: 'var(--gray-text)',
        }}>Team members</div>
      </div>

      <div style={{
        padding: '16px',
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
        gap: 12,
        overflowY: 'auto',
        flex: 1,
      }}>
        {MODULES.map((mod) => (
          <ModuleCard key={mod.key} mod={mod} />
        ))}
      </div>
    </div>
  );
}
