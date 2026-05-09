// Header.jsx — Top navigation bar
import React from 'react';

export default function Header({ phase, systemStatus }) {
  return (
    <header style={{
      background: 'var(--white)',
      borderBottom: 'var(--border)',
      display: 'flex',
      alignItems: 'stretch',
      height: 56,
    }}>
      {/* Logo cell */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '0 24px',
        borderRight: 'var(--border)',
        minWidth: 220,
      }}>
        <img src="../public/Logo_SOAI.png" alt="SOAI Labs" style={{ width: 32, height: 32 }} />
        <div>
          <div style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 13, letterSpacing: '0.1em', lineHeight: 1 }}>SOAI LABS</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--gray-text)', letterSpacing: '0.1em' }}>2026 Edition</div>
        </div>
      </div>

      {/* Project title */}
      <div style={{
        flex: 1,
        display: 'flex',
        alignItems: 'center',
        padding: '0 24px',
        borderRight: 'var(--border)',
      }}>
        <span style={{
          fontFamily: 'var(--font-display)',
          fontWeight: 700,
          fontSize: 15,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
        }}>
          Language-Guided Robotics
        </span>
        
      </div>

      {/* Status indicators */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 0,
      }}>
        {[
          { label: 'NLP', key: 'nlp' },
          { label: 'CV', key: 'cv' },
          { label: 'RL', key: 'rl' },
          { label: 'SIM', key: 'sim' },
        ].map(({ label, key }) => (
          <div key={key} style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '0 16px',
            borderLeft: 'var(--border)',
            height: '100%',
          }}>
            <span className="dot" style={{
              background: systemStatus[key] === 'active' ? 'var(--orange)'
                : systemStatus[key] === 'ready' ? '#22c55e'
                : 'var(--gray-mid)',
              animation: systemStatus[key] === 'active' ? 'pulse-dot 1.5s ease-in-out infinite' : 'none',
            }} />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '0.1em' }}>
              {label}
            </span>
          </div>
        ))}
      </div>

      {/* Lead badge */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        padding: '0 20px',
        borderLeft: 'var(--border)',
        gap: 8,
      }}>
        <div style={{
          width: 28,
          height: 28,
          background: 'var(--orange)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontFamily: 'var(--font-display)',
          fontWeight: 800,
          fontSize: 13,
          color: 'white',
        }}>SK</div>
        <div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--gray-text)' }}>PROJECT LEAD</div>
          <div style={{ fontFamily: 'var(--font-body)', fontWeight: 600, fontSize: 12 }}>Selma Khelili</div>
        </div>
      </div>
    </header>
  );
}
