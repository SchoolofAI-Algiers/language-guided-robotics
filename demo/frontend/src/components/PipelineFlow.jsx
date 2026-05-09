// PipelineFlow.jsx — Animated pipeline stages
import React from 'react';

const STAGES = [
  { key: 'nlp',    label: 'NLP',    sub: 'Language Parsing',   icon: '⌨' },
  { key: 'cv',     label: 'CV',     sub: 'Scene Perception',   icon: '◎' },
  { key: 'rl',     label: 'RL',     sub: 'Policy Decision',    icon: '◈' },
  { key: 'action', label: 'ACTION', sub: 'Arm Execution',      icon: '⟳' },
];

function StageBox({ stage, status, result, isLast }) {
  const isRunning = status === 'running';
  const isDone = status === 'done';

  return (
    <div style={{ display: 'flex', alignItems: 'center', flex: 1 }}>
      <div style={{
        flex: 1,
        border: `1.5px solid ${isRunning ? 'var(--orange)' : isDone ? '#22c55e' : 'var(--gray-light)'}`,
        background: isRunning ? 'var(--orange-dim)' : isDone ? '#f0fdf4' : 'var(--white)',
        padding: '10px 14px',
        transition: 'all 0.3s ease',
        position: 'relative',
        overflow: 'hidden',
      }}>
        {/* Scanning animation when running */}
        {isRunning && (
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: 2,
            background: 'linear-gradient(90deg, transparent, var(--orange), transparent)',
            animation: 'scanH 1s ease-in-out infinite',
          }} />
        )}

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <div style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 9,
              color: isRunning ? 'var(--orange)' : isDone ? '#16a34a' : 'var(--gray-text)',
              letterSpacing: '0.12em',
              marginBottom: 2,
            }}>{stage.icon} {stage.label}</div>
            <div style={{
              fontFamily: 'var(--font-display)',
              fontWeight: 700,
              fontSize: 13,
              textTransform: 'uppercase',
            }}>{stage.sub}</div>
          </div>
          <StatusBadge status={status} />
        </div>

        {/* Result snippet */}
        {isDone && result && (
          <div style={{
            marginTop: 6,
            fontFamily: 'var(--font-mono)',
            fontSize: 9,
            color: 'var(--gray-text)',
            lineHeight: 1.5,
          }}>
            {stage.key === 'nlp' && (
              <>
                <div>CMD: <span style={{ color: 'var(--black)', fontWeight: 600 }}>{result.commandType?.toUpperCase()}</span></div>
                <div>CONF: {result.confidence}</div>
                <div>TARGET: {result.target}</div>
              </>
            )}
            {stage.key === 'cv' && (
              <>
                <div>VISIBLE: <span style={{ color: 'var(--black)', fontWeight: 600 }}>{result.n_visible ?? result.detectedObjects?.length ?? 0} objects</span></div>
                <div>MODEL: {result.model || 'ResNet18'}</div>
                <div>FEAT: {result.feature_dim || 512}-dim per object</div>
                {result.error && <div style={{ color: '#ef4444' }}>ERR: {result.error}</div>}
              </>
            )}
            {stage.key === 'rl' && (
              <>
                <div>POSE: <span style={{ color: 'var(--orange)', fontWeight: 600 }}>{result.pose?.toUpperCase()}</span></div>
                <div>REWARD: {result.rewardEstimate}</div>
                <div>P(success): {result.successProbability}</div>
              </>
            )}
          </div>
        )}

        {/* Latency */}
        {isDone && result?.latencyMs && (
          <div style={{
            marginTop: 4,
            fontFamily: 'var(--font-mono)',
            fontSize: 8,
            color: 'var(--gray-mid)',
          }}>{result.latencyMs}ms</div>
        )}
      </div>

      {/* Arrow connector */}
      {!isLast && (
        <div style={{
          width: 24,
          flexShrink: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: isDone ? 'var(--orange)' : 'var(--gray-mid)',
          fontFamily: 'var(--font-mono)',
          fontSize: 14,
          fontWeight: 700,
          transition: 'color 0.3s',
        }}>›</div>
      )}
    </div>
  );
}

function StatusBadge({ status }) {
  const cfg = {
    idle:    { label: 'IDLE',    bg: 'var(--gray-light)',  color: 'var(--gray-text)' },
    running: { label: 'RUNNING', bg: 'var(--orange)',      color: 'white' },
    done:    { label: 'DONE',    bg: '#22c55e',            color: 'white' },
    error:   { label: 'ERROR',   bg: '#ef4444',            color: 'white' },
  }[status || 'idle'];

  return (
    <div style={{
      background: cfg.bg,
      color: cfg.color,
      fontFamily: 'var(--font-mono)',
      fontSize: 8,
      padding: '2px 6px',
      letterSpacing: '0.12em',
      flexShrink: 0,
    }}>
      {cfg.label}
    </div>
  );
}

export default function PipelineFlow({ stages, onNewScene }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>

      {/* Header */}
      <div style={{
        padding: '12px 20px',
        borderBottom: 'var(--border)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <div>
          <div className="label" style={{ marginBottom: 2 }}>Pipeline</div>
          <div className="display-md" style={{ fontSize: 16 }}>Inference Flow</div>
        </div>
        <button
            onClick={onNewScene}
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 9,
              color: 'var(--black)',
              border: '1px solid var(--black)',
              padding: '3px 8px',
              background: 'transparent',
              cursor: 'pointer',
              marginLeft: 8,
            }}
          >
            ↺ NEW SCENE
          </button>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--gray-text)' }}>
          NLP · CV · RL · ACTION
        </div>
      </div>

      {/* Stages */}
      <div style={{
        padding: '16px 20px',
        display: 'flex',
        gap: 0,
        flex: 1,
        alignItems: 'stretch',
      }}>
        {STAGES.map((stage, i) => (
          <StageBox
            key={stage.key}
            stage={stage}
            status={stages[stage.key]?.status}
            result={stages[stage.key]?.result}
            isLast={i === STAGES.length - 1}
          />
        ))}
      </div>

      <style>{`
        @keyframes scanH {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </div>
  );
}
