// ExecutionLog.jsx — Scrolling execution log
import React, { useEffect, useRef } from 'react';

function LogEntry({ entry }) {
  const typeStyles = {
    system:  { color: '#6b7280', prefix: 'SYS' },
    nlp:     { color: '#3b82f6', prefix: 'NLP' },
    cv:      { color: '#8b5cf6', prefix: 'CV ' },
    rl:      { color: '#f59e0b', prefix: 'RL ' },
    action:  { color: 'var(--orange)', prefix: 'ACT' },
    success: { color: '#22c55e', prefix: 'OK ' },
    error:   { color: '#ef4444', prefix: 'ERR' },
  };
  const style = typeStyles[entry.type] || typeStyles.system;

  return (
    <div style={{
      display: 'flex',
      gap: 12,
      fontFamily: 'var(--font-mono)',
      fontSize: 11,
      lineHeight: 1.6,
      padding: '3px 0',
      borderBottom: '1px solid var(--gray-light)',
      animation: 'fadeInUp 0.2s ease forwards',
    }}>
      <span style={{ color: 'var(--gray-mid)', flexShrink: 0, fontSize: 9 }}>
        {entry.timestamp}
      </span>
      <span style={{
        color: style.color,
        fontWeight: 600,
        flexShrink: 0,
        fontSize: 9,
        letterSpacing: '0.08em',
        minWidth: 24,
      }}>
        {style.prefix}
      </span>
      <span style={{ color: entry.type === 'error' ? '#ef4444' : 'var(--black)' }}>
        {entry.message}
      </span>
    </div>
  );
}

export default function ExecutionLog({ logs }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{
        padding: '10px 20px',
        borderBottom: 'var(--border)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <div>
          <div className="label" style={{ marginBottom: 1 }}>Output</div>
          <div className="display-md" style={{ fontSize: 15 }}>Execution Log</div>
        </div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--gray-text)' }}>
            {logs.length} ENTRIES
          </span>
          <div style={{
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: 'var(--orange)',
            animation: 'pulse-dot 2s ease-in-out infinite',
          }} />
        </div>
      </div>

      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '12px 20px',
        background: '#FAFAF8',
      }}>
        {logs.length === 0 && (
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            color: 'var(--gray-mid)',
            paddingTop: 8,
          }}>
            — awaiting instruction —
          </div>
        )}
        {logs.map((entry, i) => (
          <LogEntry key={i} entry={entry} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
