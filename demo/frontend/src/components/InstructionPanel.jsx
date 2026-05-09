// InstructionPanel.jsx — User instruction input
import React, { useState } from 'react';
import { EXAMPLE_COMMANDS } from '../lib/simulator.js';

export default function InstructionPanel({ onSubmit, isRunning, onNewScene }) {
  const [instruction, setInstruction] = useState('');

  const handleSubmit = () => {
    if (!instruction.trim() || isRunning) return;
    onSubmit(instruction.trim());
    setInstruction('');
  };

  const handleNewScene = async () => {
    if (onNewScene) {
      await onNewScene();
    }
  };

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

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
          <div className="label" style={{ marginBottom: 2 }}>Input Module</div>
          <div className="display-md" style={{ fontSize: 16 }}>Natural Language Instruction</div>
        </div>
        <div style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 9,
          color: 'var(--gray-text)',
          border: '1px solid var(--gray-light)',
          padding: '3px 8px',
        }}>NLP ▸ STUB</div>
      </div>

      {/* Text area */}
      <div style={{ padding: '10px 10px', borderBottom: 'var(--border)', flex: 1 }}>
        <textarea
          value={instruction}
          onChange={(e) => setInstruction(e.target.value)}
          onKeyDown={handleKey}
          placeholder='Type an instruction... e.g. "Pick the red block"'
          disabled={isRunning}
          style={{
            width: '100%',
            minHeight: 50,
            background: 'var(--gray-bg)',
            border: '1.5px solid var(--black)',
            padding: '12px 14px',
            fontFamily: 'var(--font-mono)',
            fontSize: 13,
            color: 'var(--black)',
            resize: 'none',
            outline: 'none',
            transition: 'border-color 0.15s',
            lineHeight: 1.6,
          }}
          onFocus={(e) => (e.target.style.borderColor = 'var(--orange)')}
          onBlur={(e) => (e.target.style.borderColor = 'var(--black)')}
        />
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginTop: 8,
        }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--gray-text)' }}>
            {instruction.length} chars · ENTER to run
          </span>
          <button
            className="btn-primary"
            onClick={handleSubmit}
            disabled={isRunning || !instruction.trim()}
          >
            {isRunning ? (
              <>
                <span style={{ display: 'inline-block', animation: 'spin 1s linear infinite' }}>⟳</span>
                RUNNING...
              </>
            ) : '▶ RUN INSTRUCTION'}
          </button>
        </div>
      </div>

      {/* Example commands */}
      <div style={{ padding: '12px 20px' }}>
        <div className="label" style={{ marginBottom: 8 }}>Quick Examples</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {EXAMPLE_COMMANDS.map((cmd) => (
            <button
              key={cmd}
              className="btn-secondary"
              style={{ fontSize: 11, padding: '5px 12px' }}
              onClick={() => setInstruction(cmd)}
              disabled={isRunning}
            >
              {cmd}
            </button>
          ))}
        </div>
      </div>

      

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}