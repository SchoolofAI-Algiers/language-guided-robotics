// App.jsx — Main layout and application state
import React, { useState, useCallback, useEffect } from 'react';
import Header from './components/Header.jsx';
import RobotScene from './components/RobotScene.jsx';
import InstructionPanel from './components/InstructionPanel.jsx';
import PipelineFlow from './components/PipelineFlow.jsx';
import ModuleStatus from './components/ModuleStatus.jsx';
import ExecutionLog from './components/ExecutionLog.jsx';
import ScenePanel from './components/ScenePanel.jsx';
import { runPipeline, resetScene, fetchScene } from './lib/simulator.js';

const ts = () => new Date().toTimeString().slice(0, 8);

const INITIAL_LOGS = [
  { type: 'system', timestamp: ts(), message: 'SOAI Labs — Language-Guided Robotics v0.1' },
  { type: 'system', timestamp: ts(), message: 'Initialized. PyBullet simulation mode active.' },
  { type: 'system', timestamp: ts(), message: 'NLP ready (all-MiniLM-L6-v2). CV ready. RL trained (Beta PPO).' },
];

// ── helpers ───────────────────────────────────────────────────

function _toCvResult(detected_objects, model = 'ResNet18-4ch Beta') {
  return {
    detected_objects:  detected_objects || [],
    detectedObjects:   detected_objects || [],   // log compat
    n_visible:         (detected_objects || []).length,
    model,
  };
}

// ═════════════════════════════════════════════════════════════
export default function App() {
  const [isRunning,      setIsRunning]      = useState(false);
  const [jointAngles,    setJointAngles]    = useState(null);
  const [pose,           setPose]           = useState('home');
  const [logs,           setLogs]           = useState(INITIAL_LOGS);
  const [pipelineStages, setPipelineStages] = useState({});
  const [cvResult,       setCvResult]       = useState(null);
  const [systemStatus,   setSystemStatus]   = useState({
    nlp: 'ready', cv: 'ready', rl: 'ready', sim: 'ready',
  });

  const addLog = useCallback((type, message) => {
    setLogs(prev => [...prev, { type, timestamp: ts(), message }]);
  }, []);

  // ── Fetch initial scene once on mount ────────────────────────
  useEffect(() => {
    fetchScene().then(data => {
      if (data.detected_objects?.length) {
        setCvResult(_toCvResult(data.detected_objects, data.model));
      }
    });
  }, []);

  // ── Continuously poll scene to show object positions ─────────
  useEffect(() => {
    const pollScene = async () => {
      try {
        const data = await fetchScene();
        // Always update, even if empty or during instruction
        setCvResult(_toCvResult(data.detected_objects || [], data.model));
      } catch (err) {
        // Silently fail - polling is just for UI refresh
      }
    };

    // Poll every 300ms to catch position changes as arm moves
    const interval = setInterval(pollScene, 300);
    return () => clearInterval(interval);
  }, []);

  // ── Stage update handler ─────────────────────────────────────
  const handleStageUpdate = useCallback((update) => {
    const { stage, status, result } = update;

    setPipelineStages(prev => ({ ...prev, [stage]: { status, result } }));
    setSystemStatus(prev => ({
      ...prev,
      [stage]: status === 'running' ? 'active' : status === 'done' ? 'ready' : prev[stage],
    }));

    if (status === 'running') {
      addLog(stage, `[${stage.toUpperCase()}] Processing...`);
    }

    if (status === 'done' && result) {
      if (stage === 'nlp') {
        addLog('nlp', `Command: ${result.commandType?.toUpperCase()} · Target: "${result.target}" · Confidence: ${result.confidence}`);
        addLog('nlp', `Embedding: [${result.embeddingPreview?.join(', ')}] (${result.embeddingDim}-dim) — ${result.latencyMs}ms`);
      }

      if (stage === 'cv') {
        // Map real backend objects to ScenePanel format
        const realObjects = result.detected_objects || result.detectedObjects || [];
        setCvResult(_toCvResult(realObjects, result.model));
        addLog('cv', `Detected ${realObjects.length} objects — ${result.model || 'ResNet18'}`);
        if (realObjects.length > 0) {
          const obj = realObjects[0];
          addLog('cv', `First: ${obj.color} ${obj.shape} @ [${obj.pos?.map(v=>v.toFixed(2)).join(', ')}]`);
        }
      }

      if (stage === 'rl') {
        setJointAngles(result.jointAngles);
        setPose(result.pose);
        addLog('rl', `Policy: ${result.policy} · Pose: ${result.pose?.toUpperCase()} · P(success)=${result.successProbability}`);
        addLog('rl', `Joint deltas: [${result.jointAngles?.map(j => j.angle.toFixed(1)).join(', ')}]°`);
        if (result.success) {
          addLog('success', `✓ Task succeeded — distance ${result.distanceFinal?.toFixed(3)}m`);
        } else {
          addLog('rl', `Final distance: ${result.distanceFinal?.toFixed(3)}m — arm stopped at best position`);
        }
      }
    }

    if (stage === 'action' && status === 'done') {
      addLog('success', '✓ Arm command dispatched. Execution complete.');
    }
  }, [addLog]);

  // ── Instruction submission ───────────────────────────────────
  const handleInstruction = useCallback(async (instruction) => {
    if (isRunning) return;

    setIsRunning(true);
    setPipelineStages({});
    addLog('system', `━━━ New instruction: "${instruction}" ━━━`);

    try {
      await runPipeline(instruction, handleStageUpdate);
    } catch (err) {
      addLog('error', `Pipeline error: ${err.message}`);
    } finally {
      setIsRunning(false);
      setSystemStatus({ nlp: 'ready', cv: 'ready', rl: 'ready', sim: 'ready' });
    }
  }, [isRunning, addLog, handleStageUpdate]);

  // ── New scene / reset ────────────────────────────────────────
  const handleNewScene = useCallback(async () => {
    addLog('system', '━━━ Spawning new scene ━━━');
    const data = await resetScene();
    if (data.detected_objects) {
      setCvResult(_toCvResult(data.detected_objects));
      addLog('system', `New scene: ${data.detected_objects.length} objects spawned`);
    } else {
      addLog('error', `Reset failed: ${data.error || 'unknown error'}`);
    }
  }, [addLog]);

  // ─────────────────────────────────────────────────────────────
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      <Header phase={1} systemStatus={systemStatus} />

      {/* Hero strip */}
      <div style={{
        background: 'var(--black)',
        padding: '10px 24px',
        display: 'flex',
        alignItems: 'center',
        gap: 24,
        borderBottom: 'var(--border)',
      }}>
        <div className="display-xl" style={{ color: 'var(--white)', fontSize: 32, letterSpacing: '0.04em' }}>
          LANGUAGE<span style={{ color: 'var(--orange)' }}>–</span>GUIDED ROBOTICS
        </div>
        <div style={{ flex: 1 }} />
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--gray-mid)' }}>
          8-WEEK SPRINT · NLP + CV + RL
        </div>
        <div style={{
          background: 'var(--orange)',
          color: 'white',
          fontFamily: 'var(--font-display)',
          fontWeight: 700,
          fontSize: 12,
          padding: '6px 16px',
          letterSpacing: '0.1em',
        }}>
          PYBULLET SIM
        </div>
      </div>

      {/* Main grid */}
      <div style={{
        flex: 1,
        display: 'grid',
        gridTemplateColumns: '1fr 380px',
        overflow: 'hidden',
        minHeight: 0,
      }}>

        {/* LEFT: Robot Arm + Pipeline */}
        <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', borderRight: 'var(--border)' }}>

          {/* Top: 3D view + Instruction + Scene */}
          <div style={{
            flex: 1,
            display: 'grid',
            gridTemplateColumns: '1fr 340px',
            borderBottom: 'var(--border)',
            minHeight: 0,
          }}>
            {/* PyBullet stream */}
            <div style={{ borderRight: 'var(--border)', overflow: 'hidden' }}>
              <RobotScene
                jointAngles={jointAngles}
                pose={pose}
                isRunning={isRunning}
              />
            </div>

            {/* Instruction + Scene panel stacked */}
            <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              <div style={{ flex: 1, borderBottom: 'var(--border)', overflow: 'auto' }}>
                <InstructionPanel
                  onSubmit={handleInstruction}
                  isRunning={isRunning}
                />
              </div>
              <div style={{ overflow: 'auto' }}>
                <ScenePanel cvResult={cvResult} />
              </div>
            </div>
          </div>

          {/* Pipeline flow */}
          <div style={{ background: 'var(--white)', borderTop: 'var(--border)', height: 220, flexShrink: 0 }}>
            <PipelineFlow stages={pipelineStages} onNewScene={handleNewScene} />
          </div>
        </div>

        {/* RIGHT: Module status + Log */}
        <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div style={{ flex: '0 0 auto', borderBottom: 'var(--border)', overflow: 'auto', maxHeight: '55%' }}>
            <ModuleStatus />
          </div>
          <div style={{ flex: 1, overflow: 'hidden' }}>
            <ExecutionLog logs={logs} />
          </div>
        </div>
      </div>
    </div>
  );
}