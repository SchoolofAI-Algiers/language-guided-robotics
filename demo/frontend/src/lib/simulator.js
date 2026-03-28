// ============================================================
// simulator.js — Phase 1 stub pipeline
// ─────────────────────────────────────────────────────────────
// USE_BACKEND flag controls where the pipeline runs:
//
//   false → everything runs in the browser (Phase 1, no Flask needed)
//   true  → calls Flask at BACKEND_URL (Phase 2/3, needs backend running)
//
// When flipping to true, start Flask first:
//   cd demo/backend && python app.py
// ============================================================

const USE_BACKEND = true;
const BACKEND_URL = 'http://localhost:5000';

// ─────────────────────────────────────────────────────────────
// Stub data — used when USE_BACKEND is false
// ─────────────────────────────────────────────────────────────

const COMMAND_TYPES = {
  pick: {
    pattern: /pick|grab|take|get/i,
    targets: ['red block', 'green cube', 'blue cylinder', 'yellow sphere'],
    actions: (target) => [
      { joint: 0, angle: Math.random() * 45 - 22.5 },
      { joint: 1, angle: Math.random() * 60 - 10 },
      { joint: 2, angle: Math.random() * 40 },
      { joint: 3, angle: Math.random() * -90 - 20 },
      { joint: 4, angle: Math.random() * 60 - 30 },
      { joint: 5, angle: Math.random() * 30 - 15 },
      { joint: 6, angle: 0 },
    ],
    pose: 'reach',
  },
  place: {
    pattern: /place|put|drop|set/i,
    targets: ['left zone', 'right zone', 'center platform', 'tray'],
    actions: () => [
      { joint: 0, angle: 30 + Math.random() * 30 },
      { joint: 1, angle: -20 + Math.random() * 20 },
      { joint: 2, angle: 60 + Math.random() * 20 },
      { joint: 3, angle: -80 },
      { joint: 4, angle: 45 },
      { joint: 5, angle: 0 },
      { joint: 6, angle: 0 },
    ],
    pose: 'place',
  },
  reach: {
    pattern: /reach|move|go|extend/i,
    targets: ['forward', 'left', 'right', 'up'],
    actions: () => [
      { joint: 0, angle: Math.random() * 90 - 45 },
      { joint: 1, angle: 30 + Math.random() * 30 },
      { joint: 2, angle: Math.random() * 40 },
      { joint: 3, angle: -60 },
      { joint: 4, angle: Math.random() * 30 },
      { joint: 5, angle: 0 },
      { joint: 6, angle: 0 },
    ],
    pose: 'extend',
  },
  home: {
    pattern: /home|reset|neutral|rest/i,
    targets: ['home position'],
    actions: () => Array.from({ length: 7 }, (_, i) => ({ joint: i, angle: 0 })),
    pose: 'home',
  },
};

const delay = (ms) => new Promise((r) => setTimeout(r, ms));

// ─────────────────────────────────────────────────────────────
// BROWSER STUBS — run locally, no Flask required
// ─────────────────────────────────────────────────────────────

async function runNLP_stub(instruction) {
  await delay(300 + Math.random() * 400);

  let commandType = null;
  let confidence = 0.0;
  let target = null;

  for (const [type, cfg] of Object.entries(COMMAND_TYPES)) {
    if (cfg.pattern.test(instruction)) {
      commandType = type;
      confidence = 0.72 + Math.random() * 0.25;
      for (const t of cfg.targets) {
        if (instruction.toLowerCase().includes(t.split(' ')[0])) {
          target = t;
          break;
        }
      }
      target = target || cfg.targets[Math.floor(Math.random() * cfg.targets.length)];
      break;
    }
  }

  if (!commandType) {
    commandType = 'pick';
    confidence = 0.41 + Math.random() * 0.2;
    target = 'nearest object';
  }

  const embedding = Array.from({ length: 8 }, () => (Math.random() * 2 - 1).toFixed(4));

  return {
    stage: 'nlp',
    instruction,
    commandType,
    target,
    confidence: confidence.toFixed(3),
    embeddingPreview: embedding,
    embeddingDim: 384,
    model: 'all-MiniLM-L6-v2 (stub)',
    latencyMs: Math.round(300 + Math.random() * 400),
  };
}

async function runCV_stub(nlpResult) {
  await delay(200 + Math.random() * 300);

  const objects = [
    { id: 'obj_0', label: 'red block',      color: '#EF4444', x: 0.32,  y: 0.14, z: 0.21,  conf: 0.91 },
    { id: 'obj_1', label: 'green cube',     color: '#22C55E', x: -0.18, y: 0.14, z: 0.28,  conf: 0.87 },
    { id: 'obj_2', label: 'blue cylinder',  color: '#3B82F6', x: 0.05,  y: 0.14, z: -0.22, conf: 0.83 },
    { id: 'obj_3', label: 'yellow sphere',  color: '#EAB308', x: -0.28, y: 0.14, z: 0.10,  conf: 0.78 },
  ];

  const targetObj = objects.find(o =>
    nlpResult.target && o.label.includes(nlpResult.target.split(' ')[0])
  ) || objects[Math.floor(Math.random() * objects.length)];

  const featureVec = Array.from({ length: 6 }, () => (Math.random()).toFixed(3));

  return {
    stage: 'cv',
    detectedObjects: objects,
    targetObject: targetObj,
    featureVector: featureVec,
    featureDim: 512,
    depthEstimate: (targetObj.z + 0.4).toFixed(3),
    model: 'CNN-ResNet18 (stub)',
    latencyMs: Math.round(200 + Math.random() * 300),
  };
}

async function runRL_stub(nlpResult, cvResult) {
  await delay(150 + Math.random() * 250);

  const cfg = COMMAND_TYPES[nlpResult.commandType] || COMMAND_TYPES.pick;
  const jointAngles = cfg.actions(cvResult.targetObject?.label || '');
  const rewardEstimate = 0.55 + Math.random() * 0.42;
  const successProb = 0.6 + Math.random() * 0.37;

  return {
    stage: 'rl',
    policy: 'PPO-v1 (untrained stub)',
    jointAngles,
    pose: cfg.pose,
    rewardEstimate: rewardEstimate.toFixed(3),
    successProbability: successProb.toFixed(3),
    episodeStep: Math.floor(Math.random() * 60) + 5,
    latencyMs: Math.round(150 + Math.random() * 250),
  };
}

// ─────────────────────────────────────────────────────────────
// BACKEND PIPELINE — calls Flask when USE_BACKEND is true
// Flask returns the full result in one POST, then we replay
// the stage updates so the UI animates the same way.
// ─────────────────────────────────────────────────────────────

async function runPipeline_backend(instruction, onStageUpdate) {
  onStageUpdate({ stage: 'nlp', status: 'running' });

  let data;
  try {
    const res = await fetch(`${BACKEND_URL}/api/instruction`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ instruction }),
    });
    if (!res.ok) throw new Error(`Flask responded with ${res.status}`);
    data = await res.json();
  } catch (err) {
    onStageUpdate({ stage: 'nlp', status: 'error' });
    onStageUpdate({ stage: 'cv',  status: 'error' });
    onStageUpdate({ stage: 'rl',  status: 'error' });
    throw new Error(`Backend unreachable — is Flask running? (${err.message})`);
  }

  const nlp = data.nlp || await runNLP_stub(instruction);
  onStageUpdate({ stage: 'nlp', status: 'done', result: { ...nlp, model: nlp.model + ' ← Flask' } });
  await delay(80);

  const cv = data.cv || await runCV_stub(nlp);
  onStageUpdate({ stage: 'cv', status: 'running' });
  await delay(80);
  onStageUpdate({ stage: 'cv', status: 'done', result: cv });

  // ── use real joint angles from Flask if available ──────────
  const rl = data.rl || {
    ...await runRL_stub(nlp, cv),
    jointAngles: data.joint_angles || Array(7).fill({ angle: 0 }),  // ← real values
  };
  onStageUpdate({ stage: 'rl', status: 'running' });
  await delay(80);
  onStageUpdate({ stage: 'rl', status: 'done', result: rl });

  onStageUpdate({ stage: 'action', status: 'running' });
  await delay(100);
  onStageUpdate({ stage: 'action', status: 'done' });

  return { nlp, cv, rl };
}

// ─────────────────────────────────────────────────────────────
// PUBLIC API — the rest of the app only calls these
// ─────────────────────────────────────────────────────────────

export async function runNLP(instruction) {
  return runNLP_stub(instruction);
}

export async function runCV(nlpResult) {
  return runCV_stub(nlpResult);
}

export async function runRL(nlpResult, cvResult) {
  return runRL_stub(nlpResult, cvResult);
}

export async function runPipeline(instruction, onStageUpdate) {
  if (USE_BACKEND) {
    return runPipeline_backend(instruction, onStageUpdate);
  }

  // ── Browser stub pipeline (USE_BACKEND = false) ────────────
  onStageUpdate({ stage: 'nlp', status: 'running' });
  const nlp = await runNLP_stub(instruction);
  onStageUpdate({ stage: 'nlp', status: 'done', result: nlp });

  onStageUpdate({ stage: 'cv', status: 'running' });
  const cv = await runCV_stub(nlp);
  onStageUpdate({ stage: 'cv', status: 'done', result: cv });

  onStageUpdate({ stage: 'rl', status: 'running' });
  const rl = await runRL_stub(nlp, cv);
  onStageUpdate({ stage: 'rl', status: 'done', result: rl });

  onStageUpdate({ stage: 'action', status: 'running' });
  await delay(100);
  onStageUpdate({ stage: 'action', status: 'done' });

  return { nlp, cv, rl };
}

export const EXAMPLE_COMMANDS = [
  'Pick the red block',
  'Grab the green cube and place it left',
  'Move to home position',
  'Reach forward and extend arm',
  'Pick the blue cylinder',
  'Reset to neutral pose',
];