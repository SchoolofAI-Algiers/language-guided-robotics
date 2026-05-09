// ============================================================
// simulator.js — wired to Beta PPO backend
// ============================================================

const USE_BACKEND = true;
const BACKEND_URL = 'http://localhost:5000';

const delay = (ms) => new Promise((r) => setTimeout(r, ms));

// ─────────────────────────────────────────────────────────────
// BROWSER STUBS — fallback / USE_BACKEND=false
// ─────────────────────────────────────────────────────────────

const COMMAND_TYPES = {
  pick:  { pattern: /pick|grab|take|get/i,    pose: 'reach' },
  place: { pattern: /place|put|drop|set/i,    pose: 'place' },
  reach: { pattern: /reach|move|go|extend/i,  pose: 'extend' },
  home:  { pattern: /home|reset|neutral|rest/i, pose: 'home' },
};

async function runNLP_stub(instruction) {
  await delay(300 + Math.random() * 400);
  let commandType = 'pick', confidence = 0.72, target = 'nearest object';
  for (const [type, cfg] of Object.entries(COMMAND_TYPES)) {
    if (cfg.pattern.test(instruction)) {
      commandType = type; confidence = 0.72 + Math.random() * 0.25; break;
    }
  }
  for (const c of ['red','green','blue','yellow']) {
    if (instruction.toLowerCase().includes(c)) { target = `${c} object`; break; }
  }
  return {
    stage: 'nlp', instruction, commandType, target,
    confidence: confidence.toFixed(3),
    embeddingPreview: Array.from({ length: 8 }, () => (Math.random() * 2 - 1).toFixed(4)),
    embeddingDim: 384, model: 'all-MiniLM-L6-v2 (stub)',
    latencyMs: Math.round(300 + Math.random() * 400),
  };
}

async function runCV_stub() {
  await delay(200 + Math.random() * 300);
  const objects = [
    { id: 3, color: 'red',    shape: 'box',      pos: [0.32, -0.10, 0.42] },
    { id: 4, color: 'green',  shape: 'sphere',   pos: [0.52,  0.20, 0.42] },
    { id: 5, color: 'blue',   shape: 'cylinder', pos: [0.27, -0.01, 0.42] },
    { id: 6, color: 'yellow', shape: 'box',      pos: [0.62, -0.01, 0.42] },
  ];
  return {
    stage: 'cv', detected_objects: objects, detectedObjects: objects,
    n_visible: objects.length, featureDim: 512,
    depthEstimate: '0.420', model: 'ResNet18-4ch (stub)',
    latencyMs: Math.round(200 + Math.random() * 300),
  };
}

async function runRL_stub(nlpResult) {
  await delay(150 + Math.random() * 250);
  return {
    stage: 'rl', policy: 'PPO-Beta (stub)',
    jointAngles: Array.from({ length: 7 }, (_, i) => ({ joint: i, angle: +(Math.random()*40-20).toFixed(1) })),
    pose: COMMAND_TYPES[nlpResult.commandType]?.pose || 'reach',
    rewardEstimate: (0.55 + Math.random() * 0.42).toFixed(3),
    successProbability: (0.6 + Math.random() * 0.37).toFixed(3),
    episodeStep: Math.floor(Math.random() * 60) + 5,
    latencyMs: Math.round(150 + Math.random() * 250),
  };
}

// ─────────────────────────────────────────────────────────────
// BACKEND PIPELINE — Beta PPO real inference
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
    ['nlp', 'cv', 'rl'].forEach(s => onStageUpdate({ stage: s, status: 'error' }));
    throw new Error(`Backend unreachable — is Flask running? (${err.message})`);
  }

  // ── NLP ─────────────────────────────────────────────────────
  const nlpResult = {
    stage: 'nlp',
    instruction,
    commandType:     data.nlp?.command_type  || _guessCommandType(instruction),
    target:          data.nlp?.target        || data.matched_instruction || instruction,
    confidence:      String(data.nlp?.confidence ?? '0.95'),
    embeddingPreview: ['...384-dim embedding...'],
    embeddingDim: 384,
    model: 'all-MiniLM-L6-v2',
    latencyMs: 0,
  };
  onStageUpdate({ stage: 'nlp', status: 'done', result: nlpResult });
  await delay(60);

  // ── CV — use REAL detected objects from backend ──────────────
  const realObjects = data.cv?.detected_objects || [];
  const cvResult = {
    stage: 'cv',
    detected_objects: realObjects,   // ScenePanel reads this
    detectedObjects:  realObjects,   // log reads this
    n_visible:        data.cv?.n_visible ?? realObjects.length,
    targetObject:     realObjects.find(o => o.color === (data.nlp?.target?.split(' ')[0])) || realObjects[0] || null,
    featureVector:    (data.ee_trajectory?.[0] || [0, 0, 0]).map(v => v.toFixed(3)),
    featureDim:       512,
    depthEstimate:    (realObjects[0]?.pos?.[2] ?? 0.42).toFixed(3),
    model:            data.cv?.model || 'ResNet18-4ch Beta',
    latencyMs: 0,
  };
  onStageUpdate({ stage: 'cv', status: 'running' });
  await delay(60);
  onStageUpdate({ stage: 'cv', status: 'done', result: cvResult });

  // ── RL — real PPO output ─────────────────────────────────────
  const lastEE  = data.ee_trajectory?.at(-1) || [0, 0, 0];
  const firstEE = data.ee_trajectory?.[0]    || [0, 0, 0];
  const rlResult = {
    stage: 'rl',
    policy: 'PPO — Beta Early Spatial Fusion',
    jointAngles: Array.from({ length: 7 }, (_, i) => ({
      joint: i,
      angle: parseFloat(((lastEE[i % 3] - firstEE[i % 3]) * 30).toFixed(1)),
    })),
    pose:               data.success ? 'reach' : 'home',
    rewardEstimate:     data.success ? '1.000' : (-(data.distance_final || 0)).toFixed(3),
    successProbability: data.success ? '1.000' : Math.max(0, 1 - (data.distance_final || 1)).toFixed(3),
    episodeStep:  data.steps || 0,
    latencyMs:    0,
    success:      data.success,
    distanceFinal: data.distance_final,
    strategy:     data.strategy,
  };
  onStageUpdate({ stage: 'rl', status: 'running' });
  await delay(60);
  onStageUpdate({ stage: 'rl', status: 'done', result: rlResult });

  onStageUpdate({ stage: 'action', status: 'running' });
  await delay(100);
  onStageUpdate({ stage: 'action', status: 'done' });

  return { nlp: nlpResult, cv: cvResult, rl: rlResult };
}

function _guessCommandType(instruction) {
  const lower = instruction.toLowerCase();
  if (/pick|grab|take|get/.test(lower))   return 'pick';
  if (/place|put|drop|set/.test(lower))   return 'place';
  if (/reach|move|go|extend/.test(lower)) return 'reach';
  if (/home|reset|neutral/.test(lower))   return 'home';
  return 'pick';
}

// ─────────────────────────────────────────────────────────────
// PUBLIC API
// ─────────────────────────────────────────────────────────────

export async function runNLP(instruction) { return runNLP_stub(instruction); }
export async function runCV(nlpResult)    { return runCV_stub(nlpResult); }
export async function runRL(nlpResult, cvResult) { return runRL_stub(nlpResult, cvResult); }

export async function resetScene() {
  try {
    const res = await fetch(`${BACKEND_URL}/api/reset`, { method: 'POST' });
    return await res.json();   // { status, detected_objects }
  } catch (e) {
    console.error('[resetScene] failed:', e);
    return { status: 'error', detected_objects: [] };
  }
}

export async function fetchScene() {
  try {
    const res = await fetch(`${BACKEND_URL}/api/scene`);
    return await res.json();   // { status, detected_objects, n_visible }
  } catch (e) {
    return { status: 'error', detected_objects: [] };
  }
}

export async function runPipeline(instruction, onStageUpdate) {
  if (USE_BACKEND) return runPipeline_backend(instruction, onStageUpdate);

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
  'Pick the red box',
  'Get the green sphere',
  'Grab the blue cylinder',
  'Pick the yellow box',
];