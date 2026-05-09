// RobotScene.jsx — Three.js 7-DOF arm visualization (Phase 1)
//                  Real PyBullet stream (Phase 2, USE_BACKEND = true)
import React, { useEffect, useRef, useCallback } from 'react';
import * as THREE from 'three';

// ─────────────────────────────────────────────────────────────
// Toggle — must match simulator.js
// false → Three.js arm (no Flask needed)
// true  → Real PyBullet video stream from Flask
// ─────────────────────────────────────────────────────────────
const USE_BACKEND = true;
const STREAM_URL = 'http://localhost:5000/api/stream';

// ─────────────────────────────────────────────────────────────
// Three.js arm builder — Phase 1
// ─────────────────────────────────────────────────────────────
function buildArm(scene) {
  const joints = [];
  const mat = new THREE.MeshPhongMaterial({ color: 0x1a1a1a, shininess: 80 });
  const jointMat = new THREE.MeshPhongMaterial({ color: 0xFF4400, shininess: 120 });
  const endMat = new THREE.MeshPhongMaterial({ color: 0xffffff, shininess: 200 });

  const LINK_LENGTHS = [0.5, 0.55, 0.45, 0.55, 0.4, 0.35, 0.2];
  const LINK_RADII   = [0.07, 0.065, 0.06, 0.055, 0.05, 0.045, 0.04];

  let parentGroup = null;

  // Base plate
  const baseMesh = new THREE.Mesh(
    new THREE.CylinderGeometry(0.18, 0.22, 0.08, 32),
    new THREE.MeshPhongMaterial({ color: 0x111111 })
  );
  baseMesh.position.y = 0;
  scene.add(baseMesh);

  for (let i = 0; i < 7; i++) {
    const group = new THREE.Group();

    // Joint sphere
    const jSphere = new THREE.Mesh(
      new THREE.SphereGeometry(LINK_RADII[i] * 1.3, 16, 16),
      jointMat
    );
    group.add(jSphere);

    // Link cylinder
    const linkLen = LINK_LENGTHS[i];
    const link = new THREE.Mesh(
      new THREE.CylinderGeometry(LINK_RADII[i] * 0.7, LINK_RADII[i], linkLen, 16),
      i === 6 ? endMat : mat
    );
    link.position.y = linkLen / 2;
    group.add(link);

    if (parentGroup) {
      group.position.y = LINK_LENGTHS[i - 1];
      parentGroup.add(group);
    } else {
      group.position.y = 0.08;
      scene.add(group);
    }

    joints.push(group);
    parentGroup = group;
  }

  // Gripper fingers
  const gripperGroup = new THREE.Group();
  gripperGroup.position.y = LINK_LENGTHS[6];
  const fingerMat = new THREE.MeshPhongMaterial({ color: 0x333333 });
  [-0.06, 0.06].forEach((offset) => {
    const finger = new THREE.Mesh(
      new THREE.BoxGeometry(0.025, 0.12, 0.025),
      fingerMat
    );
    finger.position.set(offset, 0.06, 0);
    gripperGroup.add(finger);
  });
  joints[6].add(gripperGroup);

  return joints;
}

// ─────────────────────────────────────────────────────────────
// Shared HUD overlays — used by both modes
// ─────────────────────────────────────────────────────────────
function JointHUD({ jointAngles }) {
  return (
    <div style={{
      position: 'absolute',
      bottom: 12,
      right: 12,
      background: 'rgba(10,10,10,0.85)',
      color: 'white',
      padding: '8px 12px',
      fontFamily: 'var(--font-mono)',
      fontSize: 9,
      letterSpacing: '0.05em',
      minWidth: 140,
    }}>
      <div style={{ color: 'var(--orange)', marginBottom: 4, fontSize: 8, letterSpacing: '0.15em' }}>
        JOINT ANGLES (°)
      </div>
      {(jointAngles || Array(7).fill({ angle: 0 })).map((j, i) => (
        <div key={i} style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
          <span style={{ color: 'var(--gray-mid)' }}>J{i + 1}</span>
          <span>{typeof j?.angle === 'number' ? j.angle.toFixed(1) : '0.0'}</span>
        </div>
      ))}
    </div>
  );
}

function TopLeftHUD({ line1, line2 }) {
  return (
    <div style={{
      position: 'absolute',
      top: 12,
      left: 12,
      display: 'flex',
      flexDirection: 'column',
      gap: 4,
    }}>
      <div style={{
        background: 'rgba(255,255,255,0.92)',
        border: 'var(--border)',
        padding: '4px 10px',
        fontFamily: 'var(--font-mono)',
        fontSize: 10,
        display: 'flex',
        alignItems: 'center',
        gap: 6,
      }}>
        <span style={{ color: 'var(--orange)' }}>■</span>
        <span>{line1}</span>
      </div>
      <div style={{
        background: 'rgba(255,255,255,0.92)',
        border: 'var(--border)',
        padding: '4px 10px',
        fontFamily: 'var(--font-mono)',
        fontSize: 10,
        color: 'var(--gray-text)',
      }}>
        {line2}
      </div>
    </div>
  );
}

function ExecutingBadge() {
  return (
    <div style={{
      position: 'absolute',
      top: 12,
      right: 12,
      background: 'var(--orange)',
      color: 'white',
      padding: '4px 12px',
      fontFamily: 'var(--font-mono)',
      fontSize: 10,
      letterSpacing: '0.1em',
      animation: 'pulse-dot 1s ease-in-out infinite',
    }}>
      ▶ EXECUTING
    </div>
  );
}

function PoseLabel({ pose }) {
  return (
    <div style={{
      position: 'absolute',
      bottom: 12,
      left: 12,
      background: 'var(--white)',
      border: 'var(--border)',
      padding: '4px 12px',
      fontFamily: 'var(--font-display)',
      fontWeight: 700,
      fontSize: 13,
      letterSpacing: '0.12em',
      textTransform: 'uppercase',
    }}>
      POSE: <span style={{ color: 'var(--orange)' }}>{pose.toUpperCase()}</span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Phase 2 — Real PyBullet stream view
// ─────────────────────────────────────────────────────────────
function PyBulletStreamView({ jointAngles, pose, isRunning }) {
  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', background: '#fafaf8' }}>
      <img
        src={STREAM_URL}
        style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
        alt="PyBullet live stream"
      />

      <TopLeftHUD
        line1="PyBullet LIVE — 7-DOF KUKA IIWA"
        line2="REAL STREAM · Full Pipeline"
      />

      <JointHUD jointAngles={jointAngles} />

      {isRunning && <ExecutingBadge />}
      {pose && <PoseLabel pose={pose} />}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Phase 1 — Three.js simulated arm view
// ─────────────────────────────────────────────────────────────
function ThreeJSView({ jointAngles, pose, isRunning }) {
  const mountRef = useRef(null);
  const jointsRef = useRef([]);
  const rendererRef = useRef(null);
  const frameRef = useRef(null);
  const objectMeshesRef = useRef([]);
  const currentAnglesRef = useRef(Array(7).fill(0));
  const targetAnglesRef = useRef(Array(7).fill(0));

  const lerpAngles = useCallback(() => {
    const cur = currentAnglesRef.current;
    const tgt = targetAnglesRef.current;
    for (let i = 0; i < 7; i++) {
      const diff = tgt[i] - cur[i];
      if (Math.abs(diff) > 0.001) cur[i] += diff * 0.04;
    }
  }, []);

  useEffect(() => {
    const mount = mountRef.current;
    const w = mount.clientWidth;
    const h = mount.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xfafaf8);

    // Lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const dir = new THREE.DirectionalLight(0xffffff, 1.2);
    dir.position.set(2, 4, 3);
    dir.castShadow = true;
    scene.add(dir);
    const fill = new THREE.DirectionalLight(0xFF8844, 0.3);
    fill.position.set(-3, 1, -2);
    scene.add(fill);

    // Grid + floor
    const grid = new THREE.GridHelper(3, 20, 0xcccccc, 0xe8e8e0);
    grid.position.y = -0.04;
    scene.add(grid);
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(4, 4),
      new THREE.MeshPhongMaterial({ color: 0xf4f4f2, transparent: true, opacity: 0.8 })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.041;
    scene.add(floor);

    // Arm
    jointsRef.current = buildArm(scene);

    // Scene objects
    const objConfigs = [
      { color: 0xEF4444, x: 0.5,   z: 0.3  },
      { color: 0x22C55E, x: -0.45, z: 0.25 },
      { color: 0x3B82F6, x: 0.1,   z: -0.45 },
      { color: 0xEAB308, x: -0.3,  z: -0.3 },
    ];
    const boxGeo = new THREE.BoxGeometry(0.12, 0.12, 0.12);
    objConfigs.forEach((cfg) => {
      const mesh = new THREE.Mesh(
        boxGeo,
        new THREE.MeshPhongMaterial({ color: cfg.color, shininess: 60 })
      );
      mesh.position.set(cfg.x, 0.02, cfg.z);
      scene.add(mesh);
      objectMeshesRef.current.push(mesh);
    });

    // Camera
    const camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 100);
    camera.position.set(2.2, 2.5, 2.8);
    camera.lookAt(0, 1, 0);

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    let t = 0;
    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);
      t += 0.004;

      // Orbit camera
      camera.position.x = Math.sin(t) * 3.0;
      camera.position.z = Math.cos(t) * 3.0;
      camera.position.y = 2.5 + Math.sin(t * 0.5) * 0.3;
      camera.lookAt(0, 1.2, 0);

      // Lerp joints
      lerpAngles();
      const cur = currentAnglesRef.current;
      const jts = jointsRef.current;
      if (jts[0]) jts[0].rotation.y = THREE.MathUtils.degToRad(cur[0]);
      if (jts[1]) jts[1].rotation.z = THREE.MathUtils.degToRad(cur[1]);
      if (jts[2]) jts[2].rotation.y = THREE.MathUtils.degToRad(cur[2]);
      if (jts[3]) jts[3].rotation.z = THREE.MathUtils.degToRad(cur[3]);
      if (jts[4]) jts[4].rotation.y = THREE.MathUtils.degToRad(cur[4]);
      if (jts[5]) jts[5].rotation.z = THREE.MathUtils.degToRad(cur[5]);
      if (jts[6]) jts[6].rotation.y = THREE.MathUtils.degToRad(cur[6]);

      // Object bob
      objectMeshesRef.current.forEach((m, i) => {
        m.position.y = 0.02 + Math.sin(t * 0.8 + i) * 0.005;
        m.rotation.y = t * 0.3 * (i % 2 === 0 ? 1 : -1);
      });

      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      const w2 = mount.clientWidth;
      const h2 = mount.clientHeight;
      camera.aspect = w2 / h2;
      camera.updateProjectionMatrix();
      renderer.setSize(w2, h2);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(frameRef.current);
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, [lerpAngles]);

  useEffect(() => {
    if (jointAngles && jointAngles.length === 7) {
      targetAnglesRef.current = jointAngles.map((j) => j.angle);
    } else {
      targetAnglesRef.current = Array(7).fill(0);
    }
  }, [jointAngles]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />

      <TopLeftHUD
        line1="PyBullet SIM — 7-DOF KUKA IIWA"
        line2="PHASE 1 · STUB MODE · NO HARDWARE"
      />

      <JointHUD jointAngles={jointAngles} />

      {isRunning && <ExecutingBadge />}
      {pose && <PoseLabel pose={pose} />}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Main export — switches based on USE_BACKEND
// ─────────────────────────────────────────────────────────────
export default function RobotScene({ jointAngles, pose, isRunning, sceneObjects }) {
  if (USE_BACKEND) {
    return (
      <PyBulletStreamView
        jointAngles={jointAngles}
        pose={pose}
        isRunning={isRunning}
      />
    );
  }

  return (
    <ThreeJSView
      jointAngles={jointAngles}
      pose={pose}
      isRunning={isRunning}
    />
  );
}