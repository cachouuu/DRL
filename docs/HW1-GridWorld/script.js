/**
 * docs/HW1-GridWorld/script.js
 * ------------------------------------------------
 * Static GitHub Pages demo — GridWorld Q-Learning (Stochastic MDP)
 *
 * HW 變化 applied:
 *  1. Agent stochastic policy p(a|s)
 *       → Boltzmann (softmax) policy: p(a|s) ∝ exp(Q(s,a)/τ)
 *  2. Env random variable reward p(r|s,a)
 *       → Gaussian noise N(0,σ) added to every reward
 *  3. Env random variable next state p(s'|s,a)
 *       → With probability transNoise the action slips to a random one
 *
 *  Additional interactive controls:
 *  • Transition Noise slider — controls slip probability
 *  • Reward Noise σ slider   — controls reward std-dev
 */

/* ═══════════════════════════════════════════════════════════════
   CONFIG
══════════════════════════════════════════════════════════════ */
const ALPHA       = 0.1;    // learning rate α
const GAMMA_QL    = 0.9;    // discount factor γ
const TAU_START   = 1.0;    // initial Boltzmann temperature τ  [HW 變化 1]
const TAU_MIN     = 0.1;    // floor τ
const TAU_DECAY   = 0.995;  // per-episode decay
const MAX_EPISODES = 300;
const MAX_STEPS    = 300;

const SPEED_MAP = { slow: 350, normal: 80, fast: 12, turbo: 0 };

const ARROW_KEYS  = ['↑', '↓', '←', '→'];
const DELTAS      = { '↑':[-1,0], '↓':[1,0], '←':[0,-1], '→':[0,1] };

/* Box-Muller Gaussian N(0,1) */
function randGaussian() {
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/* ═══════════════════════════════════════════════════════════════
   GRID SETUP STATE
══════════════════════════════════════════════════════════════ */
const $ = id => document.getElementById(id);

let n            = 5;
let maxObstacles = 3;
let phase        = 'start';
let startPos     = null;   // "r,c"
let endPos       = null;
let blockPos     = [];     // [[r,c],...]

/* ═══════════════════════════════════════════════════════════════
   Q-LEARNING STATE
══════════════════════════════════════════════════════════════ */
let qTable       = {};     // { "r,c": { ↑:0, ↓:0, ←:0, →:0 } }
let tau          = TAU_START;   // current Boltzmann temperature τ  [HW 變化 1]
let episodeNum   = 0;
let successCount = 0;
let animRunning  = false;
let paused       = false;
let animTimer    = null;
let agentR       = 0, agentC = 0;
let stepInEp     = 0;
let epReward     = 0;
let lastOutcome  = '';

/* ═══════════════════════════════════════════════════════════════
   GRID BUILDER
══════════════════════════════════════════════════════════════ */
function buildGrid() {
  stopAnimation();
  n            = parseInt($('grid-size').value, 10);
  maxObstacles = n - 2;
  phase        = 'start';
  startPos     = null;
  endPos       = null;
  blockPos     = [];
  qTable       = {};
  tau          = TAU_START;   // reset temperature
  episodeNum   = 0;
  successCount = 0;

  const container = $('grid-container');
  container.innerHTML = '';

  for (let r = 0; r < n; r++) {
    const row = document.createElement('div');
    row.className = 'grid-row';
    for (let c = 0; c < n; c++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.id        = `cell-${r}-${c}`;
      cell.innerHTML = `<span class="cell-coord">${r},${c}</span>`;
      cell.addEventListener('click', () => cellClicked(r, c));
      row.appendChild(cell);
    }
    container.appendChild(row);
  }

  $('info-panel').style.display = 'none';
  hideStats();
  updateStatus();
  updateLegend();
  setButtonState('idle');
}

/* ═══════════════════════════════════════════════════════════════
   CELL CLICK HANDLER
══════════════════════════════════════════════════════════════ */
function cellClicked(row, col) {
  if (animRunning) return;          // lock grid while training
  const key = `${row},${col}`;

  if (phase === 'start') {
    markCell(row, col, 'start');
    startPos = key; phase = 'goal';

  } else if (phase === 'goal') {
    if (key === startPos) { alert('Goal must differ from Start.'); return; }
    markCell(row, col, 'goal');
    endPos = key; phase = 'obstacle';

  } else if (phase === 'obstacle') {
    if (key === startPos || key === endPos) {
      alert('Obstacle cannot overlap Start or Goal.'); return;
    }
    const el = $(`cell-${row}-${col}`);
    if (el.classList.contains('obstacle')) {
      el.classList.remove('obstacle');
      blockPos = blockPos.filter(([r,c]) => !(r===row && c===col));
    } else {
      if (blockPos.length >= maxObstacles) {
        alert(`Max ${maxObstacles} obstacle(s).`); return;
      }
      markCell(row, col, 'obstacle');
      blockPos.push([row, col]);
      if (blockPos.length === maxObstacles) phase = 'done';
    }

  } else {
    const el = $(`cell-${row}-${col}`);
    if (el.classList.contains('obstacle')) {
      el.classList.remove('obstacle');
      blockPos = blockPos.filter(([r,c]) => !(r===row && c===col));
      phase = 'obstacle';
    }
  }

  clearOverlays();
  updateStatus();
}

/* ═══════════════════════════════════════════════════════════════
   MARK CELL
══════════════════════════════════════════════════════════════ */
function markCell(row, col, type) {
  const el = $(`cell-${row}-${col}`);
  el.classList.remove('start','goal','obstacle','agent');
  if (type) el.classList.add(type);
  el.querySelector('.cell-arrow') && el.querySelector('.cell-arrow').remove();
  el.querySelector('.cell-value') && el.querySelector('.cell-value').remove();
}

/* ═══════════════════════════════════════════════════════════════
   STATUS / LEGEND
══════════════════════════════════════════════════════════════ */
function updateStatus() {
  const bar = $('status-bar');
  const rem = maxObstacles - blockPos.length;
  if      (phase==='start')    bar.innerHTML = 'Click a cell → set <strong>Start</strong> (green).';
  else if (phase==='goal')     bar.innerHTML = 'Click a cell → set <strong>Goal</strong> (red).';
  else if (phase==='obstacle') bar.innerHTML = `Click <strong>${rem}</strong> more cell(s) → place Obstacles.`;
  else                         bar.innerHTML = '✅ Grid ready! Press <strong>▶ Run Q-Learning</strong> to train.';
}

function updateLegend() {
  const el = $('legend-obstacle-count');
  if (el) el.textContent = n - 2;
}

/* ═══════════════════════════════════════════════════════════════
   Q-TABLE HELPERS
══════════════════════════════════════════════════════════════ */
function initState(key) {
  if (!qTable[key]) qTable[key] = { '↑':0, '↓':0, '←':0, '→':0 };
}

function getQ(r, c, a) { const k=`${r},${c}`; initState(k); return qTable[k][a]; }
function setQ(r, c, a, v){ const k=`${r},${c}`; initState(k); qTable[k][a]=v; }
function maxQ(r, c)  { initState(`${r},${c}`); return Math.max(...ARROW_KEYS.map(a=>getQ(r,c,a))); }

/**
 * HW 變化 1 – Boltzmann (softmax) stochastic policy p(a|s)
 *
 * p(a|s) = exp(Q(s,a) / τ) / Σ_a' exp(Q(s,a') / τ)
 *
 * High τ → near-uniform (exploration)
 * Low  τ → near-greedy  (exploitation)
 */
function chooseAction(r, c) {
  initState(`${r},${c}`);
  const qVals   = ARROW_KEYS.map(a => getQ(r, c, a));
  const qMax    = Math.max(...qVals);
  const expQ    = qVals.map(q => Math.exp((q - qMax) / Math.max(tau, 1e-8)));
  const sumExp  = expQ.reduce((s, v) => s + v, 0);
  const probs   = expQ.map(e => e / sumExp);

  // sample from the distribution
  let rnd = Math.random(), cumulative = 0;
  for (let i = 0; i < ARROW_KEYS.length; i++) {
    cumulative += probs[i];
    if (rnd <= cumulative) return ARROW_KEYS[i];
  }
  return ARROW_KEYS[ARROW_KEYS.length - 1];
}

/** Deterministic greedy action for displaying the final policy arrow */
function bestAction(r, c) {
  initState(`${r},${c}`);
  const vals = ARROW_KEYS.map(a=>getQ(r,c,a));
  const mx   = Math.max(...vals);
  const ties = ARROW_KEYS.filter((_,i)=>vals[i]===mx);
  return ties[Math.floor(Math.random()*ties.length)];
}

/* ═══════════════════════════════════════════════════════════════
   ENV STEP  (stochastic transition + stochastic reward)
══════════════════════════════════════════════════════════════ */
function envStep(r, c, action) {
  // HW 變化 3: stochastic transition p(s'|s,a) — slip probability
  const transNoise = parseFloat($('trans-noise').value) || 0;
  if (Math.random() < transNoise) {
    action = ARROW_KEYS[Math.floor(Math.random() * 4)];
  }

  const [dr,dc] = DELTAS[action];
  const nr = Math.max(0, Math.min(n-1, r+dr));
  const nc = Math.max(0, Math.min(n-1, c+dc));
  const key = `${nr},${nc}`;
  const blockSet = new Set(blockPos.map(([a,b])=>`${a},${b}`));

  // Base deterministic reward
  let baseReward = 0, done = false;
  if (key === endPos)          { baseReward =  1; done = true; }
  else if (blockSet.has(key))  { baseReward = -1; done = true; }

  // HW 變化 2: stochastic reward p(r|s,a) — Gaussian noise
  const rewardSigma = parseFloat($('reward-noise').value) || 0;
  const reward = baseReward + randGaussian() * rewardSigma;

  return { nr, nc, reward, done };
}

/* ═══════════════════════════════════════════════════════════════
   ANIMATION LOOP
══════════════════════════════════════════════════════════════ */
let pendingAction = null;
let pendingState  = null;

function startAnimation() {
  if (phase !== 'done') {
    alert('Please finish setting up the grid first.'); return;
  }
  stopAnimation();
  qTable = {}; tau = TAU_START; episodeNum = 0;
  successCount = 0; lastOutcome = '';
  clearOverlays();
  showStats();
  animRunning = true; paused = false;
  setButtonState('running');
  beginEpisode();
}

function beginEpisode() {
  if (!animRunning) return;
  if (episodeNum >= MAX_EPISODES) { finishTraining(); return; }
  episodeNum++;
  const [sr,sc] = startPos.split(',').map(Number);
  agentR = sr; agentC = sc;
  stepInEp = 0; epReward = 0;
  renderAgent();
  updateStatsPanel();
  scheduleStep();
}

function scheduleStep() {
  const delay = SPEED_MAP[$('speed').value] || 80;
  if (delay === 0) {
    // turbo: run whole episode instantly, then animate next
    runEpisodeFast();
  } else {
    animTimer = setTimeout(doStep, delay);
  }
}

/* one animated step */
function doStep() {
  if (!animRunning || paused) return;

  const action  = chooseAction(agentR, agentC);
  const { nr, nc, reward, done } = envStep(agentR, agentC, action);

  // Q-learning update
  const oldQ    = getQ(agentR, agentC, action);
  const targetQ = done ? reward : reward + GAMMA_QL * maxQ(nr, nc);
  setQ(agentR, agentC, action, oldQ + ALPHA*(targetQ - oldQ));

  // Move agent
  clearAgentCell(agentR, agentC);
  agentR = nr; agentC = nc;
  epReward += reward;
  stepInEp++;

  renderAgent();
  refreshArrows();
  refreshHeatmap();

  if (done || stepInEp >= MAX_STEPS) {
    lastOutcome = done && reward > 0.5 ? 'GOAL 🏆' : done ? 'OBSTACLE 💥' : 'TIMEOUT ⏱';
    if (done && reward > 0.5) successCount++;
    tau = Math.max(TAU_MIN, tau * TAU_DECAY);   // decay temperature
    updateStatsPanel();
    clearAgentCell(agentR, agentC);
    const delay = SPEED_MAP[$('speed').value] || 80;
    animTimer = setTimeout(beginEpisode, Math.max(delay, 120));
  } else {
    updateStatsPanel();
    scheduleStep();
  }
}

/* turbo: run one full episode without rendering mid-steps */
function runEpisodeFast() {
  if (!animRunning) return;
  if (episodeNum >= MAX_EPISODES) { finishTraining(); return; }
  episodeNum++;
  const [sr,sc] = startPos.split(',').map(Number);
  let r=sr, c=sc, steps=0, outcome='TIMEOUT ⏱', rew=0;

  while (steps < MAX_STEPS) {
    const action = chooseAction(r, c);
    const { nr, nc, reward, done } = envStep(r, c, action);
    const oldQ   = getQ(r, c, action);
    const tgt    = done ? reward : reward + GAMMA_QL * maxQ(nr, nc);
    setQ(r, c, action, oldQ + ALPHA*(tgt - oldQ));
    r=nr; c=nc; rew+=reward; steps++;
    if (done) {
      outcome = reward > 0.5 ? 'GOAL 🏆' : 'OBSTACLE 💥';
      if (reward > 0.5) successCount++;
      break;
    }
  }
  lastOutcome = outcome;
  tau = Math.max(TAU_MIN, tau * TAU_DECAY);   // decay temperature
  epReward = rew; stepInEp = steps;
  agentR=r; agentC=c;

  // batch several episodes before rendering
  const BATCH = 10;
  if (episodeNum % BATCH === 0 || episodeNum >= MAX_EPISODES) {
    refreshArrows();
    refreshHeatmap();
    renderAgent();
    updateStatsPanel();
    animTimer = setTimeout(() => {
      clearAgentCell(agentR, agentC);
      if ($('speed').value === 'turbo') runEpisodeFast();
      else { beginEpisode(); }
    }, 40);
  } else {
    runEpisodeFast();
  }
}

function finishTraining() {
  animRunning = false;
  clearAgentCell(agentR, agentC);
  refreshArrows();
  refreshHeatmap();
  setButtonState('done');
  $('status-bar').innerHTML =
    `✅ Training complete! ${successCount}/${MAX_EPISODES} episodes reached the Goal.`;

  const panel = $('info-panel');
  panel.style.display = 'block';
  const rate = ((successCount/MAX_EPISODES)*100).toFixed(1);
  panel.innerHTML = `
    <h2>🏆 Q-Learning Complete (Stochastic MDP)</h2>
    <ul>
      <li>Episodes: <strong>${MAX_EPISODES}</strong></li>
      <li>Success rate: <strong>${rate}%</strong></li>
      <li>Final τ: <strong>${tau.toFixed(3)}</strong></li>
      <li>Policy: <strong>Boltzmann softmax</strong> p(a|s) ∝ exp(Q/τ)</li>
      <li>Transition noise: <strong>${$('trans-noise').value}</strong> (slip probability)</li>
      <li>Reward noise σ: <strong>${$('reward-noise').value}</strong></li>
      <li>Arrows show the <em>greedy policy</em> π*(s)=argmax Q(s,a).</li>
      <li>Cell brightness reflects learned state value (max-Q).</li>
    </ul>`;
}

function stopAnimation() {
  animRunning = false; paused = false;
  if (animTimer) { clearTimeout(animTimer); animTimer = null; }
}

function pauseResume() {
  if (!animRunning) return;
  paused = !paused;
  $('btn-pause').textContent = paused ? '▶ Resume' : '⏸ Pause';
  if (!paused) scheduleStep();
}

function resetAll() {
  stopAnimation();
  buildGrid();
}

/* ═══════════════════════════════════════════════════════════════
   RENDERING HELPERS
══════════════════════════════════════════════════════════════ */
function renderAgent() {
  const el = $(`cell-${agentR}-${agentC}`);
  if (!el) return;
  if (!el.classList.contains('start') && !el.classList.contains('goal') &&
      !el.classList.contains('obstacle')) {
    el.classList.add('agent');
  }
}

function clearAgentCell(r, c) {
  const el = $(`cell-${r}-${c}`);
  if (el) el.classList.remove('agent');
}

/* Refresh policy arrows for all free cells based on current Q-table */
function refreshArrows() {
  const goalKey = endPos;
  const blockSet = new Set(blockPos.map(([a,b])=>`${a},${b}`));
  for (let r=0; r<n; r++) {
    for (let c=0; c<n; c++) {
      const key=`${r},${c}`;
      const el=$(`cell-${r}-${c}`);
      if (!el) continue;
      let a = el.querySelector('.cell-arrow');
      if (!a) { a=document.createElement('span'); a.className='cell-arrow'; el.appendChild(a); }
      if (key===goalKey)       { a.textContent='🏁'; }
      else if(blockSet.has(key)){ a.textContent=''; }
      else { a.textContent = bestAction(r,c); }
    }
  }
}

/* Subtle heatmap tint based on max-Q value */
function refreshHeatmap() {
  const blockSet = new Set(blockPos.map(([a,b])=>`${a},${b}`));
  // find max Q across all cells for normalisation
  let maxV=-Infinity, minV=Infinity;
  for(let r=0;r<n;r++) for(let c=0;c<n;c++){
    const k=`${r},${c}`;
    if(blockSet.has(k)||k===endPos) continue;
    const v=maxQ(r,c);
    if(v>maxV) maxV=v; if(v<minV) minV=v;
  }
  const range = maxV-minV || 1;
  for(let r=0;r<n;r++) for(let c=0;c<n;c++){
    const k=`${r},${c}`;
    const el=$(`cell-${r}-${c}`);
    if(!el||blockSet.has(k)||k===startPos||k===endPos) continue;
    const v=maxQ(r,c);
    const t=((v-minV)/range);  // 0..1
    // map to a subtle teal tint
    const alpha = 0.06 + t*0.28;
    el.style.background=`rgba(96,165,250,${alpha.toFixed(3)})`;
    // small value label
    let val=el.querySelector('.cell-value');
    if(!val){val=document.createElement('span');val.className='cell-value';el.appendChild(val);}
    val.textContent=v.toFixed(2);
  }
}

/* ═══════════════════════════════════════════════════════════════
   STATS PANEL
══════════════════════════════════════════════════════════════ */
function showStats() { $('stats-panel').style.display='flex'; }
function hideStats() { $('stats-panel').style.display='none'; }

function updateStatsPanel() {
  const rate = episodeNum ? ((successCount/episodeNum)*100).toFixed(1) : '0.0';
  $('stat-episode').textContent   = `${episodeNum} / ${MAX_EPISODES}`;
  $('stat-step').textContent      = stepInEp;
  $('stat-tau').textContent       = tau.toFixed(3);   // τ replaces ε
  $('stat-success').textContent   = `${rate}%`;
  $('stat-outcome').textContent   = lastOutcome;
  const pct = Math.min((episodeNum/MAX_EPISODES)*100,100);
  $('progress-bar').style.width   = pct+'%';
}

/* ═══════════════════════════════════════════════════════════════
   RANDOM POLICY (baseline)
══════════════════════════════════════════════════════════════ */
function showRandomPolicy() {
  if (phase !== 'done') {
    alert('Please finish configuring the grid first.'); return;
  }
  clearOverlays();
  const blockSet  = new Set(blockPos.map(([r,c])=>`${r},${c}`));
  for (let r=0;r<n;r++) for(let c=0;c<n;c++){
    const key=`${r},${c}`, el=$(`cell-${r}-${c}`);
    let a=el.querySelector('.cell-arrow');
    if(!a){a=document.createElement('span');a.className='cell-arrow';el.appendChild(a);}
    if(key===endPos)       { a.textContent='🏁'; continue; }
    if(blockSet.has(key))  { a.textContent='';  continue; }
    a.textContent=ARROW_KEYS[Math.floor(Math.random()*4)];
  }
}

/* ═══════════════════════════════════════════════════════════════
   BUTTON STATE MANAGEMENT
══════════════════════════════════════════════════════════════ */
function setButtonState(state) {
  const btnRun   = $('btn-run');
  const btnPause = $('btn-pause');
  if (state==='idle') {
    btnRun.disabled=false; btnRun.textContent='▶ Run Q-Learning';
    btnPause.disabled=true; btnPause.textContent='⏸ Pause';
  } else if(state==='running') {
    btnRun.disabled=true;
    btnPause.disabled=false; btnPause.textContent='⏸ Pause';
  } else if(state==='done') {
    btnRun.disabled=false; btnRun.textContent='↺ Retrain';
    btnPause.disabled=true;
  }
}

/* ═══════════════════════════════════════════════════════════════
   CLEAR OVERLAYS
══════════════════════════════════════════════════════════════ */
function clearOverlays() {
  document.querySelectorAll('.cell-arrow,.cell-value').forEach(e=>e.remove());
  document.querySelectorAll('.cell').forEach(el=>{
    el.classList.remove('agent');
    el.style.background='';
  });
  $('info-panel').style.display='none';
}

/* ═══════════════════════════════════════════════════════════════
   INIT
══════════════════════════════════════════════════════════════ */
window.addEventListener('DOMContentLoaded', () => {
  buildGrid();
  $('grid-size').addEventListener('change', buildGrid);
});
