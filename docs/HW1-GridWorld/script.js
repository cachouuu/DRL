/**
 * docs/HW1-GridWorld/script.js
 * --------------------------------
 * Static GitHub Pages demo for the GridWorld Q-Learning HW.
 * No Python / server required — everything runs in the browser.
 *
 * Features:
 *   • Selectable grid size n (5–9)
 *   • Clickable cells: Start (1st), Goal (2nd), Obstacles (next n–2)
 *   • Random policy arrows on free cells (after "Show Policy" click)
 *   • Simple state-value display (uniform random walk value estimate)
 */

/* ---------------------------------------------------------------
   Constants / config
--------------------------------------------------------------- */
const ACTIONS = ['↑', '↓', '←', '→'];   // arrow symbols

/* ---------------------------------------------------------------
   State
--------------------------------------------------------------- */
let n            = 5;
let maxObstacles = n - 2;
let phase        = 'start';   // 'start'|'goal'|'obstacle'|'done'
let startPos     = null;      // "row,col"
let endPos       = null;
let blockPos     = [];        // [[r,c], ...]

/* ---------------------------------------------------------------
   DOM helpers
--------------------------------------------------------------- */
const $ = id => document.getElementById(id);

/* ---------------------------------------------------------------
   Build the grid
--------------------------------------------------------------- */
function buildGrid() {
  n            = parseInt($('grid-size').value, 10);
  maxObstacles = n - 2;
  phase        = 'start';
  startPos     = null;
  endPos       = null;
  blockPos     = [];

  const container = $('grid-container');
  container.innerHTML = '';

  for (let r = 0; r < n; r++) {
    const row = document.createElement('div');
    row.className = 'grid-row';
    for (let c = 0; c < n; c++) {
      const cell = document.createElement('div');
      cell.className  = 'cell';
      cell.id         = `cell-${r}-${c}`;
      cell.dataset.row = r;
      cell.dataset.col = c;
      cell.innerHTML  = `<span class="cell-coord">${r},${c}</span>`;
      cell.addEventListener('click', () => cellClicked(r, c));
      row.appendChild(cell);
    }
    container.appendChild(row);
  }

  $('info-panel').style.display = 'none';
  updateStatus();
  updateLegend();
}

/* ---------------------------------------------------------------
   Cell click handler
--------------------------------------------------------------- */
function cellClicked(row, col) {
  const key = `${row},${col}`;

  if (phase === 'start') {
    markCell(row, col, 'start');
    startPos = key;
    phase = 'goal';

  } else if (phase === 'goal') {
    if (key === startPos) { alert('Goal must differ from Start.'); return; }
    markCell(row, col, 'goal');
    endPos = key;
    phase = 'obstacle';

  } else if (phase === 'obstacle') {
    if (key === startPos || key === endPos) {
      alert('Obstacle cannot overlap Start or Goal.'); return;
    }
    const el = $(`cell-${row}-${col}`);
    if (el.classList.contains('obstacle')) {
      // toggle off
      el.classList.remove('obstacle');
      el.querySelector('.cell-arrow') && el.querySelector('.cell-arrow').remove();
      blockPos = blockPos.filter(([r, c]) => !(r === row && c === col));
    } else {
      if (blockPos.length >= maxObstacles) {
        alert(`Max ${maxObstacles} obstacle(s) for a ${n}×${n} grid.`); return;
      }
      markCell(row, col, 'obstacle');
      blockPos.push([row, col]);
      if (blockPos.length === maxObstacles) phase = 'done';
    }

  } else {
    // 'done' – allow toggling obstacles off
    const el = $(`cell-${row}-${col}`);
    if (el.classList.contains('obstacle')) {
      el.classList.remove('obstacle');
      blockPos = blockPos.filter(([r, c]) => !(r === row && c === col));
      phase = 'obstacle';
    }
  }

  clearPolicy();   // wipe arrows/values if grid changes
  updateStatus();
}

/* ---------------------------------------------------------------
   Apply a CSS class to a cell
--------------------------------------------------------------- */
function markCell(row, col, type) {
  const el = $(`cell-${row}-${col}`);
  el.classList.remove('start', 'goal', 'obstacle');
  el.classList.add(type);
  // remove any arrow left over
  const arrow = el.querySelector('.cell-arrow');
  if (arrow) arrow.remove();
  const val = el.querySelector('.cell-value');
  if (val)   val.remove();
}

/* ---------------------------------------------------------------
   Status bar
--------------------------------------------------------------- */
function updateStatus() {
  const bar       = $('status-bar');
  const remaining = maxObstacles - blockPos.length;

  if (phase === 'start') {
    bar.innerHTML = 'Click a cell to set the <strong>Start</strong> position.';
  } else if (phase === 'goal') {
    bar.innerHTML = 'Click a cell to set the <strong>Goal</strong> position.';
  } else if (phase === 'obstacle') {
    bar.innerHTML = `Click <strong>${remaining}</strong> more cell(s) to place Obstacles.`;
  } else {
    bar.innerHTML = '✅ Grid configured! Click <strong>Show Random Policy</strong> to display arrows.';
  }
}

/* ---------------------------------------------------------------
   Legend – update obstacle count dynamically
--------------------------------------------------------------- */
function updateLegend() {
  const el = $('legend-obstacle-count');
  if (el) el.textContent = n - 2;
}

/* ---------------------------------------------------------------
   Reset
--------------------------------------------------------------- */
function resetGrid() {
  buildGrid();   // rebuilds with current n selection
}

/* ---------------------------------------------------------------
   Random Policy: assign a random arrow to every free cell
   and compute a rough state value via 20-step random rollout.
--------------------------------------------------------------- */
function showPolicy() {
  if (phase !== 'done') {
    alert('Please finish configuring the grid first (set Start, Goal, and all Obstacles).');
    return;
  }

  clearPolicy();

  const [sr, sc]   = startPos.split(',').map(Number);
  const [er, ec]   = endPos.split(',').map(Number);
  const blockSet   = new Set(blockPos.map(([r, c]) => `${r},${c}`));
  const goalKey    = `${er},${ec}`;

  const deltas = { '↑': [-1,0], '↓': [1,0], '←': [0,-1], '→': [0,1] };

  // For each free cell compute a simple average return under random policy
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const key = `${r},${c}`;
      if (key === goalKey || blockSet.has(key)) continue;

      const el = $(`cell-${r}-${c}`);

      // Random arrow
      const arrowSym = ACTIONS[Math.floor(Math.random() * ACTIONS.length)];
      const arrowEl  = document.createElement('span');
      arrowEl.className   = 'cell-arrow';
      arrowEl.textContent = arrowSym;
      el.appendChild(arrowEl);

      // Simple Monte-Carlo value: average return over 30 random rollouts
      let totalReturn = 0;
      const ROLLOUTS  = 30;
      const HORIZON   = 30;
      const GAMMA     = 0.9;

      for (let trial = 0; trial < ROLLOUTS; trial++) {
        let cr = r, cc = c, ret = 0, discount = 1;
        for (let step = 0; step < HORIZON; step++) {
          const sym = ACTIONS[Math.floor(Math.random() * ACTIONS.length)];
          const [dr, dc] = deltas[sym];
          let nr = Math.max(0, Math.min(n - 1, cr + dr));
          let nc = Math.max(0, Math.min(n - 1, cc + dc));
          const nKey = `${nr},${nc}`;
          let reward = 0, done = false;
          if (nKey === goalKey)       { reward =  1; done = true; }
          else if (blockSet.has(nKey)){ reward = -1; done = true; }
          ret += discount * reward;
          discount *= GAMMA;
          cr = nr; cc = nc;
          if (done) break;
        }
        totalReturn += ret;
      }

      const val    = (totalReturn / ROLLOUTS).toFixed(2);
      const valEl  = document.createElement('span');
      valEl.className   = 'cell-value';
      valEl.textContent = val;
      el.appendChild(valEl);
    }
  }

  // Show goal marker
  const goalEl = $(`cell-${er}-${ec}`);
  if (goalEl) {
    const arrowEl = document.createElement('span');
    arrowEl.className   = 'cell-arrow';
    arrowEl.textContent = '🏁';
    goalEl.appendChild(arrowEl);
    const valEl = document.createElement('span');
    valEl.className   = 'cell-value';
    valEl.textContent = '1.00';
    goalEl.appendChild(valEl);
  }

  // Show info panel
  const panel = $('info-panel');
  panel.style.display = 'block';
  panel.innerHTML = `
    <h2>📊 Random Policy Summary</h2>
    <ul>
      <li>Grid size: <strong>${n} × ${n}</strong></li>
      <li>Start: <strong>${startPos}</strong> &nbsp;|&nbsp; Goal: <strong>${endPos}</strong></li>
      <li>Obstacles: <strong>${blockPos.map(([r,c])=>`(${r},${c})`).join(', ')}</strong></li>
      <li>Arrows show a uniformly random action per cell.</li>
      <li>Numbers estimate the expected discounted return (γ = 0.9) under the random policy via Monte-Carlo rollouts.</li>
    </ul>
  `;
}

/* ---------------------------------------------------------------
   Clear policy arrows / values
--------------------------------------------------------------- */
function clearPolicy() {
  document.querySelectorAll('.cell-arrow, .cell-value').forEach(el => el.remove());
  $('info-panel').style.display = 'none';
}

/* ---------------------------------------------------------------
   Initialise on page load
--------------------------------------------------------------- */
window.addEventListener('DOMContentLoaded', () => {
  buildGrid();
  $('grid-size').addEventListener('change', buildGrid);
});
