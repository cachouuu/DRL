// Constants
const ROWS = 4;
const COLS = 12;
const START = { r: 3, c: 0 };
const GOAL = { r: 3, c: 11 };

// Actions: 0=Up, 1=Right, 2=Down, 3=Left
const ACTIONS = [0, 1, 2, 3];
const ACTION_DIRS = [
    { dr: -1, dc: 0, symbol: '↑' },
    { dr: 0, dc: 1, symbol: '→' },
    { dr: 1, dc: 0, symbol: '↓' },
    { dr: 0, dc: -1, symbol: '←' }
];

// Environment
class CliffWalking {
    constructor() {
        this.reset();
    }

    reset() {
        this.state = { ...START };
        return this._stateToIndex(this.state);
    }

    _stateToIndex(s) {
        return s.r * COLS + s.c;
    }

    step(action) {
        let { r, c } = this.state;
        let { dr, dc } = ACTION_DIRS[action];
        
        let nr = Math.max(0, Math.min(ROWS - 1, r + dr));
        let nc = Math.max(0, Math.min(COLS - 1, c + dc));
        
        let nstate = { r: nr, c: nc };
        let reward = -1;
        let done = false;

        // Check cliff (bottom row, between start and goal)
        if (nr === 3 && nc > 0 && nc < 11) {
            reward = -100;
            nstate = { ...START };
        } else if (nr === 3 && nc === 11) {
            done = true;
        }

        this.state = nstate;
        return { nextState: this._stateToIndex(nstate), reward, done };
    }
}

// Agent Base Class
class RLAgent {
    constructor(alpha, gamma, epsilon) {
        this.alpha = alpha;
        this.gamma = gamma;
        this.epsilon = epsilon;
        this.numStates = ROWS * COLS;
        this.qTable = new Array(this.numStates).fill(0).map(() => new Array(ACTIONS.length).fill(0));
    }

    chooseAction(state) {
        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * ACTIONS.length);
        } else {
            let maxQ = Math.max(...this.qTable[state]);
            let bestActions = [];
            for (let a = 0; a < ACTIONS.length; a++) {
                if (this.qTable[state][a] === maxQ) bestActions.push(a);
            }
            return bestActions[Math.floor(Math.random() * bestActions.length)];
        }
    }
}

class QLearning extends RLAgent {
    update(s, a, r, ns, na, done) {
        let maxNextQ = done ? 0 : Math.max(...this.qTable[ns]);
        this.qTable[s][a] += this.alpha * (r + this.gamma * maxNextQ - this.qTable[s][a]);
    }
}

class SARSA extends RLAgent {
    update(s, a, r, ns, na, done) {
        let nextQ = done ? 0 : this.qTable[ns][na];
        this.qTable[s][a] += this.alpha * (r + this.gamma * nextQ - this.qTable[s][a]);
    }
}

// UI and Main Loop
let env = new CliffWalking();
let agent = null;
let agent2 = null; // for comparison
let isAnimating = false;
let isComparing = false;
let episodeCount = 0;
let maxEpisodes = 500;
let animationTimeout = null;

// Chart
let chartInstance = null;

// DOM Elements
const domGrid = document.getElementById('grid-container');
const btnStart = document.getElementById('btn-start');
const btnPause = document.getElementById('btn-pause');
const btnStep = document.getElementById('btn-step');
const btnRunFull = document.getElementById('btn-run-full');
const btnReset = document.getElementById('btn-reset');
const statusDisplay = document.getElementById('status-display');
const policySection = document.getElementById('policy-comparison');
const btnShowPolicy = document.getElementById('btn-show-policy');

// Initialization
function initGrid(containerId, isSmall = false) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            let cell = document.createElement('div');
            cell.className = 'grid-cell';
            cell.id = `${containerId}-cell-${r}-${c}`;
            if (r === 3 && c === 0) cell.classList.add('start');
            else if (r === 3 && c === 11) cell.classList.add('goal');
            else if (r === 3 && c > 0 && c < 11) cell.classList.add('cliff');
            container.appendChild(cell);
        }
    }
}

function updateAgentPos(r, c) {
    document.querySelectorAll('.agent').forEach(e => e.remove());
    const cell = document.getElementById(`grid-container-cell-${r}-${c}`);
    if (cell) {
        let agentDiv = document.createElement('div');
        agentDiv.className = 'agent';
        cell.appendChild(agentDiv);
    }
}

function initChart() {
    const ctx = document.getElementById('learningCurve').getContext('2d');
    if (chartInstance) chartInstance.destroy();
    
    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Q-learning Rewards',
                data: [],
                borderColor: '#e74c3c',
                backgroundColor: 'transparent',
                tension: 0.1,
                pointRadius: 0
            }, {
                label: 'SARSA Rewards',
                data: [],
                borderColor: '#3498db',
                backgroundColor: 'transparent',
                tension: 0.1,
                pointRadius: 0,
                hidden: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { min: -100, max: 0, title: { display: true, text: 'Total Reward' } },
                x: { title: { display: true, text: 'Episode' } }
            },
            animation: false
        }
    });
}

function updateChart(ep, qReward, sReward = null) {
    if (ep % 5 === 0 || ep === maxEpisodes) {
        chartInstance.data.labels.push(ep);
        chartInstance.data.datasets[0].data.push(qReward);
        if (sReward !== null) {
            chartInstance.data.datasets[1].data.push(sReward);
        }
        chartInstance.update();
    }
}

function updateChartFull(qData, sData = null) {
    let labels = Array.from({length: qData.length}, (_, i) => i + 1);
    chartInstance.data.labels = labels;
    
    // Smoothing function (simple moving average)
    function smooth(data, windowSize=10) {
        let res = [];
        for(let i=0; i<data.length; i++) {
            let start = Math.max(0, i - windowSize + 1);
            let subset = data.slice(start, i+1);
            let avg = subset.reduce((a,b)=>a+b, 0) / subset.length;
            res.push(avg);
        }
        return res;
    }

    let smoothQ = smooth(qData.map(v => Math.max(v, -100)));
    chartInstance.data.datasets[0].data = smoothQ;
    
    if (sData) {
        let smoothS = smooth(sData.map(v => Math.max(v, -100)));
        chartInstance.data.datasets[1].data = smoothS;
        chartInstance.data.datasets[1].hidden = false;
    }
    chartInstance.update();
}

function drawPolicy(agentToDraw, containerId) {
    initGrid(containerId, true);
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            if (r === 3 && c > 0) continue; // Skip cliff and goal
            let state = r * COLS + c;
            let qValues = agentToDraw.qTable[state];
            let maxQ = Math.max(...qValues);
            let bestA = qValues.indexOf(maxQ); 
            
            let cell = document.getElementById(`${containerId}-cell-${r}-${c}`);
            if (cell) {
                cell.innerHTML = `<span class="arrow">${ACTION_DIRS[bestA].symbol}</span>`;
            }
        }
    }
}

// Training Logic
async function runEpisode(currentAgent, animate = false, delayMs = 50) {
    let state = env.reset();
    let action = currentAgent.chooseAction(state);
    let totalReward = 0;
    
    if (animate) updateAgentPos(env.state.r, env.state.c);

    while (true) {
        if (!isAnimating && animate) break; // Pause check
        
        let { nextState, reward, done } = env.step(action);
        let nextAction = currentAgent.chooseAction(nextState);
        
        currentAgent.update(state, action, reward, nextState, nextAction, done);
        
        state = nextState;
        action = nextAction;
        totalReward += reward;

        if (animate) {
            updateAgentPos(env.state.r, env.state.c);
            await new Promise(r => { animationTimeout = setTimeout(r, delayMs) });
        }

        if (done) break;
    }
    return totalReward;
}

async function startTrainingAnimation() {
    isAnimating = true;
    btnStart.disabled = true;
    btnStep.disabled = true;
    btnRunFull.disabled = true;
    btnPause.disabled = false;
    document.getElementById('algorithm').disabled = true;
    
    let algoType = document.getElementById('algorithm').value;
    let delayMs = parseInt(document.getElementById('speed').value);
    
    if (!agent) resetEnvAndAgents();

    while (isAnimating && episodeCount < maxEpisodes) {
        episodeCount++;
        statusDisplay.innerText = `狀態: 訓練中 (Episode ${episodeCount}/${maxEpisodes})`;
        
        if (algoType === 'compare') {
            let rQ = await runEpisode(agent, false);
            let rS = await runEpisode(agent2, false);
            updateChart(episodeCount, Math.max(rQ, -100), Math.max(rS, -100));
        } else {
            let r = await runEpisode(agent, true, delayMs);
            updateChart(episodeCount, Math.max(r, -100));
        }
    }

    if (episodeCount >= maxEpisodes) {
        stopTraining();
        statusDisplay.innerText = `狀態: 訓練完成`;
        if (algoType === 'compare') {
            btnShowPolicy.style.display = 'block';
        } else {
            drawPolicy(agent, 'grid-container');
        }
    }
}

function stopTraining() {
    isAnimating = false;
    clearTimeout(animationTimeout);
    btnStart.disabled = false;
    btnStep.disabled = false;
    btnRunFull.disabled = false;
    btnPause.disabled = true;
    statusDisplay.innerText = `狀態: 已暫停 (Episode ${episodeCount}/${maxEpisodes})`;
}

async function runFullTraining() {
    if (!agent) resetEnvAndAgents();
    let algoType = document.getElementById('algorithm').value;
    statusDisplay.innerText = `狀態: 快速訓練中...`;
    
    // allow UI to update
    await new Promise(r => setTimeout(r, 50));
    
    let qRewards = [];
    let sRewards = [];

    // Run remaining episodes
    for (let i = episodeCount; i < maxEpisodes; i++) {
        if (algoType === 'compare') {
            qRewards.push(await runEpisode(agent, false));
            sRewards.push(await runEpisode(agent2, false));
        } else {
            qRewards.push(await runEpisode(agent, false));
        }
        episodeCount++;
    }

    if (algoType === 'compare') {
        updateChartFull(qRewards, sRewards);
        btnShowPolicy.style.display = 'block';
    } else {
        updateChartFull(qRewards);
        drawPolicy(agent, 'grid-container');
    }
    
    statusDisplay.innerText = `狀態: 訓練完成`;
}

async function stepOneEpisode() {
    if (!agent) resetEnvAndAgents();
    if (episodeCount >= maxEpisodes) return;
    
    let algoType = document.getElementById('algorithm').value;
    let delayMs = parseInt(document.getElementById('speed').value);
    episodeCount++;
    
    if (algoType === 'compare') {
        let rQ = await runEpisode(agent, false);
        let rS = await runEpisode(agent2, false);
        updateChart(episodeCount, Math.max(rQ, -100), Math.max(rS, -100));
    } else {
        let r = await runEpisode(agent, true, delayMs);
        updateChart(episodeCount, Math.max(r, -100));
    }
    statusDisplay.innerText = `狀態: 已暫停 (Episode ${episodeCount}/${maxEpisodes})`;
}

function resetEnvAndAgents() {
    stopTraining();
    document.getElementById('algorithm').disabled = false;
    
    let alpha = parseFloat(document.getElementById('alpha').value);
    let gamma = parseFloat(document.getElementById('gamma').value);
    let epsilon = parseFloat(document.getElementById('epsilon').value);
    maxEpisodes = parseInt(document.getElementById('episodes').value);
    
    let algoType = document.getElementById('algorithm').value;
    isComparing = algoType === 'compare';
    
    if (algoType === 'qlearning' || isComparing) agent = new QLearning(alpha, gamma, epsilon);
    else agent = new SARSA(alpha, gamma, epsilon);
    
    if (isComparing) agent2 = new SARSA(alpha, gamma, epsilon);

    episodeCount = 0;
    
    initGrid('grid-container');
    initChart();
    
    if (isComparing) {
        chartInstance.data.datasets[1].hidden = false;
        chartInstance.data.datasets[0].label = 'Q-learning';
        btnStart.innerText = '開始比較(無動畫)';
    } else {
        chartInstance.data.datasets[1].hidden = true;
        chartInstance.data.datasets[0].label = algoType === 'qlearning' ? 'Q-learning' : 'SARSA';
        btnStart.innerText = '開始動畫訓練';
    }
    chartInstance.update();
    
    policySection.style.display = 'none';
    btnShowPolicy.style.display = isComparing ? 'inline-block' : 'none';
    if(!isComparing) btnShowPolicy.style.display = 'none';

    statusDisplay.innerText = `狀態: 準備就緒`;
}

// Event Listeners
document.getElementById('algorithm').addEventListener('change', resetEnvAndAgents);

['epsilon', 'alpha', 'gamma', 'episodes', 'speed'].forEach(id => {
    document.getElementById(id).addEventListener('input', (e) => {
        document.getElementById(`${id}-val`).innerText = e.target.value;
    });
});

btnStart.addEventListener('click', startTrainingAnimation);
btnPause.addEventListener('click', stopTraining);
btnStep.addEventListener('click', stepOneEpisode);
btnRunFull.addEventListener('click', runFullTraining);
btnReset.addEventListener('click', resetEnvAndAgents);

btnShowPolicy.addEventListener('click', () => {
    if (agent && isComparing && episodeCount >= maxEpisodes) {
        policySection.style.display = 'grid';
        drawPolicy(agent, 'grid-qlearning');
        drawPolicy(agent2, 'grid-sarsa');
    } else {
        alert("請先選擇「比較兩者」並完成完整訓練。");
    }
});

// Init on load
window.onload = resetEnvAndAgents;
