# HW1 — GridWorld Q-Learning

## Project Overview

This project implements a **GridWorld reinforcement learning environment** trained with a tabular **Q-Learning** algorithm. The user configures an **n × n** grid (n = 5–9) through a web interface, designating start, goal, and obstacle cells, and then triggers a training run that prints episode-by-episode output directly in the browser.

### Architecture

| File | Role |
|---|---|
| `app.py` | Flask web server — routes `/`, `/grid`, `/train` |
| `maze_env.py` | `GridWorld` environment (step / reset / reward) |
| `RL_brain.py` | `QLearningTable` agent (ε-greedy, Bellman update) |
| `run_this.py` | CLI training script, called as a subprocess by `app.py` |
| `templates/index.html` | Landing page — pick grid size |
| `templates/square.html` | Interactive grid — configure & start training |

### Reinforcement-Learning Parameters

| Parameter | Value |
|---|---|
| Learning rate α | 0.01 |
| Discount factor γ | 0.9 |
| Epsilon ε (greedy) | 0.9 |
| Max episodes | 100 |
| Max steps/episode | 200 |

---

## How to Run Locally (Flask)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/DRL.git
cd DRL/HW1-GridWorld
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Flask development server

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

### 5. Usage

1. Enter a grid size between **5 and 9**.
2. Click **Generate Grid →**.
3. Click a cell to set the **Start** (green), then another for the **Goal** (red).
4. Place exactly **n − 2** obstacle cells (gray).
5. Click **▶ Start Training** to run Q-Learning and see the output.

---

## GitHub Pages Demo

A fully static version of the UI (no Python required) is hosted via **GitHub Pages**:

👉 **https://\<your-username\>.github.io/DRL/HW1-GridWorld/**

The demo includes:

- Selectable grid size (5–9)
- Clickable cells for start / goal / obstacle placement
- **Random policy arrows** overlaid on each free cell
- Simple **state value display** (computed from the random policy)
- All logic runs in vanilla HTML + CSS + JavaScript — no server needed

---

## Git Commands to Push to GitHub

```bash
git add .
git commit -m "feat: add DRL HW1 GridWorld project with GitHub Pages demo"
git push
```

> **Enable GitHub Pages:** Go to your repository → *Settings → Pages → Source* → select **`/docs`** folder on the `main` branch.
