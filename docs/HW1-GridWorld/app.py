"""
app.py
------
Flask web application for the GridWorld Q-Learning homework project.

Routes
------
GET  /           → index.html  (user inputs grid size n)
POST /grid       → square.html (interactive n×n grid for setup)
POST /train      → launches run_this.py via subprocess, then
                   re-renders square.html showing training output

The training subprocess is run synchronously so that its stdout can be
captured and displayed to the user.
"""

import subprocess
import sys
import json
from flask import Flask, render_template, request

app = Flask(__name__)


# ------------------------------------------------------------------
# Route: landing page
# ------------------------------------------------------------------

@app.route('/', methods=['GET'])
def index():
    """Render the input form where the user picks grid size n."""
    return render_template('index.html')


# ------------------------------------------------------------------
# Route: generate grid
# ------------------------------------------------------------------

@app.route('/grid', methods=['POST'])
def grid():
    """
    Receive grid size from the form, validate it, and render the
    interactive grid page (square.html).

    Form field: n  (int, 5–9)
    """
    try:
        n = int(request.form.get('n', 5))
        # Clamp to allowed range
        n = max(5, min(9, n))
    except (TypeError, ValueError):
        n = 5

    return render_template('square.html', n=n, training_output=None)


# ------------------------------------------------------------------
# Route: start training
# ------------------------------------------------------------------

@app.route('/train', methods=['POST'])
def train():
    """
    Receive grid configuration from square.html, launch the Q-Learning
    training script, capture its output, and re-render square.html.

    Form fields
    -----------
    n        : int   - grid size
    startPos : str   - "row,col"
    endPos   : str   - "row,col"
    blockPos : str   - JSON array of [row, col] pairs, e.g. "[[1,2]]"
    """
    # --- Parse grid size ---
    try:
        n = int(request.form.get('n', 5))
        n = max(5, min(9, n))
    except (TypeError, ValueError):
        n = 5

    # --- Parse start position ---
    start_str = request.form.get('startPos', '0,0')
    try:
        s_row, s_col = [int(x) for x in start_str.split(',')]
    except Exception:
        s_row, s_col = 0, 0

    # --- Parse end (goal) position ---
    end_str = request.form.get('endPos', f'{n-1},{n-1}')
    try:
        e_row, e_col = [int(x) for x in end_str.split(',')]
    except Exception:
        e_row, e_col = n - 1, n - 1

    # --- Parse obstacle positions ---
    block_str = request.form.get('blockPos', '[]')
    try:
        blocks = json.loads(block_str)   # list of [row, col]
    except Exception:
        blocks = []

    # --- Build subprocess command ---
    # python run_this.py n s_row s_col e_row e_col block_json
    cmd = [
        sys.executable,          # same Python interpreter running Flask
        'run_this.py',
        str(n),
        str(s_row), str(s_col),
        str(e_row), str(e_col),
        json.dumps(blocks),
    ]

    # --- Run training (blocking) and capture output ---
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,          # safety timeout (seconds)
        )
        training_output = result.stdout
        if result.returncode != 0:
            training_output += f"\n[STDERR]\n{result.stderr}"
    except subprocess.TimeoutExpired:
        training_output = "Training timed out after 120 seconds."
    except Exception as exc:
        training_output = f"Error launching training: {exc}"

    # Re-render the grid page, passing training output
    return render_template(
        'square.html',
        n=n,
        start_pos=start_str,
        end_pos=end_str,
        block_pos=blocks,
        training_output=training_output,
    )


# ------------------------------------------------------------------
# Entry-point
# ------------------------------------------------------------------

if __name__ == '__main__':
    # Debug=True gives auto-reload during development
    app.run(debug=True, port=5000)
