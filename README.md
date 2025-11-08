# TicTacToeRL — Paper‑faithful RL Tic‑Tac‑Toe (3×3)

A compact reinforcement‑learning implementation of Tic‑Tac‑Toe with a Tkinter GUI and a headless trainer. The project mirrors the classic “paper” setup: a linear value function over hand‑crafted board features, greedy play with random tie‑breaks, TD(0) with LMS updates, and optional exploration.

---

## Highlights

- **GUI app** to train, evaluate, view logs, and play against the agent.
- **Preset weight profiles** (snapshots by total self‑play games) that can be applied from the GUI.
- **Self‑play evaluation** (paper style) and **vs. Random** evaluation.
- **Model I/O**: save the current weights to JSON and load them back.
- **Headless trainer** for faster long‑runs (no GUI).
- **Stability**: weight, target, prediction, and error **clipping** to avoid divergence.

> The agent uses a **linear value function** \
> \\( \\, \hat{V}(s) = w_0 + \sum_{i=1}^{6} w_i x_i(s) \\, \\) \
> over six intuitive board features (open‑line counts, two‑in‑a‑row threats, and completed lines) measured **from the current player’s perspective**.

---

## Repo layout

```
.
├── tictactoe_rl.py            # GUI app (training, eval, play, presets, plots)
├── tictactoe_without_gui.py   # Headless trainer (fast self-play loops)
└── README.md                  # This file
```

---

## Requirements

- Python ≥ 3.9
- Tkinter (usually ships with Python on Windows/macOS; on Linux install via your distro)
- matplotlib

Install Python deps (matplotlib only):
```bash
pip install matplotlib
```

> If Tkinter is missing on Linux, for example on Ubuntu/Debian:
> ```bash
> sudo apt-get install python3-tk
> ```

---

## Quick start

### 1) Run the GUI

```bash
python tictactoe_rl.py
```

**What you can do in the GUI:**
- **Apply Preset:** choose a preset (keyed by number of training games) and load the corresponding weights.
- **Train N:** run N self‑play games with TD(0)+LMS updates. A progress bar and a log panel show status.
- **Evaluate (Self‑Play):** run self‑play games **without training** to measure W/D/L with fixed weights.
- **Eval vs Random:** pit the agent against a random player as **X** or **O**.
- **Save/Load Model:** save current \\(w_0..w_6\\) to JSON or load a saved JSON.
- **Plot Rates / Weights:** visualize win/draw/loss rates over snapshots and weight trajectories.

> Tip: to mimic the “paper” setup precisely, keep evaluation **greedy** (no exploration) and start games with the agent as **X**.


### 2) Train headless (fast)

```bash
python tictactoe_without_gui.py
```

This script runs a long self‑play training loop, prints periodic progress, and saves the final model to a JSON (e.g., `trained_model_1.5M.json`). Adjust constants at the top of the file:

```python
NUM_GAMES_TO_TRAIN = 1_500_000
LEARNING_RATE = 0.05
SNAPSHOT_EVERY = 50_000
```

---

## Learning rate, exploration, and stability

- **Learning rate (`lr`)**: set when creating the agent, e.g. `Agent(lr=0.4)` in the GUI app or `LEARNING_RATE` in the headless runner. Smaller values are more stable; larger can learn faster but may oscillate.
- **Exploration (`epsilon`)**: in `PerformanceSystem`, `epsilon` controls ε‑greedy random moves during **training**. For evaluation, the code forces pure greedy selection.
- **Clipping**: four clips keep training healthy:
  - `CLIP_W` bounds individual weights.
  - `CLIP_PRED` bounds value predictions.
  - `CLIP_TGT` bounds TD targets.
  - `CLIP_ERR` bounds TD errors before the LMS step.

These guards prevent runaway values and make long runs far more robust.

---

## Features (x₁..x₆) at a glance

For the player to move (`me`) against the opponent (`you`):

1. **x1** — count of `me` marks in **open** lines (no `you` marks there).
2. **x2** — count of `you` marks in **open** lines (no `me` marks).
3. **x3** — number of two‑in‑a‑row + one empty for `me` on chosen lines.
4. **x4** — number of two‑in‑a‑row + one empty for `you` on chosen lines.
5. **x5** — completed three‑in‑a‑row for `me` on chosen lines.
6. **x6** — completed three‑in‑a‑row for `you` on chosen lines.

By default, lines include rows, columns, **and diagonals** (configurable via flags near the top of the file).

---

## Reproducing “paper‑style” results

- Use **self‑play** training.
- Evaluate with **greedy policy** (no ε) and **agent starts as X** if that’s how the reference results were reported.
- Consider **preset weights** to jump to known good regions and then continue training.

If you want the agent to be strong as both first and second player, train with **mixed starting sides**; just note it can change the W/D/L distribution compared to always‑X training.

---

## Models and presets

The GUI includes a dictionary of **preset weights** keyed by training game counts. Choose one in the dropdown, click **Apply Preset**, and the agent immediately uses those weights. You can then continue training from there or evaluate/play.

---

## Saving and loading models

- **Save**: click **Save Model** and choose a path—weights (`w0..w6`) and `lr` are serialized to JSON.
- **Load**: click **Load Model** to use a previously saved JSON (the GUI rewires components automatically).

---

## Roadmap / Ideas

- GUI controls for **learning rate** and **ε** (live‑tunable).
- **Decay schedules** for learning rate and ε.
- A 4×4 or NxN variant and alternative feature sets.
- Compare against **Minimax** baseline automatically.
- Export **training curves** as images/CSV for reports.

---

## License

MIT

---

## Acknowledgments

- Classic linear‑value RL setup for Tic‑Tac‑Toe.
- Thanks to open‑source examples of Minimax Tic‑Tac‑Toe used for baseline comparisons.
