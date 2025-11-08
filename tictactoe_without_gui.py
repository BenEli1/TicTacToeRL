import time
from tictactoe_rl import Agent

# Faster training without GUI
NUM_GAMES_TO_TRAIN = 1_500_000
LEARNING_RATE = 0.05
SNAPSHOT_EVERY = 50_000

agent = Agent(lr=LEARNING_RATE)

def log_to_console(msg):
    print(msg)

def progress_to_console(done, total):
    if done % SNAPSHOT_EVERY == 0:
        g = max(1, agent.games)
        wr = agent.wins / g
        dr = agent.draws / g
        lr = agent.losses / g
        print(f"Progress: {done}/{total} | Total={g} | Win={wr:.3f} Draw={dr:.3f} Loss={lr:.3f}")

print(f"Starting headless training for {NUM_GAMES_TO_TRAIN} games...")
t0 = time.time()

agent.train(
    NUM_GAMES_TO_TRAIN,
    snap_every=SNAPSHOT_EVERY,
    progress_cb=progress_to_console,
    log_cb=log_to_console
)

dt = time.time() - t0

print("\n--- Training Complete ---")
print(f"Total time: {dt:.2f} seconds")
print(f"Final Weights: w0..w6 = {', '.join(f'{x:.3f}' for x in agent.value.w)}")

model_filename = "trained_model_1.5M.json"
agent.value.save(model_filename)
print(f"Model saved to {model_filename}")