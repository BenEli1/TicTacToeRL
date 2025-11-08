# tictactoe_rl_random_start.py Which means the agent doesnt always start first!
# Paper-faithful RL Tic-Tac-Toe (3x3) with GUI, logs, presets, training, evaluation, and play-vs-agent.
# - 4 components: PerformanceSystem, Critic, Generalizer, ExperimentGenerator
# - Greedy policy, random tie-breaks, correct perspective (minimize opponent's next-state value)
# - TD(0) with LMS (optional training) + clipping for stability
# - Preset weight profiles (your table) selectable from the GUI
# - Evaluate paper-style (self-play) and vs Random
# - Play against the current model (choose X/O)
# - Log pane for training/eval/game updates


from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
from dataclasses import dataclass
from typing import List, Tuple, Dict
import random, math, json, time
import matplotlib.pyplot as plt

# ------------ Preset weights (Games → [w0..w6]) ------------
PRESET_WEIGHTS: Dict[int, List[float]] = {
    1000:   [75.2,  -8.4,  -166.8, -39.4, -171.0,  55.4, -115.1],
    10000:  [70.6,  -9.9,  -122.3,  31.0,  -93.2,  43.0, -208.4],
    100000: [72.8,  -0.2,   -23.2,  54.4, -158.2,  45.9, -208.0],
    200000: [44.3,  -9.4,     1.0,  38.0,  -85.9,  56.8, -175.5],
    300000: [81.4, -157.9,   -1.1,  16.9,  -15.4,  23.6, -202.8],
    400000: [62.2, -157.9,  -19.0,  42.8,  -34.0,  41.0, -197.1],
    500000: [93.6,  -17.6,  -53.5,  29.5, -135.3,  29.8, -227.4],
    600000: [97.9, -168.2,  -51.2,  29.0,   -7.2,  51.5, -218.5],
    700000: [67.3, -168.2,  -41.9,  27.0,  -47.5,  40.8, -186.5],
    800000: [77.1, -168.2,    -2.2, 22.9,  -20.1,  62.5, -198.9],
    900000: [84.2, -168.2,  -28.9,  29.4,   10.5,  41.0, -197.4],
    1000000:[67.0, -168.2,   11.4,  16.8,   12.0,  34.1, -197.5],
    1100000:[61.7, -168.2,  -72.9,  21.6,  -20.8,  70.0, -203.9],
    1200000:[95.4, -168.2,  -61.6,  76.9,  -25.1,  33.0, -160.1],
    1300000:[58.1, -168.2,  -76.4,  21.6,  -19.4,  36.6, -183.3],
    1400000:[60.8, -168.2,  -36.4,  30.6,   -6.0,  47.3, -209.5],
    1500000:[93.4, -168.2,  -27.3,  12.4,   -6.0,  18.9, -198.6],
}

# ------------ Board basics ------------
X, O, E = 'X', 'O', ' '
Board = List[str]
ROW = [(0,1,2),(3,4,5),(6,7,8)]
COL = [(0,3,6),(1,4,7),(2,5,8)]
RC  = ROW + COL
DIAG= [(0,4,8),(2,4,6)]
ALL = RC + DIAG

def legal_moves(b: Board) -> List[int]:
    return [i for i,v in enumerate(b) if v == E]

def apply(b: Board, mv: int, p: str) -> Board:
    nb = b[:]; nb[mv]=p; return nb

def winner(b: Board) -> str|None:
    for i,j,k in ALL:
        if b[i] != E and b[i] == b[j] == b[k]:
            return b[i]
    return 'D' if E not in b else None

USE_DIAGONALS_FOR_OPEN_COUNTS = True
USE_DIAGONALS_FOR_TWOS        = True
USE_DIAGONALS_FOR_THREES      = True

LINES_OPEN   = RC + (DIAG if USE_DIAGONALS_FOR_OPEN_COUNTS else [])
LINES_TWOS   = RC + (DIAG if USE_DIAGONALS_FOR_TWOS else [])
LINES_THREES = RC + (DIAG if USE_DIAGONALS_FOR_THREES else [])
# ------------ Features (paper-lean; your spec) ------------
def features(b: Board, me: str) -> List[float]:
    you = O if me == X else X
    x1 = x2 = x3 = x4 = x5 = x6 = 0.0

    for a, b1, c in LINES_OPEN:
        line = [b[a], b[b1], b[c]]
        if you not in line: x1 += line.count(me)
        if me  not in line: x2 += line.count(you)

    for a, b1, c in LINES_TWOS:
        line = [b[a], b[b1], b[c]]
        if (line[0]==me and line[1]==me and line[2]==E): x3 += 1
        if (line[0]==me and line[1]==E  and line[2]==me): x3 += 1
        if (line[0]==E  and line[1]==me and line[2]==me): x3 += 1
        if (line[0]==you and line[1]==you and line[2]==E): x4 += 1
        if (line[0]==you and line[1]==E   and line[2]==you): x4 += 1
        if (line[0]==E   and line[1]==you and line[2]==you): x4 += 1

    for a, b1, c in LINES_THREES:
        line = [b[a], b[b1], b[c]]
        if line.count(me)  == 3: x5 += 1
        if line.count(you) == 3: x6 += 1

    return [x1, x2, x3, x4, x5, x6]


# ------------ Value function (clipped/stable) ------------
@dataclass
class LinearValue:
    w: List[float]
    lr: float = 0.05
    CLIP_W=200.0; CLIP_PRED=200.0; CLIP_TGT=200.0; CLIP_ERR=50.0

    def _finite(self):
        if any(not math.isfinite(x) for x in self.w):
            self.w=[0.5]*7

    def predict(self, b: Board, me: str) -> float:
        self._finite()
        xs=features(b,me)
        y=self.w[0]+sum(wi*xi for wi,xi in zip(self.w[1:],xs))
        if y > self.CLIP_PRED: y = self.CLIP_PRED
        elif y < -self.CLIP_PRED: y = -self.CLIP_PRED
        return y

    def update_lms(self, b: Board, me: str, target: float):
        self._finite()
        if target > self.CLIP_TGT: target = self.CLIP_TGT
        elif target < -self.CLIP_TGT: target = -self.CLIP_TGT
        xs=features(b,me)
        y=self.w[0]+sum(wi*xi for wi,xi in zip(self.w[1:],xs))
        err = target - y
        if err > self.CLIP_ERR: err = self.CLIP_ERR
        elif err < -self.CLIP_ERR: err = -self.CLIP_ERR
        self.w[0]+=self.lr*err
        for i,xi in enumerate(xs, start=1):
            self.w[i]+=self.lr*err*xi
        for i in range(len(self.w)):
            if self.w[i] > self.CLIP_W: self.w[i] = self.CLIP_W
            elif self.w[i] < -self.CLIP_W: self.w[i] = -self.CLIP_W

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"w": self.w, "lr": self.lr}, f)

    @staticmethod
    def load(path: str) -> "LinearValue":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        w = list(map(float, d["w"]))
        lr = float(d.get("lr", 0.05))
        lv = LinearValue(w=w, lr=lr)
        lv._finite()
        return lv

# ------------ Components ------------
class PerformanceSystem:
    def __init__(self, v: LinearValue): 
        self.v = v
        self.epsilon = 0.1 

    def choose_best(self, b: Board, player: str) -> int:
        
        if random.random() < self.epsilon:
            moves = legal_moves(b)
            return random.choice(moves) if moves else 0

        opp=O if player==X else X
        best,score=None,None
        moves=legal_moves(b)
        random.shuffle(moves)
        for mv in moves:
            nb=apply(b,mv,player)
            s = self.v.predict(nb, player)
            if score is None or s>score: score=s; best=mv
        return best if best is not None else random.choice(moves) if moves else 0

class Critic:
    WIN,DRAW,LOSS=100.0,0.0,-100.0
    
    def util(self, res:str, player:str)->float:
        """Calculates utility based on the game result for the player."""
        if res=='D': return self.DRAW
        return self.WIN if res==player else self.LOSS
        
    def build_examples(self, trace:List[Tuple[Board,str]], res:str, v:LinearValue):
        if not trace: return []
        
        s_last, p_last = trace[-1]
        final_utility = self.util(res, p_last) 
        
        ex=[]
        for s_t,p_t in trace:
            ex.append((s_t, p_t, final_utility)) 
            
        return ex

class Generalizer:
    def __init__(self,v:LinearValue): self.v=v
    def fit(self,examples): 
        for b,p,t in examples: self.v.update_lms(b,p,t)

class ExperimentGenerator:
    def __init__(self, perf:PerformanceSystem): self.perf=perf
        
    def self_play(self,start:str)->Tuple[List[Tuple[Board,str]],str]:
        b=[E]*9; p=start; trace=[]
        while True:
            trace.append((b[:],p))
            
            mv=self.perf.choose_best(b,p)
            
            b=apply(b,mv,p)
            res=winner(b)
            if res is not None: return trace,res
            p=O if p==X else X

# ------------ Agent wrapper ------------
class Agent:
    def __init__(self, lr=0.05, preset=None):
        base=0.5
        w = ([base]*7 if preset is None else list(preset))
        self.value=LinearValue(w=w, lr=lr)
        self.perf=PerformanceSystem(self.value)
        self.critic=Critic()
        self.gen=Generalizer(self.value)
        self.exp=ExperimentGenerator(self.perf)
        self.reset_stats()

        self.hist_g=[]; self.hist_wr=[]; self.hist_dr=[]; self.hist_lr=[]; self.hist_w=[]

    def reset_stats(self):
        self.games=0; self.wins=0; self.draws=0; self.losses=0

    def snapshot(self):
        g=self.games or 1
        self.hist_g.append(self.games)
        self.hist_wr.append(self.wins/g)
        self.hist_dr.append(self.draws/g)
        self.hist_lr.append(self.losses/g)
        self.hist_w.append(self.value.w[:])

    def train(self, n:int, snap_every:int=5000, progress_cb=None, log_cb=None):
        t0=time.time()
        for k in range(1,n+1):
            start = X if ((self.games+1)%2==1) else O
            trace,res=self.exp.self_play(start)
            self.gen.fit(self.critic.build_examples(trace,res,self.value))
            self.games+=1
            if res=='D': self.draws+=1
            elif res==X: self.wins+=1
            else: self.losses+=1

            if progress_cb and (k % max(1, n//100) == 0 or k==n):
                progress_cb(k, n)
            if log_cb and (self.games % max(1000, n//20) == 0 or k==n):
                wr=self.wins/self.games; dr=self.draws/self.games; lr=self.losses/self.games
                log_cb(f"[TRAIN] {self.games} games | Win={wr:.3f} Draw={dr:.3f} Loss={lr:.3f}")

            if self.games % snap_every==0: self.snapshot()
        self.snapshot()
        if log_cb:
            dt=time.time()-t0
            log_cb(f"[TRAIN] Done {n} games in {dt:.2f}s | totals: W/D/L = {self.wins}/{self.draws}/{self.losses}")

    def evaluate_self_play(self, n:int, snap_every:int=0, log_cb=None):
        original_epsilon = self.perf.epsilon
        self.perf.epsilon = 0.0
        g0=self.games; w0=self.wins; d0=self.draws; l0=self.losses
        wins=losses=draws=0
        t0=time.time()
        for k in range(1,n+1):
            start=X if (k%2==1) else O
            b=[E]*9; p=start
            while True:
                mv=self.perf.choose_best(b,p)
                b=apply(b,mv,p)
                res=winner(b)
                if res is not None:
                    if res=='D': draws+=1
                    elif res=='X': wins+=1
                    else: losses+=1
                    break
                p=O if p==X else X
            if log_cb and (k % max(1000, n//10) == 0 or k==n):
                log_cb(f"[EVAL-SP] {k}/{n} → W={wins} D={draws} L={losses}")
        wdr = (wins/draws) if draws>0 else float('inf')
        if log_cb:
            dt=time.time()-t0
            log_cb(f"[EVAL-SP] Done {n} games in {dt:.2f}s | W={wins} D={draws} L={losses} | Win/Draw={wdr:.2f}")
        self.games, self.wins, self.draws, self.losses = g0, w0, d0, l0
        return wins, losses, draws, wdr

    def evaluate_vs_random(self, n:int, agent_side='X', log_cb=None):
        wins=losses=draws=0
        t0=time.time()
        for k in range(1,n+1):
            b=[E]*9; p='X'
            while True:
                if p==agent_side:
                    mv=self.perf.choose_best(b,p)
                else:
                    mv=random.choice(legal_moves(b))
                b=apply(b,mv,p)
                res=winner(b)
                if res is not None:
                    if res=='D': draws+=1
                    elif res==agent_side: wins+=1
                    else: losses+=1
                    break
                p=O if p==X else X
            if log_cb and (k % max(1000, n//10) == 0 or k==n):
                log_cb(f"[EVAL-RND] {k}/{n} → W={wins} D={draws} L={losses}")
        wr=wins/n; dr=draws/n; lr=losses/n
        if log_cb:
            dt=time.time()-t0
            log_cb(f"[EVAL-RND] Done {n} games in {dt:.2f}s | Win={wr:.3f} Draw={dr:.3f} Loss={lr:.3f}")
        return wr, dr, lr

# ------------ GUI ------------
class App:
    def __init__(self, root):
        self.root=root
        root.title("RL Tic-Tac-Toe (Paper) — Presets, Training, Eval, Play")
        self.agent=Agent()
        self.human_side = tk.StringVar(value="O")
        self.board=[E]*9
        self._build_ui()
        self._new_game()

    def log(self, msg:str):
        self.logbox.insert(tk.END, msg+"\n")
        self.logbox.see(tk.END)
        self.root.update_idletasks()

    def _build_ui(self):
        top=ttk.Frame(self.root, padding=8); top.pack(fill=tk.X)
        ttk.Label(top,text="Preset weights (Games):").pack(side=tk.LEFT)
        self.preset_var=tk.StringVar(value=str(min(PRESET_WEIGHTS.keys())))
        ttk.Combobox(top,textvariable=self.preset_var,values=[str(k) for k in sorted(PRESET_WEIGHTS.keys())],width=12,state="readonly").pack(side=tk.LEFT,padx=4)
        ttk.Button(top,text="Apply Preset",command=self.apply_preset).pack(side=tk.LEFT,padx=6)
        ttk.Button(top,text="Save Model",command=self.save_model).pack(side=tk.LEFT,padx=4)
        ttk.Button(top,text="Load Model",command=self.load_model).pack(side=tk.LEFT,padx=4)

        mid=ttk.Frame(self.root, padding=(8,4)); mid.pack(fill=tk.X)
        ttk.Label(mid,text="Train N:").grid(row=0,column=0,sticky="w")
        self.train_n=tk.StringVar(value="0")
        ttk.Entry(mid,textvariable=self.train_n,width=10).grid(row=0,column=1,sticky="w")
        ttk.Button(mid,text="Train",command=self.do_train).grid(row=0,column=2,padx=6,sticky="w")

        ttk.Label(mid,text="Eval (Self-Play) N:").grid(row=0,column=3,sticky="w",padx=(16,2))
        self.eval_n=tk.StringVar(value="100000")
        ttk.Entry(mid,textvariable=self.eval_n,width=10).grid(row=0,column=4,sticky="w")
        ttk.Button(mid,text="Evaluate",command=self.do_eval_sp).grid(row=0,column=5,padx=6,sticky="w")

        ttk.Label(mid,text="Eval vs Random N:").grid(row=1,column=0,sticky="w",pady=(4,0))
        self.eval_rnd_n=tk.StringVar(value="50000")
        ttk.Entry(mid,textvariable=self.eval_rnd_n,width=10).grid(row=1,column=1,sticky="w",pady=(4,0))
        self.rnd_side=tk.StringVar(value="X")
        ttk.Radiobutton(mid,text="Agent as X",variable=self.rnd_side,value="X").grid(row=1,column=2,sticky="w",pady=(4,0))
        ttk.Radiobutton(mid,text="Agent as O",variable=self.rnd_side,value="O").grid(row=1,column=3,sticky="w",pady=(4,0))
        ttk.Button(mid,text="Eval vs Random",command=self.do_eval_rnd).grid(row=1,column=4,padx=6,sticky="w",pady=(4,0))

        ttk.Button(mid,text="Plot Rates",command=self.plot_rates).grid(row=0,column=6,padx=(16,4))
        ttk.Button(mid,text="Plot Weights",command=self.plot_weights).grid(row=0,column=7,padx=4)

        prog=ttk.Frame(self.root, padding=(8,0,8,4)); prog.pack(fill=tk.X)
        self.progress=ttk.Progressbar(prog, orient="horizontal", mode="determinate", length=360)
        self.progress.pack(side=tk.LEFT)
        self.progress["value"]=0; self.progress["maximum"]=100
        self.status=tk.StringVar(value="Ready.")
        ttk.Label(prog,textvariable=self.status).pack(side=tk.LEFT,padx=8)

        play=ttk.Frame(self.root, padding=8); play.pack()
        sidef=ttk.Frame(play); sidef.grid(row=0,column=0,sticky="n")
        ttk.Label(sidef,text="Your side:").pack(anchor="w")
        ttk.Radiobutton(sidef,text="X",variable=self.human_side,value="X",command=self._new_game).pack(anchor="w")
        ttk.Radiobutton(sidef,text="O",variable=self.human_side,value="O",command=self._new_game).pack(anchor="w")
        ttk.Button(sidef,text="New Game",command=self._new_game).pack(pady=6,anchor="w")

        boardf=ttk.Frame(play); boardf.grid(row=0,column=1,padx=16)
        self.btns=[]
        for r in range(3):
            for c in range(3):
                i=r*3+c
                b=tk.Button(boardf, text=" ", width=4, height=2,
                            font=("Segoe UI", 24, "bold"),
                            command=lambda idx=i: self.on_cell(idx))
                b.grid(row=r,column=c,padx=3,pady=3)
                self.btns.append(b)

        logf=ttk.Frame(self.root, padding=(8,4,8,8)); logf.pack(fill=tk.BOTH, expand=True)
        ttk.Label(logf,text="Logs").pack(anchor="w")
        self.logbox = ScrolledText(logf, height=12, wrap=tk.WORD)
        self.logbox.pack(fill=tk.BOTH, expand=True)

        self.weights_txt=tk.Text(self.root,height=3,width=90)
        self.weights_txt.pack(padx=8,pady=(0,8))
        self._show_weights()

    def save_model(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")], initialfile="model.json")
        if not path: return
        self.agent.value.save(path)
        self.log(f"[MODEL] Saved to {path}")

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if not path: return
        try:
            lv = LinearValue.load(path)
            self.agent.value = lv
            self.agent.perf = PerformanceSystem(self.agent.value)
            self.agent.critic = Critic()
            self.agent.gen = Generalizer(self.agent.value)
            self.agent.exp = ExperimentGenerator(self.agent.perf)
            self.log(f"[MODEL] Loaded from {path}")
            self._show_weights()
            self._new_game()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_preset(self):
        try:
            k=int(self.preset_var.get())
        except:
            messagebox.showerror("Error","Invalid preset key."); return
        self.agent=Agent(preset=PRESET_WEIGHTS[k])
        self.status.set(f"Applied preset for {k} games.")
        self.log(f"[PRESET] Applied preset {k}: {PRESET_WEIGHTS[k]}")
        self._show_weights()
        self._new_game()

    def do_train(self):
        try:
            n=int(self.train_n.get()); assert n>=0
        except:
            messagebox.showerror("Error","Enter a non-negative integer for Train."); return
        if n==0:
            self.status.set("Skipped training (N=0)."); return

        self.progress["value"]=0; self.progress["maximum"]=n
        self.status.set(f"Training {n} games...")
        self.root.update_idletasks()
        self.log(f"[TRAIN] Start {n} games")

        def on_prog(done,total):
            self.progress["value"]=done
            g=max(1, self.agent.games)
            wr=self.agent.wins/g; dr=self.agent.draws/g; lr=self.agent.losses/g
            self.status.set(f"Training: {done}/{total} | Total={g} | Win={wr:.3f} Draw={dr:.3f} Loss={lr:.3f}")
            self.root.update_idletasks()

        self.agent.train(n, snap_every=max(2000, n//20), progress_cb=on_prog, log_cb=self.log)
        self._show_weights()
        self._new_game()

    def do_eval_sp(self):
        try:
            n=int(self.eval_n.get()); assert n>0
        except:
            messagebox.showerror("Error","Enter a positive integer for Evaluate."); return
        self.log(f"[EVAL-SP] Start {n} self-play games")
        wins, losses, draws, wdr = self.agent.evaluate_self_play(n, log_cb=self.log)
        self.status.set(f"Self-Play: W={wins} D={draws} L={losses} | W/D={('%.2f'%wdr) if math.isfinite(wdr) else '—'}")

    def do_eval_rnd(self):
        try:
            n=int(self.eval_rnd_n.get()); assert n>0
        except:
            messagebox.showerror("Error","Enter a positive integer for Eval vs Random."); return
        side=self.rnd_side.get()
        self.log(f"[EVAL-RND] Start {n} games vs Random (Agent={side})")
        wr, dr, lr = self.agent.evaluate_vs_random(n, agent_side=side, log_cb=self.log)
        self.status.set(f"vs Random ({side}): Win={wr:.3f} Draw={dr:.3f} Loss={lr:.3f}")

    def plot_rates(self):
        if not self.agent.hist_g:
            messagebox.showinfo("Info","No snapshots yet. Train/Evaluate first."); return
        plt.figure()
        plt.plot(self.agent.hist_g,self.agent.hist_wr,label="Win rate")
        plt.plot(self.agent.hist_g,self.agent.hist_dr,label="Draw rate")
        plt.plot(self.agent.hist_g,self.agent.hist_lr,label="Loss rate")
        plt.xlabel("Total games (self-play)")
        plt.ylabel("Rate"); plt.title("Self-Play: Win/Draw/Loss")
        plt.legend(); plt.tight_layout(); plt.show()

    def plot_weights(self):
        if not self.agent.hist_w:
            plt.figure()
            labels=[f"w{i}" for i in range(len(self.agent.value.w))]
            plt.bar(labels,self.agent.value.w)
            plt.title("Current Weights (no history)")
            plt.tight_layout(); plt.show()
            return
        ws=list(zip(*self.agent.hist_w))
        plt.figure()
        for i,series in enumerate(ws):
            plt.plot(self.agent.hist_g,series,label=f"w{i}")
        plt.xlabel("Total games (self-play)")
        plt.ylabel("Weight value")
        plt.title("Weight Evolution (w0..w6)")
        plt.legend(); plt.tight_layout(); plt.show()

    def _show_weights(self):
        w=self.agent.value.w
        self.weights_txt.delete("1.0",tk.END)
        self.weights_txt.insert(tk.END, f"w0..w6 = {', '.join(f'{x:.3f}' for x in w)}")

    def _new_game(self):
        self.board=[E]*9
        for b in self.btns:
            b.config(text=" ", state=tk.NORMAL)
        hs=self.human_side.get()
        self.status.set(f"New game. You are {hs}.")
        self.log(f"[GAME] New game. Human={hs}, Agent={'O' if hs=='X' else 'X'}")
        if hs=="O":
            self._agent_move_if_needed()

    def on_cell(self, idx: int):
        if self.board[idx] != E: return
        human=self.human_side.get()
        self.board = apply(self.board, idx, human)
        self._refresh_board()
        res = winner(self.board)
        if res:
            self._finish_game(res); return
        self._agent_move_if_needed()

    def _agent_move_if_needed(self):
        human=self.human_side.get()
        agent_symbol = X if human == O else O
        if self._turn_of(agent_symbol):
            mv = self.agent.perf.choose_best(self.board, agent_symbol)
            self.board = apply(self.board, mv, agent_symbol)
            self.log(f"[GAME] Agent({agent_symbol}) played at cell {mv}")
            self._refresh_board()
            res = winner(self.board)
            if res:
                self._finish_game(res)

    def _turn_of(self, player: str) -> bool:
        moves_made = 9 - self.board.count(E)
        return (player == X and moves_made % 2 == 0) or (player == O and moves_made % 2 == 1)

    def _refresh_board(self):
        for i, b in enumerate(self.btns):
            t = self.board[i] if self.board[i] != E else " "
            b.config(text=t)
            if self.board[i] != E:
                b.config(state=tk.DISABLED)

    def _finish_game(self, res: str):
        for b in self.btns:
            b.config(state=tk.DISABLED)
        if res=='D':
            msg="Draw."
        else:
            msg=f"{res} wins."
        self.status.set("Game over: "+msg)
        self.log("[GAME] "+msg)

if __name__ == "__main__":
    random.seed(42)
    root=tk.Tk()
    App(root)
    root.mainloop()