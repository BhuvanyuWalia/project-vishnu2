"""
Project Vishnu — FastAPI Backend
=================================
Serves the Dueling DDQN agent for inference on the 200×200 IOR grid.

Endpoints:
    POST /navigate  — runs greedy episode, returns route
    GET  /grid      — returns the 200×200 grid + port locations
    GET  /health    — liveness check
    GET  /          — serves frontend
"""

import os, math, json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ─────────────────────────────────────────────
#  CONFIG — must match training notebook exactly
# ─────────────────────────────────────────────
CFG = {
    "ROWS": 200, "COLS": 200, "PAD": 5, "WINDOW": 11, "N_ACTIONS": 8,
    "CNN_FILTERS": [32, 64], "CNN_KERNEL": 3,
    "CNN_FC": 128, "COORD_FC": 64,
    "DECISION_FC": 256, "DUELING_FC": 128,
    "R_GOAL": 500.0, "R_BUMP": -20.0,
    "FUEL_COST_START": -1.0, "FUEL_COST_END": -3.0,
    "R_CLOSER": 3.0, "R_FARTHER": -1.5,
    "R_VISIT_PENALTY": -0.5, "R_REPEAT_PENALTY": -1.0,
    "MAX_FUEL": 6000,   # Generous for inference
}

ACTION_DELTAS = [
    (-1, 0),(-1,+1),(0,+1),(+1,+1),(+1,0),(+1,-1),(0,-1),(-1,-1)
]
ACTION_OPPOSITE = {0:4,1:5,2:6,3:7,4:0,5:1,6:2,7:3}

device = torch.device("cpu")

# ─────────────────────────────────────────────
#  GRID — loaded from JSON
# ─────────────────────────────────────────────
GRID_PATH = Path("indian_ocean_200x200.json")

def load_grid():
    if not GRID_PATH.exists():
        raise RuntimeError(f"Grid file not found: {GRID_PATH}")
    with open(GRID_PATH, "r") as f:
        data = json.load(f)
    grid = np.array(data, dtype=np.int32)
    assert grid.shape == (200, 200), f"Expected (200,200), got {grid.shape}"
    return grid

GRID = load_grid()

PORTS = {
    "Mumbai":        ( 75,  85),
    "Visakhapatnam": ( 77, 130),
    "Chennai":       (104,  116),
    "Kochi":         (117,  100),
    "Colombo":       (130, 117),
    "Karachi":       ( 39,  49),
    "Goa":           ( 98,  62),
    "Aden":          ( 80,  4),
    "Singapore":     (121, 198),
}

# Carve 3×3 water patches around ports
for name, (r, c) in PORTS.items():
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            rr, cc = r+dr, c+dc
            if 0 <= rr < 200 and 0 <= cc < 200:
                GRID[rr, cc] = 0

def make_padded_grid(grid, pad):
    rows, cols = grid.shape
    padded = np.ones((rows+2*pad, cols+2*pad), dtype=np.int32)
    padded[pad:pad+rows, pad:pad+cols] = grid
    return padded

PADDED_GRID = make_padded_grid(GRID, CFG["PAD"])
WATER_CELLS = list(zip(*np.where(GRID == 0)))

print(f"✅  Grid loaded: 200×200  |  Water cells: {len(WATER_CELLS):,}")

# ─────────────────────────────────────────────
#  DUELING DDQN NETWORK — identical to notebook
# ─────────────────────────────────────────────
class DuelingNavDQN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        f1, f2 = cfg["CNN_FILTERS"]
        k       = cfg["CNN_KERNEL"]

        # Vision branch: (1,11,11) → 11→9→7 → flatten 64×7×7=3136 → 128
        self.vision = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=k, padding=0),  nn.ReLU(),
            nn.Conv2d(f1, f2, kernel_size=k, padding=0), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(f2 * 7 * 7, cfg["CNN_FC"]), nn.ReLU(),
        )
        # Coord branch: (4,) → 64
        self.coord = nn.Sequential(
            nn.Linear(4, cfg["COORD_FC"]), nn.ReLU(),
        )
        # Shared layer: 128+64=192 → 256
        self.shared = nn.Sequential(
            nn.Linear(cfg["CNN_FC"] + cfg["COORD_FC"], cfg["DECISION_FC"]), nn.ReLU(),
        )
        dfc = cfg["DUELING_FC"]
        # Value stream → scalar V(S)
        self.value_stream = nn.Sequential(
            nn.Linear(cfg["DECISION_FC"], dfc), nn.ReLU(),
            nn.Linear(dfc, 1),
        )
        # Advantage stream → A(S,a) for 8 actions
        self.advantage_stream = nn.Sequential(
            nn.Linear(cfg["DECISION_FC"], dfc), nn.ReLU(),
            nn.Linear(dfc, cfg["N_ACTIONS"]),
        )

    def forward(self, window, coord):
        v_feat = self.vision(window)
        c_feat = self.coord(coord)
        shared = self.shared(torch.cat([v_feat, c_feat], dim=1))
        V = self.value_stream(shared)
        A = self.advantage_stream(shared)
        return V + (A - A.mean(dim=1, keepdim=True))   # Q = V + A - mean(A)

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
MODEL_PATH = Path("vishnu2_checkpoint.pth")
online_net = DuelingNavDQN(CFG).to(device)
MODEL_LOADED = False

if MODEL_PATH.exists():
    try:
        ckpt = torch.load(MODEL_PATH, map_location=device)
        online_net.load_state_dict(ckpt["online_state_dict"])
        MODEL_LOADED = True
        print(f"✅  Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️   Could not load checkpoint: {e}")
else:
    print("⚠️   vishnu_checkpoint.pth not found — using random weights.")

online_net.eval()

# ─────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────
def get_state(r, c, goal_r, goal_c):
    W = CFG["WINDOW"]
    window = PADDED_GRID[r:r+W, c:c+W].astype(np.float32)
    dx = (goal_r - r) / (CFG["ROWS"] - 1)
    dy = (goal_c - c) / (CFG["COLS"] - 1)
    coord = np.array([r/(CFG["ROWS"]-1), c/(CFG["COLS"]-1), dx, dy], dtype=np.float32)
    return window, coord

def greedy_episode(start_r, start_c, goal_r, goal_c, max_steps=6000):
    r, c         = start_r, start_c
    route        = [(r, c)]
    total_reward = 0.0
    bumps        = 0
    last_action  = None
    prev_dist    = math.sqrt((r-goal_r)**2 + (c-goal_c)**2)
    ep_visits    = np.zeros((CFG["ROWS"], CFG["COLS"]), dtype=np.int32)
    ep_visits[r, c] = 1

    with torch.no_grad():
        for step in range(max_steps):
            # Fuel-proportional step cost (for reward tracking only)
            fuel_frac  = (max_steps - step) / max_steps
            step_cost  = CFG["FUEL_COST_START"] + \
                         (CFG["FUEL_COST_END"] - CFG["FUEL_COST_START"]) * (1 - fuel_frac)
            total_reward += step_cost

            # Action-repeat penalty tracking
            window, coord = get_state(r, c, goal_r, goal_c)
            w_t = torch.tensor(window).unsqueeze(0).unsqueeze(0).to(device)
            c_t = torch.tensor(coord).unsqueeze(0).to(device)
            q   = online_net(w_t, c_t).squeeze(0).cpu().numpy()
            action = int(np.argmax(q))

            if last_action is not None and action == ACTION_OPPOSITE[last_action]:
                total_reward += CFG["R_REPEAT_PENALTY"]

            dr, dc = ACTION_DELTAS[action]
            nr, nc = r+dr, c+dc
            oob    = not (0 <= nr < CFG["ROWS"] and 0 <= nc < CFG["COLS"])
            land   = (not oob) and (GRID[nr, nc] == 1)

            if oob or land:
                total_reward += CFG["R_BUMP"]
                bumps        += 1
            else:
                r, c = nr, nc
                if ep_visits[r, c] > 0:
                    total_reward += CFG["R_VISIT_PENALTY"]
                ep_visits[r, c] += 1

                curr_dist = math.sqrt((r-goal_r)**2 + (c-goal_c)**2)
                delta     = prev_dist - curr_dist
                if delta > 0:
                    total_reward += CFG["R_CLOSER"] * delta
                else:
                    total_reward += CFG["R_FARTHER"] * abs(delta)
                prev_dist = curr_dist
                route.append((r, c))

            last_action = action

            if r == goal_r and c == goal_c:
                total_reward += CFG["R_GOAL"]
                break

    reached = (r == goal_r and c == goal_c)
    return route, len(route)-1, reached, total_reward, bumps

# ─────────────────────────────────────────────
#  FASTAPI
# ─────────────────────────────────────────────
app = FastAPI(
    title="Project Vishnu — Naval Navigation API",
    description="Dueling DDQN agent on 200×200 Indian Ocean Region grid",
    version="3.0.0"
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

class NavigateRequest(BaseModel):
    start_row: int
    start_col: int
    goal_row:  int
    goal_col:  int

class NavigateResponse(BaseModel):
    route:        List[List[int]]
    steps:        int
    reached_goal: bool
    total_reward: float
    bumps:        int
    model_loaded: bool

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "grid_shape": [200, 200],
        "water_cells": len(WATER_CELLS),
        "phase": "Project Vishnu — Phase 3",
    }

@app.get("/grid")
def get_grid():
    return {
        "grid":  GRID.tolist(),
        "ports": {k: list(v) for k, v in PORTS.items()},
        "rows":  200, "cols": 200,
    }

@app.post("/navigate", response_model=NavigateResponse)
def navigate(req: NavigateRequest):
    for val, name in [
        (req.start_row,"start_row"),(req.start_col,"start_col"),
        (req.goal_row, "goal_row"), (req.goal_col, "goal_col")
    ]:
        if not (0 <= val < 200):
            raise HTTPException(400, f"{name}={val} out of range [0,199]")
    if GRID[req.start_row, req.start_col] == 1:
        raise HTTPException(400, f"Start ({req.start_row},{req.start_col}) is land.")
    if GRID[req.goal_row, req.goal_col] == 1:
        raise HTTPException(400, f"Goal ({req.goal_row},{req.goal_col}) is land.")
    if req.start_row == req.goal_row and req.start_col == req.goal_col:
        raise HTTPException(400, "Start and goal must be different cells.")

    route, steps, reached, reward, bumps = greedy_episode(
        req.start_row, req.start_col, req.goal_row, req.goal_col
    )
    return NavigateResponse(
        route        = [[r, c] for r, c in route],
        steps        = steps,
        reached_goal = reached,
        total_reward = round(reward, 2),
        bumps        = bumps,
        model_loaded = MODEL_LOADED,
    )

@app.get("/")
def root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
