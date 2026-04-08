# 🚁 Drone Delivery Fleet — OpenEnv Environment

> **An AI agent routes a fleet of drones for last-mile delivery — managing battery life, weather windows, no-fly zones, and priority parcels simultaneously.**

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌍 Environment Overview

A configurable grid-world where **multiple drones** must coordinate to deliver parcels across a city. The agent must reason about:

- **Fleet coordination** — Assign parcels to drones efficiently, avoid redundant work
- **Battery management** — Drones have limited battery; recharge at stations or strand
- **Weather effects** — Clear (1× drain), Windy (2×), Stormy (4×)
- **No-fly zones** — Restricted airspace the drones must route around
- **Priority parcels** — Some deliveries are worth more and should be prioritised

---

## 🎯 Tasks

| Task | Difficulty | Drones | Parcels | Weather | No-Fly Zones | Max Steps | Baseline Score |
|------|-----------|--------|---------|---------|-------------|-----------|---------------|
| **Easy** | ⭐ | 3 | 5 | Clear | 0 | 100 | 1.0000 |
| **Medium** | ⭐⭐ | 3 | 7 | Windy (2× drain) | 1 | 150 | 1.0000 |
| **Hard** | ⭐⭐⭐ | 2 | 8 | Stormy (4× drain) | 2 | 500 | 1.0000 |

### Easy — No-Constraint Delivery
Deliver 5 parcels with 3 drones in clear weather. No obstacles, full battery. Tests basic pathfinding and parcel assignment.

### Medium — Battery Management
Deliver 7 parcels in windy weather (2× battery drain) with 1 no-fly zone. Drones must strategically recharge at stations to complete all deliveries.

### Hard — Storm + No-Fly + Priority
Deliver 8 parcels (2 priority) with only 2 drones in a storm (4× battery drain), navigating around 2 no-fly zones. Requires BFS pathfinding, multi-drone coordination, and smart battery management.

---

## 🔧 API Reference

### `env.reset() → DroneDeliveryState`
Initialise the environment and return the starting state.

### `env.step(action: StepAction) → StepResult`
Execute one action. Returns `StepResult(state, reward, done, info)`.

### `env.get_state() → DroneDeliveryState`
Return the current state without advancing the simulation.

### `env.score() → float`
Return a normalised score (0.0–1.0) for grading.

---

## 📋 Observation Space

The state is a fully observable `DroneDeliveryState`:

```python
DroneDeliveryState(
    drones=[                       # List of Drone objects
        Drone(id, x, y, battery, carrying_parcel)
    ],
    parcels=[                      # List of Parcel objects
        Parcel(id, x, y, priority, delivered)
    ],
    no_fly_zones=[                 # Restricted areas
        NoFlyZone(x_min, x_max, y_min, y_max)
    ],
    recharge_stations=[            # Battery recharge points
        RechargeStation(x, y)
    ],
    weather="clear"|"windy"|"stormy",
    step_count=0,
    max_steps=100,
    grid_size=10,
)
```

## 🎮 Action Space

Each step commands **one drone** with a `StepAction(drone_id, action)`:

| Action | Effect | Battery Cost |
|--------|--------|-------------|
| `move_north` | y += 1 | 1× weather multiplier |
| `move_south` | y -= 1 | 1× weather multiplier |
| `move_east`  | x += 1 | 1× weather multiplier |
| `move_west`  | x -= 1 | 1× weather multiplier |
| `deliver`    | Deliver parcel at current position | 0 |
| `recharge`   | Fully recharge at a station | 0 |
| `wait`       | Do nothing | 0 |

## 💰 Reward Structure

| Event | Reward |
|-------|--------|
| Move (any direction) | −0.01 |
| Wait | −0.02 |
| Deliver parcel | +1.0 |
| Deliver priority parcel | +1.5 |
| Recharge at station | +0.1 |
| All parcels delivered | +2.0 bonus |
| Move out of bounds | −0.1 |
| Enter no-fly zone | −0.3 |
| Move with dead battery | −0.5 |
| Invalid deliver/recharge | −0.05 |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-repo/drone-delivery-env.git
cd drone-delivery-env
pip install -r requirements.txt
```

### Run Baseline Inference

```bash
# Run all tasks
python inference.py

# Run a specific task
python inference.py --task hard --seed 42
```

**Expected output:**
```
============================================================
  Drone Delivery Fleet — Baseline Inference
============================================================
  Seed: 42

  Task        Score   Time (s)     Status
  ---------- -------- ---------- ----------
  easy         1.0000      0.010   PASS ✓
  medium       1.0000      0.025   PASS ✓
  hard         1.0000      1.200   PASS ✓

  Average score: 1.0000
============================================================
```

### Run Individual Tasks

```bash
cd tasks
python easy.py      # Easy task
python medium.py    # Medium task
python hard.py      # Hard task
```

### Launch Gradio Demo

```bash
python app.py
# Open http://localhost:7860
```

---

## 🐳 Docker / HuggingFace Spaces

```bash
docker build -t drone-delivery-env .
docker run -p 7860:7860 drone-delivery-env
```

---

## 🏗️ Project Structure

```
drone-delivery-env/
├── app.py               # Gradio web UI for HF Spaces
├── environment.py       # Core environment (step/reset/state)
├── models.py            # Pydantic data models
├── inference.py         # Baseline inference script
├── openenv.yaml         # OpenEnv specification
├── Dockerfile           # HF Spaces deployment
├── requirements.txt     # Python dependencies
├── debug.py             # Debug/inspection script
├── README.md            # This file
└── tasks/
    ├── __init__.py
    ├── easy.py          # Easy task + greedy agent
    ├── medium.py        # Medium task + battery-aware agent
    └── hard.py          # Hard task + BFS pathfinding agent
```

## 🧪 Writing Your Own Agent

```python
from environment import DroneDeliveryEnvironment
from models import WeatherCondition, StepAction, DroneAction

env = DroneDeliveryEnvironment(
    grid_size=10, num_drones=2, num_parcels=5,
    max_steps=100, weather=WeatherCondition.CLEAR, seed=42
)

state = env.reset()
done = False

while not done:
    # Your agent logic here
    action = StepAction(drone_id="drone_0", action=DroneAction.MOVE_EAST)
    result = env.step(action)
    state  = result.state
    done   = result.done
    # result.reward gives per-step reward
    # result.info gives event details

print(f"Score: {env.score():.4f}")
```

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
