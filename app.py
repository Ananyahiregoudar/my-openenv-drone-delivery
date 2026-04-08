"""
app.py — Gradio web UI for the Drone Delivery Fleet OpenEnv environment.

Provides an interactive demo where users can:
  1. Select a task difficulty (Easy / Medium / Hard)
  2. Run the baseline agent and watch the score
  3. Step through the environment manually
  4. View the grid visualization

Deployed on HuggingFace Spaces via Docker.
"""
import sys
import os
import gradio as gr
import time
from drone_delivery_env.environment import DroneDeliveryEnvironment
from drone_delivery_env.models import WeatherCondition, DroneAction, StepAction, DroneDeliveryState
from drone_delivery_env.tasks.easy   import run_easy_task,   greedy_action
from drone_delivery_env.tasks.medium import run_medium_task, battery_aware_action
from drone_delivery_env.tasks.hard   import run_hard_task,   obstacle_aware_action

# ─────────────────────────────────────────────────────────
#  Grid Visualiser (text-based for maximum compatibility)
# ─────────────────────────────────────────────────────────

def render_grid(state) -> str:
    """Render the current environment state as an ASCII grid."""
    gs = state.grid_size
    # Build empty grid
    grid = [["·" for _ in range(gs)] for _ in range(gs)]

    # No-fly zones
    for z in state.no_fly_zones:
        for x in range(z.x_min, z.x_max + 1):
            for y in range(z.y_min, z.y_max + 1):
                if 0 <= x < gs and 0 <= y < gs:
                    grid[y][x] = "▓"

    # Recharge stations
    for s in state.recharge_stations:
        if 0 <= s.x < gs and 0 <= s.y < gs:
            grid[s.y][s.x] = "⚡"

    # Parcels (undelivered)
    for p in state.parcels:
        if not p.delivered and 0 <= p.x < gs and 0 <= p.y < gs:
            grid[p.y][p.x] = "★" if p.priority else "📦"

    # Drones
    for i, d in enumerate(state.drones):
        if 0 <= d.x < gs and 0 <= d.y < gs:
            grid[d.y][d.x] = f"D{i}"

    # Render (flip Y so north is up)
    lines = []
    lines.append("   " + " ".join(f"{x:>2}" for x in range(gs)))
    lines.append("   " + "---" * gs)
    for y in range(gs - 1, -1, -1):
        row = " ".join(f"{grid[y][x]:>2}" for x in range(gs))
        lines.append(f"{y:>2}| {row}")
    lines.append("")
    lines.append("Legend: Dn=Drone  ★=Priority  📦=Parcel  ⚡=Station  ▓=No-Fly  ·=Empty")
    return "\n".join(lines)


def format_status(state) -> str:
    """Format drone/parcel status as readable text."""
    lines = []
    lines.append(f"Step: {state.step_count}/{state.max_steps}  |  "
                 f"Weather: {state.weather.value.upper()}  |  "
                 f"Grid: {state.grid_size}×{state.grid_size}")
    lines.append("")

    delivered = sum(1 for p in state.parcels if p.delivered)
    total = len(state.parcels)
    lines.append(f"Parcels: {delivered}/{total} delivered")

    for d in state.drones:
        lines.append(f"  {d.id}: pos=({d.x},{d.y})  battery={d.battery:.0f}%")

    undelivered = [p for p in state.parcels if not p.delivered]
    if undelivered:
        lines.append("")
        lines.append("Remaining parcels:")
        for p in undelivered:
            tag = " [PRIORITY]" if p.priority else ""
            lines.append(f"  {p.id}: ({p.x},{p.y}){tag}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
#  Task runner for the Gradio interface
# ─────────────────────────────────────────────────────────

TASK_CONFIG = {
    "Easy": {
        "grid_size": 10, "num_drones": 3, "num_parcels": 5,
        "num_no_fly_zones": 0, "max_steps": 100,
        "weather": WeatherCondition.CLEAR,
        "description": "3 drones, 5 parcels, clear weather, no obstacles",
    },
    "Medium": {
        "grid_size": 10, "num_drones": 3, "num_parcels": 7,
        "num_no_fly_zones": 1, "max_steps": 150,
        "weather": WeatherCondition.WINDY,
        "description": "3 drones, 7 parcels, windy weather (2× battery drain), 1 no-fly zone",
    },
    "Hard": {
        "grid_size": 10, "num_drones": 2, "num_parcels": 8,
        "num_no_fly_zones": 2, "max_steps": 500,
        "weather": WeatherCondition.STORMY,
        "description": "2 drones, 8 parcels, stormy weather (4× drain), 2 no-fly zones, priority parcels",
    },
}


def run_baseline(difficulty: str, seed: int):
    """Run baseline agent on selected difficulty, return results."""
    import tasks.medium as med_mod
    import tasks.hard   as hard_mod

    # Reset module-level coordination globals
    med_mod._turn_index = 0
    med_mod._assigned.clear()
    hard_mod._turn_index = 0
    hard_mod._assigned.clear()

    cfg = TASK_CONFIG[difficulty]
    env = DroneDeliveryEnvironment(
        grid_size=cfg["grid_size"],
        num_drones=cfg["num_drones"],
        num_parcels=cfg["num_parcels"],
        num_no_fly_zones=cfg["num_no_fly_zones"],
        max_steps=cfg["max_steps"],
        weather=cfg["weather"],
        seed=seed,
    )

    state = env.reset()

    # Mark priority parcels for hard task
    if difficulty == "Hard":
        for i in range(min(2, len(state.parcels))):
            state.parcels[i].priority = True

    # Select agent
    agents = {
        "Easy":   greedy_action,
        "Medium": battery_aware_action,
        "Hard":   obstacle_aware_action,
    }
    agent = agents[difficulty]

    # Run simulation
    done = False
    step_log = []
    while not done:
        action = agent(state)
        result = env.step(action)
        state  = result.state
        done   = result.done

        if result.info.get("delivered"):
            step_log.append(f"  Step {state.step_count}: {result.info['delivered']} delivered! (+{result.reward:.1f})")
        elif result.info.get("recharged"):
            step_log.append(f"  Step {state.step_count}: {result.info['recharged']} recharged")
        elif result.info.get("error"):
            step_log.append(f"  Step {state.step_count}: ERROR — {result.info['error']}")

    score = env.score()
    delivered = sum(1 for p in state.parcels if p.delivered)
    total = len(state.parcels)

    # Build result text
    result_text = f"Score: {score:.4f}\n"
    result_text += f"Parcels: {delivered}/{total} delivered\n"
    result_text += f"Steps: {state.step_count}/{state.max_steps}\n\n"
    result_text += "Event Log:\n"
    result_text += "\n".join(step_log[-30:]) if step_log else "  (no notable events)"

    grid_text   = render_grid(state)
    status_text = format_status(state)

    return result_text, grid_text, status_text


# ─────────────────────────────────────────────────────────
#  Gradio Interface
# ─────────────────────────────────────────────────────────

def build_app():
    with gr.Blocks(
        title="Drone Delivery Fleet — OpenEnv",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="cyan",
        ),
    ) as demo:
        gr.Markdown("""
# 🚁 Drone Delivery Fleet — OpenEnv Environment

An AI agent routes a fleet of drones for last-mile delivery, managing **battery life**,
**weather windows**, **no-fly zones**, and **priority parcels** simultaneously.

Select a difficulty level and click **Run Baseline** to see the built-in agent in action.
        """)

        with gr.Row():
            difficulty = gr.Dropdown(
                choices=["Easy", "Medium", "Hard"],
                value="Easy",
                label="Task Difficulty",
                info="Easy: no constraints | Medium: battery management | Hard: storm + no-fly zones",
            )
            seed = gr.Number(
                value=42,
                label="Seed",
                info="Random seed for reproducibility",
                precision=0,
            )
            run_btn = gr.Button("▶ Run Baseline", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=1):
                result_box = gr.Textbox(
                    label="📊 Results & Event Log",
                    lines=18,
                    interactive=False,
                )
            with gr.Column(scale=1):
                grid_box = gr.Textbox(
                    label="🗺️ Final Grid State",
                    lines=18,
                    interactive=False,
                )

        status_box = gr.Textbox(
            label="📋 Environment Status",
            lines=6,
            interactive=False,
        )

        gr.Markdown("""
---
### 📖 About This Environment

| Aspect | Details |
|--------|---------|
| **Action Space** | `move_north`, `move_south`, `move_east`, `move_west`, `deliver`, `recharge`, `wait` |
| **Observation** | Full state: drone positions/battery, parcel locations, no-fly zones, weather |
| **Reward** | +1.0 per delivery (+1.5 priority), −0.01 step cost, +2.0 completion bonus |
| **Weather Effects** | Clear: 1× drain, Windy: 2×, Stormy: 4× |
| **API** | `env.reset()` → state, `env.step(action)` → (state, reward, done, info) |

Built for the OpenEnv Hackathon • [View Source](https://github.com)
        """)

        run_btn.click(
            fn=run_baseline,
            inputs=[difficulty, seed],
            outputs=[result_box, grid_box, status_box],
        )

    return demo


if __name__ == "__main__":
    import uvicorn
    from openenv.core.env_server.http_server import create_app
    from drone_delivery_env.environment import DroneDeliveryEnvironment
    from drone_delivery_env.models import StepAction
    
    # 1. Create standard OpenEnv API server
    fastapi_app = create_app(
        DroneDeliveryEnvironment,
        StepAction,
        DroneDeliveryState,  # observation type
        env_name="drone-delivery-fleet",
        max_concurrent_envs=1,
    )
    
    # 2. Build Gradio UI
    demo = build_app()
    
    # 3. Mount Gradio at root (OpenEnv specific routes like /reset remain preserved)
    app = gr.mount_gradio_app(fastapi_app, demo, path="/")
    
    # Run the unified server
    uvicorn.run(app, host="0.0.0.0", port=7860)
