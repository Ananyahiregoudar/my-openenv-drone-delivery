import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from environment import DroneDeliveryEnvironment
from models import WeatherCondition, StepAction, DroneAction

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy-key"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASKS = {
    "easy": {
        "grid_size": 10, "num_drones": 3, "num_parcels": 5,
        "num_no_fly_zones": 0, "max_steps": 100, "weather": WeatherCondition.CLEAR
    },
    "medium": {
        "grid_size": 10, "num_drones": 3, "num_parcels": 7,
        "num_no_fly_zones": 1, "max_steps": 150, "weather": WeatherCondition.WINDY
    },
    "hard": {
        "grid_size": 10, "num_drones": 2, "num_parcels": 8,
        "num_no_fly_zones": 2, "max_steps": 500, "weather": WeatherCondition.STORMY
    }
}

TEMPERATURE = 0.7
MAX_TOKENS = 50
SUCCESS_SCORE_THRESHOLD = 0.5  # normalized score in [0, 1]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI agent routing a fleet of drones for last-mile delivery.
    Your goal is to maximize total reward by delivering parcels.
    Available actions for a single drone: move_north, move_south, move_east, move_west, deliver, recharge, wait.
    Only one drone acts per step.
    Weather: clear(1x battery drain), windy(2x), stormy(4x). Battery usage per move step is the weather multiplier.
    Reply with EXACTLY one action in the format: <drone_id>:<action>
    Example: drone_0:move_east
    Do not add extra text, JSON, or quotes.
    """
).strip()


def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, state_str: str, last_action: str, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Current State:
        {state_str}

        Last action taken: {last_action}
        Last reward: {last_reward:.2f}
        Recent history:
        {history_block}
        
        Send your next action for one drone in format "drone_id:action".
        """
    ).strip()


def get_model_action(client: OpenAI, step: int, state_str: str, last_action: str, last_reward: float, history: List[str]) -> StepAction:
    user_prompt = build_user_prompt(step, state_str, last_action, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        parts = text.split(':')
        if len(parts) == 2:
            return StepAction(drone_id=parts[0].strip(), action=DroneAction(parts[1].strip()))
    except Exception as exc:
        print(f"[DEBUG] Model request failed or incorrectly formatted: {exc}", flush=True)
    
    # Fallback default action
    return StepAction(drone_id="drone_0", action=DroneAction.WAIT)

def format_state(state):
    import json
    return json.dumps({
        "drones": [d.dict() for d in state.drones],
        "parcels": [p.dict() for p in state.parcels],
        "no_fly_zones": [z.dict() for z in state.no_fly_zones],
        "recharge_stations": [s.dict() for s in state.recharge_stations],
        "weather": state.weather.value
    })


def run_episode(client: OpenAI, task_name: str) -> None:
    task_cfg = TASKS[task_name]
    env = DroneDeliveryEnvironment(
        grid_size=task_cfg["grid_size"],
        num_drones=task_cfg["num_drones"],
        num_parcels=task_cfg["num_parcels"],
        num_no_fly_zones=task_cfg["num_no_fly_zones"],
        max_steps=task_cfg["max_steps"],
        weather=task_cfg["weather"],
        seed=42,
    )
    
    state = env.reset()
    if task_name == "hard":
        for i in range(min(2, len(state.parcels))):
            state.parcels[i].priority = True
            
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env_name="drone-delivery-fleet", model=MODEL_NAME)

    last_action_str = "None"
    last_reward = 0.0
    
    try:
        for step in range(1, env.max_steps + 1):
            state_str = format_state(state)
            action_obj = get_model_action(client, step, state_str, last_action_str, last_reward, history)
            action_str = f"{action_obj.drone_id}:{action_obj.action.value}"
            
            result = env.step(action_obj)
            state = result.state
            
            reward = result.reward or 0.0
            done = result.done
            error = result.info.get("error", None)

            rewards.append(reward)
            steps_taken = step
            last_action_str = action_str
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            if done:
                break

        score = env.score()
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as e:
        print(f"[DEBUG] Episode execution error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    for task_name in ["easy", "medium", "hard"]:
        run_episode(client, task_name)


if __name__ == "__main__":
    main()
