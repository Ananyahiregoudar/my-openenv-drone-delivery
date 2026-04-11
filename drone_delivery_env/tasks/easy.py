"""
Easy Task — 3 drones, 5 parcels, clear weather, no no-fly zones.
Agent just needs to deliver all parcels with full resources.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone_delivery_env.environment import DroneDeliveryEnvironment
from drone_delivery_env.models import WeatherCondition, StepAction, DroneAction


TASK_DESCRIPTION = """
Easy Task: Deliver all parcels in clear weather.
- 3 drones, 5 parcels on a 10x10 grid
- No no-fly zones, full battery, clear weather
- Score = parcels delivered / total parcels (+ efficiency bonus)
"""


def run_easy_task(agent_fn=None, seed: int = 42) -> float:
    """
    Run the easy task with an optional agent function.
    agent_fn(state) -> StepAction
    If no agent provided, uses a simple greedy baseline.
    Returns score between 0.0 and 1.0.
    """
    env = DroneDeliveryEnvironment(
        grid_size=10,
        num_drones=3,
        num_parcels=5,
        num_no_fly_zones=0,       # no obstacles
        max_steps=100,
        weather=WeatherCondition.CLEAR,
        seed=seed,
    )

    state = env.reset()
    done  = False

    while not done:
        if agent_fn is not None:
            action = agent_fn(state)
        else:
            action = greedy_action(state)

        result = env.step(action)
        state  = result.state
        done   = result.done

    return env.score()


def greedy_action(state) -> StepAction:
    """
    Baseline greedy agent:
    For each drone (first one with battery), move toward
    the nearest undelivered parcel and deliver it.
    """
    for drone in state.drones:
        if drone.battery <= 0:
            continue

        # find nearest undelivered parcel
        undelivered = [p for p in state.parcels if not p.delivered]
        if not undelivered:
            return StepAction(drone_id=drone.id, action=DroneAction.WAIT)

        target = min(undelivered, key=lambda p: abs(p.x - drone.x) + abs(p.y - drone.y))

        # if on parcel, deliver
        if drone.x == target.x and drone.y == target.y:
            return StepAction(drone_id=drone.id, action=DroneAction.DELIVER)

        # move toward parcel
        if drone.x < target.x:
            return StepAction(drone_id=drone.id, action=DroneAction.MOVE_EAST)
        elif drone.x > target.x:
            return StepAction(drone_id=drone.id, action=DroneAction.MOVE_WEST)
        elif drone.y < target.y:
            return StepAction(drone_id=drone.id, action=DroneAction.MOVE_NORTH)
        else:
            return StepAction(drone_id=drone.id, action=DroneAction.MOVE_SOUTH)

    return StepAction(drone_id=state.drones[0].id, action=DroneAction.WAIT)


if __name__ == "__main__":
    score = run_easy_task()
    print(f"Easy task score: {score:.4f}")

def grader(trajectory: dict = None) -> float:
    """
    Grader for easy drone delivery tasks.

    Accepts a trajectory dict with a 'rewards' key (list of per-step rewards).
    Returns a score strictly in (0.01, 0.99).
    """
    trajectory = trajectory or {}
    rewards = trajectory.get("rewards", [])

    if not rewards:
        # No trajectory data — run the task with the built-in baseline agent
        try:
            score = run_easy_task()
        except Exception:
            score = 0.5
        return min(max(round(score, 4), 0.01), 0.99)

    # Compute score from trajectory rewards
    n = len(rewards)
    total_reward = sum(rewards)

    # Normalise: positive rewards indicate deliveries (+1.0/+1.5 each)
    # Negative rewards indicate penalties.  Max possible ≈ 5 parcels * 1.5 + 2.0 bonus
    positive = sum(r for r in rewards if r > 0)
    negative = sum(abs(r) for r in rewards if r < 0)

    # Delivery ratio: how much positive reward vs theoretical max (~9.5 for easy)
    max_possible = 5 * 1.0 + 2.0  # 5 parcels at 1.0 each + completion bonus
    delivery_ratio = min(positive / max(max_possible, 0.01), 1.0)

    # Penalty factor: penalise excessive mistakes
    penalty_factor = min(negative * 0.05, 0.4)

    # Efficiency: fewer steps is better (reward fewer total steps)
    efficiency_bonus = max(0.0, 0.05 * (1.0 - n / 100.0)) if n < 100 else 0.0

    score = delivery_ratio * 0.85 - penalty_factor + efficiency_bonus

    return min(max(round(score, 4), 0.01), 0.99)

