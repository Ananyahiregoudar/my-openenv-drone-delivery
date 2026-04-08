"""
Hard Task — 2 drones, 8 parcels, stormy weather, multiple no-fly zones, priority parcels.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import DroneDeliveryEnvironment
from models import WeatherCondition, StepAction, DroneAction


TASK_DESCRIPTION = """
Hard Task: Deliver priority parcels in a storm with no-fly zones.
- 2 drones, 8 parcels on a 10x10 grid
- Stormy weather drains battery 4x faster
- 2 no-fly zones blocking paths
- 2 priority parcels worth extra reward
- Score = weighted delivery score with priority multiplier
"""


def run_hard_task(agent_fn=None, seed: int = 42) -> float:
    env = DroneDeliveryEnvironment(
        grid_size=10,
        num_drones=2,
        num_parcels=8,
        num_no_fly_zones=2,
        max_steps=500,
        weather=WeatherCondition.STORMY,
        seed=seed,
    )

    state = env.reset()

    # mark first 2 parcels as priority
    for i in range(min(2, len(state.parcels))):
        state.parcels[i].priority = True

    done = False
    while not done:
        if agent_fn is not None:
            action = agent_fn(state)
        else:
            action = obstacle_aware_action(state)

        result = env.step(action)
        state  = result.state
        done   = result.done

    return compute_hard_score(state)


def compute_hard_score(state) -> float:
    total_weight     = 0.0
    delivered_weight = 0.0
    for parcel in state.parcels:
        weight = 2.0 if parcel.priority else 1.0
        total_weight += weight
        if parcel.delivered:
            delivered_weight += weight
    if total_weight == 0:
        return 0.0
    base = delivered_weight / total_weight
    if all(p.delivered for p in state.parcels):
        efficiency = 1.0 - (state.step_count / state.max_steps)
        base += 0.15 * efficiency
    return round(min(base, 1.0), 4)


# ---- module-level state for multi-drone coordination ----
_turn_index = 0          # round-robin counter so both drones act
_assigned: dict = {}     # drone_id -> parcel_id currently being targeted


def obstacle_aware_action(state) -> StepAction:
    global _turn_index, _assigned
    from collections import deque

    gs = state.grid_size

    def in_no_fly(x, y):
        for zone in state.no_fly_zones:
            if zone.x_min <= x <= zone.x_max and zone.y_min <= y <= zone.y_max:
                return True
        return False

    def manhattan(ax, ay, bx, by):
        return abs(ax - bx) + abs(ay - by)

    # BFS to find the first move toward (tx, ty) avoiding no-fly zones.
    # Returns the DroneAction for the first step, or WAIT if unreachable.
    def bfs_first_move(sx, sy, tx, ty):
        if sx == tx and sy == ty:
            return DroneAction.WAIT
        moves = [
            (DroneAction.MOVE_NORTH, 0,  1),
            (DroneAction.MOVE_SOUTH, 0, -1),
            (DroneAction.MOVE_EAST,  1,  0),
            (DroneAction.MOVE_WEST, -1,  0),
        ]
        visited = set()
        visited.add((sx, sy))
        # queue entries: (x, y, first_action)
        queue = deque()
        for action, dx, dy in moves:
            nx, ny = sx + dx, sy + dy
            if 0 <= nx < gs and 0 <= ny < gs and not in_no_fly(nx, ny):
                if nx == tx and ny == ty:
                    return action
                visited.add((nx, ny))
                queue.append((nx, ny, action))
        while queue:
            cx, cy, first_action = queue.popleft()
            for _, dx, dy in moves:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited and 0 <= nx < gs and 0 <= ny < gs and not in_no_fly(nx, ny):
                    if nx == tx and ny == ty:
                        return first_action
                    visited.add((nx, ny))
                    queue.append((nx, ny, first_action))
        return DroneAction.WAIT  # unreachable

    # BFS shortest path length (used for battery estimation)
    def bfs_distance(sx, sy, tx, ty):
        if sx == tx and sy == ty:
            return 0
        visited = {(sx, sy)}
        queue = deque([(sx, sy, 0)])
        while queue:
            cx, cy, d = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited and 0 <= nx < gs and 0 <= ny < gs and not in_no_fly(nx, ny):
                    if nx == tx and ny == ty:
                        return d + 1
                    visited.add((nx, ny))
                    queue.append((nx, ny, d + 1))
        return 9999  # unreachable

    # --- pick which drone acts this step (round-robin among alive drones) ---
    alive_drones = [d for d in state.drones if d.battery > 0]
    if not alive_drones:
        return StepAction(drone_id=state.drones[0].id, action=DroneAction.WAIT)

    drone = alive_drones[_turn_index % len(alive_drones)]
    _turn_index += 1

    # --- clean up assignments for already-delivered parcels ---
    delivered_ids = {p.id for p in state.parcels if p.delivered}
    for did in list(_assigned):
        if _assigned[did] in delivered_ids:
            del _assigned[did]

    # --- battery management (use BFS distance for accuracy around NFZs) ---
    drain = 4.0  # stormy
    nearest_station = min(
        state.recharge_stations,
        key=lambda s: bfs_distance(drone.x, drone.y, s.x, s.y)
    )
    station_path_dist = bfs_distance(drone.x, drone.y, nearest_station.x, nearest_station.y)
    battery_needed = (station_path_dist + 2) * drain  # +2 safety buffer
    should_recharge = drone.battery <= battery_needed

    if should_recharge:
        if drone.x == nearest_station.x and drone.y == nearest_station.y:
            return StepAction(drone_id=drone.id, action=DroneAction.RECHARGE)
        move = bfs_first_move(drone.x, drone.y, nearest_station.x, nearest_station.y)
        return StepAction(drone_id=drone.id, action=move)

    # --- parcel targeting with coordination ---
    undelivered = [p for p in state.parcels if not p.delivered]
    if not undelivered:
        return StepAction(drone_id=drone.id, action=DroneAction.WAIT)

    # if this drone already has an assigned parcel that's still undelivered, keep it
    current_target = None
    if drone.id in _assigned:
        current_target = next(
            (p for p in undelivered if p.id == _assigned[drone.id]), None
        )

    if current_target is None:
        # assign a new parcel — prefer one not already assigned to another drone
        other_assigned = {v for k, v in _assigned.items() if k != drone.id}
        unassigned = [p for p in undelivered if p.id not in other_assigned]
        candidates = unassigned if unassigned else undelivered
        # priority first, then nearest by BFS distance
        candidates.sort(key=lambda p: (
            not p.priority,
            bfs_distance(drone.x, drone.y, p.x, p.y)
        ))
        current_target = candidates[0]
        _assigned[drone.id] = current_target.id

    # --- act on target ---
    if drone.x == current_target.x and drone.y == current_target.y:
        if drone.id in _assigned:
            del _assigned[drone.id]
        return StepAction(drone_id=drone.id, action=DroneAction.DELIVER)

    move = bfs_first_move(drone.x, drone.y, current_target.x, current_target.y)
    return StepAction(drone_id=drone.id, action=move)


if __name__ == "__main__":
    score = run_hard_task()
    print(f"Hard task score: {score:.4f}")

    # debug info — reset global coordination state for a fresh run
    _turn_index = 0
    _assigned.clear()
    env = DroneDeliveryEnvironment(
        grid_size=10, num_drones=2, num_parcels=8,
        num_no_fly_zones=2, max_steps=500,
        weather=WeatherCondition.STORMY, seed=42
    )
    state = env.reset()
    for i in range(min(2, len(state.parcels))):
        state.parcels[i].priority = True
    done = False
    while not done:
        action = obstacle_aware_action(state)
        result = env.step(action)
        state = result.state
        done = result.done
    delivered = sum(1 for p in state.parcels if p.delivered)
    print(f"Parcels delivered: {delivered}/{len(state.parcels)}")
    print(f"Steps used: {state.step_count}/{state.max_steps}")
    for d in state.drones:
        print(f"  {d.id} final battery: {d.battery:.1f}")