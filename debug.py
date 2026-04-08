import sys
sys.path.insert(0, '.')
from environment import DroneDeliveryEnvironment
from models import WeatherCondition, StepAction, DroneAction

env = DroneDeliveryEnvironment(
    grid_size=10, num_drones=2, num_parcels=8,
    num_no_fly_zones=2, max_steps=200,
    weather=WeatherCondition.STORMY, seed=42
)
state = env.reset()

print("=== INITIAL STATE ===")
for d in state.drones:
    print(f"  Drone {d.id}: pos=({d.x},{d.y}) battery={d.battery}")
for p in state.parcels:
    print(f"  Parcel {p.id}: pos=({p.x},{p.y}) priority={p.priority} delivered={p.delivered}")
for z in state.no_fly_zones:
    print(f"  NoFly: x={z.x_min}-{z.x_max} y={z.y_min}-{z.y_max}")

# mark priority
for i in range(min(2, len(state.parcels))):
    state.parcels[i].priority = True
print("\nAfter marking priority:")
for p in state.parcels:
    print(f"  Parcel {p.id}: priority={p.priority}")

# take one step
result = env.step(StepAction(drone_id='drone_0', action=DroneAction.MOVE_EAST))
print(f"\nAfter 1 step: reward={result.reward} done={result.done}")
print(f"Drone 0 new pos: ({result.state.drones[0].x},{result.state.drones[0].y})")
print(f"Delivered: {[p.delivered for p in result.state.parcels]}")