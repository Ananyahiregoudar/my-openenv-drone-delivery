import random
from typing import List, Optional
from .models import (
    DroneDeliveryState, Drone, Parcel, NoFlyZone,
    RechargeStation, WeatherCondition, StepAction,
    DroneAction, StepResult
)


class DroneDeliveryEnvironment:
    def __init__(
        self,
        grid_size: int = 10,
        num_drones: int = 3,
        num_parcels: int = 5,
        num_no_fly_zones: int = 2,
        max_steps: int = 100,
        weather: WeatherCondition = WeatherCondition.CLEAR,
        seed: Optional[int] = None,
    ):
        self.grid_size       = grid_size
        self.num_drones      = num_drones
        self.num_parcels     = num_parcels
        self.num_no_fly_zones = num_no_fly_zones
        self.max_steps       = max_steps
        self.weather         = weather
        self.seed            = seed
        self.state: Optional[DroneDeliveryState] = None

    # ------------------------------------------------------------------ #
    #  reset()                                                             #
    # ------------------------------------------------------------------ #
    def reset(self) -> DroneDeliveryState:
        if self.seed is not None:
            random.seed(self.seed)

        recharge_stations = [
            RechargeStation(x=0, y=0),
            RechargeStation(x=self.grid_size - 1, y=self.grid_size - 1),
        ]

        no_fly_zones = []
        for i in range(self.num_no_fly_zones):
            x1 = random.randint(1, self.grid_size - 3)
            y1 = random.randint(1, self.grid_size - 3)
            no_fly_zones.append(NoFlyZone(
                x_min=x1, x_max=x1 + 1,
                y_min=y1, y_max=y1 + 1
            ))

        drones = []
        for i in range(self.num_drones):
            drones.append(Drone(
                id=f"drone_{i}",
                x=random.randint(0, self.grid_size - 1),
                y=random.randint(0, self.grid_size - 1),
                battery=100.0,
                carrying_parcel=None
            ))

        parcels = []
        for i in range(self.num_parcels):
            parcels.append(Parcel(
                id=f"parcel_{i}",
                x=random.randint(0, self.grid_size - 1),
                y=random.randint(0, self.grid_size - 1),
                priority=(i == 0),   # first parcel is always priority
                delivered=False
            ))

        self.state = DroneDeliveryState(
            drones=drones,
            parcels=parcels,
            no_fly_zones=no_fly_zones,
            recharge_stations=recharge_stations,
            weather=self.weather,
            step_count=0,
            max_steps=self.max_steps,
            grid_size=self.grid_size,
        )
        return self.state

    # ------------------------------------------------------------------ #
    #  state()                                                             #
    # ------------------------------------------------------------------ #
    def get_state(self) -> DroneDeliveryState:
        if self.state is None:
            raise ValueError("Environment not initialised. Call reset() first.")
        return self.state

    # ------------------------------------------------------------------ #
    #  helpers                                                             #
    # ------------------------------------------------------------------ #
    def _in_no_fly_zone(self, x: int, y: int) -> bool:
        for zone in self.state.no_fly_zones:
            if zone.x_min <= x <= zone.x_max and zone.y_min <= y <= zone.y_max:
                return True
        return False

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.state.grid_size and 0 <= y < self.state.grid_size

    def _at_recharge_station(self, x: int, y: int) -> bool:
        for station in self.state.recharge_stations:
            if station.x == x and station.y == y:
                return True
        return False

    def _battery_drain(self) -> float:
        drains = {
            WeatherCondition.CLEAR:  1.0,
            WeatherCondition.WINDY:  2.0,
            WeatherCondition.STORMY: 4.0,
        }
        return drains[self.state.weather]

    # ------------------------------------------------------------------ #
    #  step()                                                              #
    # ------------------------------------------------------------------ #
    def step(self, action: StepAction) -> StepResult:
        if self.state is None:
            raise ValueError("Environment not initialised. Call reset() first.")

        # find the drone
        drone = next((d for d in self.state.drones if d.id == action.drone_id), None)
        if drone is None:
            raise ValueError(f"Drone {action.drone_id} not found.")

        reward = 0.0
        info   = {}

        # ----- movement actions ----------------------------------------
        new_x, new_y = drone.x, drone.y

        if action.action == DroneAction.MOVE_NORTH:
            new_y += 1
        elif action.action == DroneAction.MOVE_SOUTH:
            new_y -= 1
        elif action.action == DroneAction.MOVE_EAST:
            new_x += 1
        elif action.action == DroneAction.MOVE_WEST:
            new_x -= 1

        if action.action in (
            DroneAction.MOVE_NORTH, DroneAction.MOVE_SOUTH,
            DroneAction.MOVE_EAST,  DroneAction.MOVE_WEST
        ):
            drain = self._battery_drain()

            if drone.battery <= 0:
                reward -= 0.5
                info["error"] = "drone has no battery"

            elif not self._in_bounds(new_x, new_y):
                reward -= 0.1
                info["error"] = "out of bounds"

            elif self._in_no_fly_zone(new_x, new_y):
                reward -= 0.3
                info["error"] = "no-fly zone violation"

            else:
                drone.x = new_x
                drone.y = new_y
                drone.battery = max(0.0, drone.battery - drain)
                reward -= 0.01       # small step cost

        # ----- deliver action ------------------------------------------
        elif action.action == DroneAction.DELIVER:
            delivered_any = False
            for parcel in self.state.parcels:
                if not parcel.delivered and parcel.x == drone.x and parcel.y == drone.y:
                    parcel.delivered = True
                    delivered_any    = True
                    if parcel.priority:
                        reward += 1.5    # bonus for priority parcel
                    else:
                        reward += 1.0
                    info["delivered"] = parcel.id
                    break

            if not delivered_any:
                reward -= 0.05
                info["error"] = "no parcel at this location"

        # ----- recharge action -----------------------------------------
        elif action.action == DroneAction.RECHARGE:
            if self._at_recharge_station(drone.x, drone.y):
                drone.battery = 100.0
                reward += 0.1
                info["recharged"] = drone.id
            else:
                reward -= 0.05
                info["error"] = "not at a recharge station"

        # ----- wait action ---------------------------------------------
        elif action.action == DroneAction.WAIT:
            reward -= 0.02

        # ----- advance step count --------------------------------------
        self.state.step_count += 1

        # ----- check done ----------------------------------------------
        all_delivered = all(p.delivered for p in self.state.parcels)
        out_of_steps  = self.state.step_count >= self.state.max_steps

        if all_delivered:
            reward += 2.0      # big bonus for completing all deliveries
            info["success"] = True

        done = all_delivered or out_of_steps

        return StepResult(state=self.state, reward=reward, done=done, info=info)

    # ------------------------------------------------------------------ #
    #  score  — normalised 0.0-1.0 for graders                           #
    # ------------------------------------------------------------------ #
    def score(self) -> float:
        if self.state is None:
            return 0.0
        total    = len(self.state.parcels)
        delivered = sum(1 for p in self.state.parcels if p.delivered)
        if total == 0:
            return 0.0
        # partial credit for each parcel + time efficiency bonus
        base = delivered / total
        if delivered == total:
            efficiency = 1.0 - (self.state.step_count / self.state.max_steps)
            base += 0.2 * efficiency
        return round(min(base, 1.0), 4)

    def close(self):
        """Clean up resources (not needed for this environment)."""
        pass

    async def reset_async(self):
        """Asynchronous version of reset."""
        return self.reset()

    async def step_async(self, action):
        """Asynchronous version of step."""
        return self.step(action)

    async def get_state_async(self):
        """Asynchronous version of get_state."""
        return self.get_state()


        