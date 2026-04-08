from pydantic import BaseModel
from typing import List, Optional
from enum import Enum
 
 
class DroneAction(str, Enum):
    MOVE_NORTH = "move_north"
    MOVE_SOUTH = "move_south"
    MOVE_EAST  = "move_east"
    MOVE_WEST  = "move_west"
    DELIVER    = "deliver"
    RECHARGE   = "recharge"
    WAIT       = "wait"
 
 
class WeatherCondition(str, Enum):
    CLEAR  = "clear"
    WINDY  = "windy"
    STORMY = "stormy"
 
 
class Parcel(BaseModel):
    id: str
    x: int
    y: int
    priority: bool = False
    delivered: bool = False
 
 
class NoFlyZone(BaseModel):
    x_min: int
    x_max: int
    y_min: int
    y_max: int
 
 
class Drone(BaseModel):
    id: str
    x: int
    y: int
    battery: float        # 0.0 to 100.0
    carrying_parcel: Optional[str] = None   # parcel id or None
 
 
class RechargeStation(BaseModel):
    x: int
    y: int
 
 
class DroneDeliveryState(BaseModel):
    drones: List[Drone]
    parcels: List[Parcel]
    no_fly_zones: List[NoFlyZone]
    recharge_stations: List[RechargeStation]
    weather: WeatherCondition
    step_count: int
    max_steps: int
    grid_size: int
 
 
class StepAction(BaseModel):
    drone_id: str
    action: DroneAction
 
 
class StepResult(BaseModel):
    state: DroneDeliveryState
    reward: float
    done: bool
    info: dict