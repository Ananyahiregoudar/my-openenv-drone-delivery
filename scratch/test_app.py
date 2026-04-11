
from openenv.core.env_server.http_server import create_app
from drone_delivery_env.environment import DroneDeliveryEnvironment
from drone_delivery_env.models import StepAction, DroneDeliveryState
import uvicorn

print("Creating app...")
fastapi_app = create_app(
    DroneDeliveryEnvironment,
    StepAction,
    DroneDeliveryState,
    env_name="drone-delivery-fleet",
    max_concurrent_envs=1,
)
print("App created.")

if __name__ == "__main__":
    print("Starting server on 7861...")
    uvicorn.run(fastapi_app, host="127.0.0.1", port=7861)
