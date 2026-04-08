import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import DroneDeliveryEnvironment

def main():
    # This is a dummy OpenEnv server entrypoint for validation purposes.
    # In a real deployed OpenEnv setting, you would wrap DroneDeliveryEnvironment 
    # to serve the standard HTTP API points: /reset, /step, /state.
    env = DroneDeliveryEnvironment()
    print("OpenEnv server emulator starting...")

if __name__ == "__main__":
    main()
