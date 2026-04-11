"""Quick test to verify all graders return scores strictly in (0, 1)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone_delivery_env.tasks.easy import grader as eg, run_easy_task
from drone_delivery_env.tasks.medium import grader as mg, run_medium_task
from drone_delivery_env.tasks.hard import grader as hg, run_hard_task
from drone_delivery_env.environment import DroneDeliveryEnvironment
from drone_delivery_env.models import WeatherCondition

all_scores = []

# Test 1: graders with no trajectory (fallback to baseline)
print('=== Test: Graders with no trajectory ===')
e = eg()
print(f'Easy grader (no traj):   {e}  -> in (0,1): {0 < e < 1}')
all_scores.append(e)
m = mg()
print(f'Medium grader (no traj): {m}  -> in (0,1): {0 < m < 1}')
all_scores.append(m)
h = hg()
print(f'Hard grader (no traj):   {h}  -> in (0,1): {0 < h < 1}')
all_scores.append(h)

# Test 2: graders with empty trajectory
print()
print('=== Test: Graders with empty trajectory ===')
e2 = eg({})
print(f'Easy grader (empty):   {e2}  -> in (0,1): {0 < e2 < 1}')
all_scores.append(e2)

# Test 3: graders with sample rewards
print()
print('=== Test: Graders with sample rewards ===')
good_rewards = [1.0, 1.0, 1.0, -0.01, -0.01, 1.0, 1.0, 2.0]
e3 = eg({'rewards': good_rewards})
print(f'Easy grader (good):   {e3}  -> in (0,1): {0 < e3 < 1}')
all_scores.append(e3)
m3 = mg({'rewards': good_rewards})
print(f'Medium grader (good): {m3}  -> in (0,1): {0 < m3 < 1}')
all_scores.append(m3)
h3 = hg({'rewards': good_rewards})
print(f'Hard grader (good):   {h3}  -> in (0,1): {0 < h3 < 1}')
all_scores.append(h3)

# Test 4: graders with bad rewards (all negative)
print()
print('=== Test: Graders with bad rewards ===')
bad_rewards = [-0.5, -0.3, -0.1, -0.05, -0.02]
e4 = eg({'rewards': bad_rewards})
print(f'Easy grader (bad):   {e4}  -> in (0,1): {0 < e4 < 1}')
all_scores.append(e4)
m4 = mg({'rewards': bad_rewards})
print(f'Medium grader (bad): {m4}  -> in (0,1): {0 < m4 < 1}')
all_scores.append(m4)
h4 = hg({'rewards': bad_rewards})
print(f'Hard grader (bad):   {h4}  -> in (0,1): {0 < h4 < 1}')
all_scores.append(h4)

# Test 5: env.score() directly
print()
print('=== Test: env.score() directly ===')
env = DroneDeliveryEnvironment(seed=42)
env.reset()
s = env.score()
print(f'Score (no deliveries): {s}  -> in (0,1): {0 < s < 1}')
all_scores.append(s)

# Test 6: env.score() with None state
env2 = DroneDeliveryEnvironment()
s2 = env2.score()
print(f'Score (None state):    {s2}  -> in (0,1): {0 < s2 < 1}')
all_scores.append(s2)

# Final check
print()
all_pass = all(0 < x < 1 for x in all_scores)
if all_pass:
    print('=== ALL CHECKS PASSED: Every score is strictly in (0, 1) ===')
else:
    failed = [(i, s) for i, s in enumerate(all_scores) if not (0 < s < 1)]
    print(f'=== SOME CHECKS FAILED: {failed} ===')
    sys.exit(1)
