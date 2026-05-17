"""
Phase 2b hardcoded test: solve the chop-lettuce task with the exact action sequence.

Plan (6 actions, total time = 1+1+1+1+4+1 = 9 time units):
  1. MOVE_TO(x=0, y=0)              adjacent to dispenser1
  2. INTERACT(target=dispenser1)    pick up raw lettuce
  3. MOVE_TO(x=2, y=1)              adjacent to chopping_board1
  4. INTERACT(target=chopping_board1)  put lettuce on board
  5. PROCESS(target=chopping_board1)   chop (time +4)
  6. INTERACT(target=chopping_board1)  pick up chopped lettuce -> done
"""
from collaborative_gym.envs import CoParaCookEnv


PLAN = [
    "MOVE_TO(x=0, y=0)",
    "INTERACT(target=dispenser1)",
    "MOVE_TO(x=2, y=1)",
    "INTERACT(target=chopping_board1)",
    "PROCESS(target=chopping_board1)",
    "INTERACT(target=chopping_board1)",
]


def run_plan(env, plan):
    obs, _ = env.reset()
    print("=== Initial ===")
    print(obs["public"])
    print()
    for i, action in enumerate(plan, 1):
        print(f"=== Step {i}: {action} ===")
        obs, reward, terminated, _, info = env.step("agent1", action)
        print(f"  agent_position: {obs['public']['agent_position']}")
        print(f"  agent_holding:  {obs['public']['agent_holding']}")
        print(f"  stations:       {obs['public']['stations']}")
        print(f"  time:           {obs['public']['current_time']}")
        print(f"  terminated:     {terminated}")
        if info.get("action_error"):
            print(f"  *** ERROR: {info['action_error']} ***")
            return False
        if terminated:
            break
        print()
    print("\n=== Final performance ===")
    print(env.evaluate_task_performance())
    return True


def main():
    env = CoParaCookEnv(team_members=["agent1"], env_id="phase2b_chop_lettuce")
    ok = run_plan(env, PLAN)
    if ok:
        perf = env.evaluate_task_performance()
        assert perf["task_completion"] == 1, "task did not complete"
        assert perf["performance_rating"] == 1.0, "score is not 1.0"
        print("\nHARDCODED SOLUTION PASSED")
    else:
        print("\nHARDCODED SOLUTION FAILED")


if __name__ == "__main__":
    main()
