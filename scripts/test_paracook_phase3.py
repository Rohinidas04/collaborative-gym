"""
Phase 3 hardcoded test: 2 agents make salad_advanced.

Plan (11 actions, expected total_time = 17):
  agent1: get lettuce -> board1 -> chop -> get plate -> transfer lettuce
          -> move (2,0) -> transfer tomato -> serve
  agent2: get tomato -> board2 -> chop (in parallel)
"""
from collaborative_gym.envs import CoParaCookEnv


PLAN = [
    ("agent1", "INTERACT(target=dispenser_lettuce)"),
    ("agent2", "INTERACT(target=dispenser_tomato)"),
    ("agent1", "INTERACT(target=chopping_board1)"),
    ("agent2", "INTERACT(target=chopping_board2)"),
    ("agent1", "PROCESS(target=chopping_board1)"),
    ("agent2", "PROCESS(target=chopping_board2)"),
    ("agent1", "INTERACT(target=plate_stack)"),
    ("agent1", "INTERACT(target=chopping_board1)"),
    ("agent1", "MOVE_TO(x=2, y=0)"),
    ("agent1", "INTERACT(target=chopping_board2)"),
    ("agent1", "INTERACT(target=serving_window)"),
]


def main():
    env = CoParaCookEnv(team_members=["agent1", "agent2"], env_id="phase3_hardcoded")
    obs, _ = env.reset()
    print("=== Initial ===")
    pub = obs["public"]
    print(f"  agents: {pub['agents']}")
    print(f"  pending: {pub['pending_orders']}")
    print()

    for i, (role, action) in enumerate(PLAN, 1):
        obs, _, term, _, info = env.step(role, action)
        pub = obs["public"]
        err = info.get("action_error")
        print(f"Step {i:2d}: [{role}] {action}")
        print(f"        t={pub['current_time']}  err={err}")
        for tm, a in pub["agents"].items():
            print(f"        {tm}: pos={a['position']} holding={a['holding']}")
        stations_with_items = {n: s["item"] for n, s in pub["stations"].items()
                               if s["item"] not in ("None", None)}
        if stations_with_items:
            print(f"        stations: {stations_with_items}")
        if pub["finished_orders"]:
            print(f"        finished: {pub['finished_orders']}")
        if err:
            print(f"  *** ERROR — stopping ***")
            break
        if term:
            print(f"        TERMINATED")
            break
        print()

    print("\n=== Final performance ===")
    perf = env.evaluate_task_performance()
    for k, v in perf.items():
        if k == "query":
            continue
        print(f"  {k}: {v}")

    assert perf["task_completion"] == 1, "task did not complete"
    assert perf["correctness"] == 1.0, "correctness < 1.0"
    print("\nHARDCODED 2-AGENT TEST PASSED")


if __name__ == "__main__":
    main()
