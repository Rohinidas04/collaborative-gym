"""
Phase 2a smoke test: directly instantiate CoParaCookEnv and step it manually.
No LLM, no Redis, no Docker. Just prove the env works.
"""
from collaborative_gym.envs import CoParaCookEnv


def main():
    env = CoParaCookEnv(team_members=["agent1"], env_id="phase2a_test")
    obs, _ = env.reset()
    print("=== Initial observation ===")
    print(obs)
    print()

    # Step 1: move to (0, 1) -- partial progress
    print("=== Step 1: MOVE_TO(x=0, y=1) ===")
    obs, reward, terminated, _, info = env.step("agent1", "MOVE_TO(x=0, y=1)")
    print(f"obs: {obs['public']}")
    print(f"terminated: {terminated}, action_error: {info.get('action_error')}")
    print()

    # Step 2: move to goal (1, 1)
    print("=== Step 2: MOVE_TO(x=1, y=1) ===")
    obs, reward, terminated, _, info = env.step("agent1", "MOVE_TO(x=1, y=1)")
    print(f"obs: {obs['public']}")
    print(f"terminated: {terminated}, action_error: {info.get('action_error')}")
    print()

    # Final scoring
    print("=== Final performance ===")
    print(env.evaluate_task_performance())


if __name__ == "__main__":
    main()
