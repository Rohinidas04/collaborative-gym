"""
Phase 2b LLM test: same chop-lettuce task, but driven by an LLM instead of hardcoded.

This is a minimal loop:
  1. Get observation from env
  2. Build a prompt with task description + observation + action history
  3. Call OpenAI, parse "Thought: ...\nAction: <ACTION>"
  4. Step the env with that action
  5. Repeat until terminated or max steps
"""
import os
import re
import sys
import toml
from openai import OpenAI

from collaborative_gym.envs import CoParaCookEnv


SYSTEM_PROMPT = """You are an agent in a small kitchen simulation. Your job is to complete the given task.

You must output exactly ONE action per turn in this format:
Thought: <your reasoning>
Action: <ACTION_NAME(...)>

Valid actions:
  MOVE_TO(x=<int>, y=<int>)
  INTERACT(target=<station_name>)
  PROCESS(target=<station_name>)

Rules:
- Use the kitchen layout to plan your moves
- You can only INTERACT with stations that are adjacent (one tile up/down/left/right)
- You cannot stand on a station tile -- you must stand next to it
- You hold at most one item at a time
- A dispenser gives you the raw item when you INTERACT with it (you must be empty-handed)
- A chopping board: INTERACT places an item on it (if you're holding one) OR picks one up (if you're empty)
- PROCESS chops whatever is on the chopping board"""


def build_prompt(task_description, obs, action_history):
    history_str = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(action_history)) or "  (none yet)"
    return f"""TASK:
{task_description}

CURRENT OBSERVATION:
  agent_position: {obs['agent_position']}
  agent_holding: {obs['agent_holding']}
  stations: {obs['stations']}
  current_time: {obs['current_time']}

ACTION HISTORY:
{history_str}

What is your next action? Respond with:
Thought: ...
Action: ..."""


def parse_action(text):
    """Extract the action string from 'Action: ...' in the LLM response."""
    m = re.search(r"Action:\s*(.+)", text)
    if not m:
        return None
    action = m.group(1).strip()
    # Cut off anything after the closing paren of the function call
    if action.endswith(")"):
        return action
    # Try to find first complete function call
    m2 = re.search(r"^([A-Z_]+\([^)]*\))", action)
    if m2:
        return m2.group(1)
    return action


def main():
    # Load API key
    secrets = toml.load("secrets.toml")
    api_key = secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: no OPENAI_API_KEY in secrets.toml or env")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    model = "gpt-4o"
    max_steps = 15

    env = CoParaCookEnv(team_members=["agent1"], env_id="phase2b_llm_test")
    obs, _ = env.reset()
    print("=== Initial observation ===")
    print(obs["public"])
    print()

    action_history = []
    terminated = False

    for step_num in range(1, max_steps + 1):
        prompt = build_prompt(env.task_description, obs["public"], action_history)
        print(f"=== Step {step_num} ===")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        text = response.choices[0].message.content
        print(f"LLM response:\n{text}\n")

        action = parse_action(text)
        if action is None:
            print(f"!! Could not parse action from response, stopping.")
            break

        print(f"Parsed action: {action}")
        obs, reward, terminated, _, info = env.step("agent1", action)
        action_history.append(action)

        print(f"  agent_holding: {obs['public']['agent_holding']}")
        print(f"  time: {obs['public']['current_time']}")
        if info.get("action_error"):
            print(f"  *** ERROR: {info['action_error']} ***")
        if terminated:
            print(f"  TERMINATED at step {step_num}")
            break
        print()

    perf = env.evaluate_task_performance()
    print("\n=== Final performance ===")
    print(perf)
    print(f"\nSuccess: {perf['task_completion'] == 1}")
    print(f"Steps used: {len(action_history)}")


if __name__ == "__main__":
    main()
