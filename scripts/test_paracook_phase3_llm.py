"""
Phase 3 LLM test: 2 agents solve salad_advanced.

Each agent is a separate OpenAI call. Alternates agent1, agent2, agent1, ...
Each turn the agent sees: task, current obs, chat history, action history.

Usage:
    python -m scripts.test_paracook_phase3_llm                       # with messaging
    python -m scripts.test_paracook_phase3_llm --disable-messaging   # without
"""
import argparse
import os
import re
import sys
import toml
from openai import OpenAI

from collaborative_gym.envs import CoParaCookEnv


SYSTEM_PROMPT_BASE = """You are an agent in a small kitchen simulation working with a teammate.
Your job is to help complete the order.

Output EXACTLY ONE action per turn in this format:
Thought: <your reasoning>
Action: <ACTION>

Valid actions:
  MOVE_TO(x=<int>, y=<int>)
  INTERACT(target=<station_name>)
  PROCESS(target=<station_name>)
{msg_line}  FINISH()

Rules:
- You can only INTERACT or PROCESS with stations one tile away (up/down/left/right)
- You cannot stand on a station tile -- you must stand next to it
- You cannot stand on the same tile as your teammate
- You hold at most one item at a time
- A plate is a container; INTERACT with chopping board while holding a plate transfers
  chopped item from board onto plate
- Serving window accepts a plate whose contents match the recipe
"""


def build_system_prompt(disable_messaging):
    msg_line = "" if disable_messaging else "  SEND_MESSAGE(message=<text>)\n"
    return SYSTEM_PROMPT_BASE.format(msg_line=msg_line)


def build_user_prompt(role, task_description, public_obs, action_history):
    history_str = "\n".join(
        f"  {i+1}. [{r}] {a}" for i, (r, a) in enumerate(action_history)
    ) or "  (none yet)"
    chat_str = "\n".join(f"  {m}" for m in public_obs["chat_history"]) or "  (no messages)"

    return f"""TASK:
{task_description}

YOU ARE: {role}

CURRENT OBSERVATION:
  agents:   {public_obs['agents']}
  stations: {public_obs['stations']}
  pending_orders:  {public_obs['pending_orders']}
  finished_orders: {public_obs['finished_orders']}
  current_time: {public_obs['current_time']}

CHAT HISTORY:
{chat_str}

ACTION HISTORY (across both agents):
{history_str}

What is your next action? Respond with:
Thought: ...
Action: ..."""


def parse_action(text):
    m = re.search(r"Action:\s*(.+)", text)
    if not m:
        return None
    action = m.group(1).strip()
    # If the line has multiple parens, find last complete one
    if action.endswith(")"):
        return action
    m2 = re.search(r"^([A-Z_]+\([^)]*\))", action)
    if m2:
        return m2.group(1)
    return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable-messaging", action="store_true")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--model", default="gpt-4o")
    args = parser.parse_args()

    secrets = toml.load("secrets.toml")
    api_key = secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: no OPENAI_API_KEY")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    system_prompt = build_system_prompt(args.disable_messaging)

    env = CoParaCookEnv(
        team_members=["agent1", "agent2"],
        env_id=f"phase3_llm_{'nomsg' if args.disable_messaging else 'msg'}",
        disable_messaging=args.disable_messaging,
    )
    obs, _ = env.reset()

    print(f"=== Config: messaging={'OFF' if args.disable_messaging else 'ON'}, model={args.model} ===")
    print(f"=== Initial ===")
    print(f"  agents: {obs['public']['agents']}")
    print(f"  pending: {obs['public']['pending_orders']}")
    print()

    action_history = []
    roles = ["agent1", "agent2"]

    for step_num in range(1, args.max_steps + 1):
        role = roles[(step_num - 1) % 2]
        user_prompt = build_user_prompt(
            role, env.task_description, obs["public"], action_history
        )
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        text = response.choices[0].message.content
        action = parse_action(text)

        print(f"=== Step {step_num} ({role}) ===")
        print(f"LLM: {text[:300]}{'...' if len(text) > 300 else ''}")

        if action is None:
            print(f"!! Could not parse action, stopping.")
            break

        print(f"Parsed: {action}")
        obs, _, terminated, _, info = env.step(role, action)
        action_history.append((role, action))

        print(f"  t={obs['public']['current_time']}  err={info.get('action_error')}")
        for tm, a in obs["public"]["agents"].items():
            print(f"  {tm}: pos={a['position']} holding={a['holding']}")
        if obs["public"]["finished_orders"]:
            print(f"  finished: {obs['public']['finished_orders']}")
        if terminated:
            print(f"  TERMINATED at step {step_num}")
            break
        print()

    perf = env.evaluate_task_performance()
    print("\n=== Final performance ===")
    for k, v in perf.items():
        if k == "query":
            continue
        print(f"  {k}: {v}")
    print(f"\nSteps used: {len(action_history)}")


if __name__ == "__main__":
    main()
