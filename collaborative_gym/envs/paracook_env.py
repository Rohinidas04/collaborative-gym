"""
Phase 2b: Minimal ParaCook env with one-task chopping.

3x3 kitchen:
  - dispenser1 at (0,1) provides raw lettuce
  - chopping_board1 at (1,1) chops things placed on it
  - Agent starts at (1,0)

Task: get a chopped lettuce (held by agent).

Actions:
  MOVE_TO(x=<int>, y=<int>)
  INTERACT(target=<station_name>)
  PROCESS(target=<station_name>)

Rules:
  - Agent holds at most ONE item
  - INTERACT only works if adjacent (Manhattan dist = 1)
  - PROCESS works only at chopping_board with a raw item on it (takes 4 time units)
"""

import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from collaborative_gym.core import CoEnv, ObservationTypes, logger
from collaborative_gym.envs.registry import EnvFactory
from collaborative_gym.spaces import (
    MAX_UNICODE_LENGTH,
    MultiSpace,
    UnicodeWithRegexPattern,
)
from collaborative_gym.utils.string import post_process_parsed_function_arg


class ParaCookActions(Enum):
    MOVE_TO = "MOVE_TO"
    INTERACT = "INTERACT"
    PROCESS = "PROCESS"

    def __str__(self):
        return self.value


@EnvFactory.register("paracook")
class CoParaCookEnv(CoEnv):
    """Phase 2b: chop one lettuce."""

    def __init__(
        self,
        team_members: List[str],
        env_id: str,
        max_time_budget: int = 30,
    ):
        super().__init__(team_members=team_members, env_id=env_id)

        self.grid_w = 3
        self.grid_h = 3
        self.max_time_budget = max_time_budget

        self.stations = {
            "dispenser1": {"x": 0, "y": 1, "type": "dispenser", "provides": "lettuce", "item": None},
            "chopping_board1": {"x": 1, "y": 1, "type": "chopping_board", "item": None},
        }

        self.agent_x = 1
        self.agent_y = 0
        self.agent_holding: Optional[Dict] = None
        self.current_time = 0
        self.done = False

        self.task_description = (
            "You are in a small kitchen. Your task is to get a CHOPPED lettuce.\n\n"
            "Kitchen layout (3x3):\n"
            "  dispenser1 at (0,1) -- provides raw lettuce\n"
            "  chopping_board1 at (1,1) -- chop things here\n"
            "  You start at (1,0).\n\n"
            "Actions:\n"
            "  MOVE_TO(x=<int>, y=<int>) -- move to position (cannot stand on a station tile)\n"
            "  INTERACT(target=<station>) -- pick up / put down at adjacent station\n"
            "  PROCESS(target=<station>) -- chop the item on the board (takes 4 time units)\n\n"
            "Done when you are holding a chopped lettuce."
        )

        self.additional_task_info = {
            tm: {"role_info": "You control the only agent."}
            for tm in team_members
        }

        self._build_action_spaces()

        self.example_question = "Get a chopped lettuce."
        self.example_trajectory = [
            {"thought": "Move adjacent to dispenser1.", "action": "MOVE_TO(x=0, y=0)", "observation": "at (0,0)"},
            {"thought": "Pick up lettuce.", "action": "INTERACT(target=dispenser1)", "observation": "holding raw lettuce"},
            {"thought": "Move adjacent to chopping_board1.", "action": "MOVE_TO(x=2, y=1)", "observation": "at (2,1)"},
            {"thought": "Place on board.", "action": "INTERACT(target=chopping_board1)", "observation": "board has raw lettuce"},
            {"thought": "Chop.", "action": "PROCESS(target=chopping_board1)", "observation": "board has chopped lettuce"},
            {"thought": "Pick up. Done.", "action": "INTERACT(target=chopping_board1)", "observation": "holding chopped lettuce"},
        ]

    def _build_action_spaces(self):
        move_to = UnicodeWithRegexPattern(
            min_length=0, max_length=MAX_UNICODE_LENGTH,
            regex_pattern=re.compile(r"^MOVE_TO\(x=(\d+),\s*y=(\d+)\)$", re.DOTALL),
            params=["x", "y"],
            machine_readable_identifier=ParaCookActions.MOVE_TO,
            human_readable_name="Move To",
            human_readable_description="Move to (x, y). Pattern: MOVE_TO(x=<int>, y=<int>)",
        )
        interact = UnicodeWithRegexPattern(
            min_length=0, max_length=MAX_UNICODE_LENGTH,
            regex_pattern=re.compile(r"^INTERACT\(target=(.*)\)$", re.DOTALL),
            params=["target"],
            machine_readable_identifier=ParaCookActions.INTERACT,
            human_readable_name="Interact",
            human_readable_description="Interact with adjacent station. Pattern: INTERACT(target=<station>)",
        )
        process = UnicodeWithRegexPattern(
            min_length=0, max_length=MAX_UNICODE_LENGTH,
            regex_pattern=re.compile(r"^PROCESS\(target=(.*)\)$", re.DOTALL),
            params=["target"],
            machine_readable_identifier=ParaCookActions.PROCESS,
            human_readable_name="Process",
            human_readable_description="Chop the item on the board. Pattern: PROCESS(target=<station>)",
        )
        self.action_space = MultiSpace((move_to, interact, process))
        self.private_action_space = MultiSpace(())

    def _is_adjacent(self, ax, ay, sx, sy) -> bool:
        return abs(ax - sx) + abs(ay - sy) == 1

    def _world_state_str(self) -> str:
        held = self.agent_holding
        held_str = f"{held['name']}({held['state']})" if held else "nothing"
        stations_str = ", ".join(
            f"{name}@({s['x']},{s['y']})"
            + (f"[has {s['item']['name']}({s['item']['state']})]" if s.get("item") else "")
            for name, s in self.stations.items()
        )
        return f"agent at ({self.agent_x},{self.agent_y}) holding {held_str}; stations: {stations_str}; time={self.current_time}"

    def get_obs(self) -> Dict:
        return {
            "public": {
                "agent_position": f"({self.agent_x}, {self.agent_y})",
                "agent_holding": (
                    f"{self.agent_holding['name']} ({self.agent_holding['state']})"
                    if self.agent_holding else "nothing"
                ),
                "stations": {
                    name: {
                        "x": s["x"], "y": s["y"], "type": s["type"],
                        "item": (f"{s['item']['name']} ({s['item']['state']})" if s.get("item") else None),
                    }
                    for name, s in self.stations.items()
                },
                "current_time": self.current_time,
                "done": self.done,
            },
            "private": {tm: {} for tm in self.team_members},
        }

    def obs_type(self) -> Dict[str, ObservationTypes]:
        return {
            "agent_position": ObservationTypes.NO_RENDER,
            "agent_holding": ObservationTypes.NO_RENDER,
            "stations": ObservationTypes.NO_RENDER,
            "current_time": ObservationTypes.NO_RENDER,
            "done": ObservationTypes.NO_RENDER,
        }

    def reset(self, options: dict[str, Any] | None = None):
        self.agent_x = 1
        self.agent_y = 0
        self.agent_holding = None
        self.current_time = 0
        self.done = False
        for s in self.stations.values():
            s["item"] = None
        return self.get_obs(), {}

    def _do_move_to(self, new_x: int, new_y: int) -> Optional[str]:
        if not (0 <= new_x < self.grid_w and 0 <= new_y < self.grid_h):
            return f"({new_x},{new_y}) out of grid {self.grid_w}x{self.grid_h}."
        for name, s in self.stations.items():
            if s["x"] == new_x and s["y"] == new_y:
                return f"({new_x},{new_y}) is occupied by station '{name}'."
        self.agent_x = new_x
        self.agent_y = new_y
        self.current_time += 1
        return None

    def _do_interact(self, target: str) -> Optional[str]:
        if target not in self.stations:
            return f"Unknown station '{target}'. Known: {list(self.stations.keys())}"
        s = self.stations[target]
        if not self._is_adjacent(self.agent_x, self.agent_y, s["x"], s["y"]):
            return f"Not adjacent to '{target}'. Agent at ({self.agent_x},{self.agent_y}), station at ({s['x']},{s['y']})."

        if s["type"] == "dispenser":
            if self.agent_holding is not None:
                return f"Cannot pick up from '{target}', already holding {self.agent_holding['name']}."
            self.agent_holding = {"name": s["provides"], "state": "raw"}
            self.current_time += 1
            return None

        if s["type"] == "chopping_board":
            if self.agent_holding is not None and s["item"] is None:
                s["item"] = self.agent_holding
                self.agent_holding = None
                self.current_time += 1
                return None
            elif self.agent_holding is None and s["item"] is not None:
                self.agent_holding = s["item"]
                s["item"] = None
                self.current_time += 1
                return None
            elif self.agent_holding is not None and s["item"] is not None:
                return f"'{target}' already has {s['item']['name']}."
            else:
                return f"'{target}' is empty and agent is empty-handed."

        return f"Don't know how to INTERACT with '{s['type']}'."

    def _do_process(self, target: str) -> Optional[str]:
        if target not in self.stations:
            return f"Unknown station '{target}'."
        s = self.stations[target]
        if not self._is_adjacent(self.agent_x, self.agent_y, s["x"], s["y"]):
            return f"Not adjacent to '{target}'."
        if s["type"] != "chopping_board":
            return f"Can only PROCESS at a chopping_board."
        if s["item"] is None:
            return f"Nothing on '{target}' to chop."
        if s["item"]["state"] != "raw":
            return f"Item already {s['item']['state']}."
        s["item"]["state"] = "chopped"
        self.current_time += 4
        return None

    def step(self, role: str, action: str):
        info = {"action_start_time": time.time()}

        parsed_action, private, action_id, err_msg = self.parse_and_validate_action(role, action)
        if err_msg:
            return self.handle_action_error(err_msg, private)

        for k in parsed_action:
            parsed_action[k] = post_process_parsed_function_arg(parsed_action[k])

        info["action"] = action_id
        info["action_error"] = None
        terminated = False

        try:
            if action_id == ParaCookActions.MOVE_TO:
                err = self._do_move_to(int(parsed_action["x"]), int(parsed_action["y"]))
            elif action_id == ParaCookActions.INTERACT:
                err = self._do_interact(parsed_action["target"].strip())
            elif action_id == ParaCookActions.PROCESS:
                err = self._do_process(parsed_action["target"].strip())
            else:
                err = f"Unknown action {action_id}"

            if err:
                return self.handle_action_error(err, private)

            logger.info(self._world_state_str())

            if (
                self.agent_holding
                and self.agent_holding["name"] == "lettuce"
                and self.agent_holding["state"] == "chopped"
            ):
                self.done = True
                terminated = True

            if self.current_time >= self.max_time_budget:
                terminated = True

        except Exception as e:
            return self.handle_action_error(f"Step error: {e}", private)
        finally:
            info["action_end_time"] = time.time()

        return self.get_obs(), 0, terminated, False, info

    def close(self):
        pass

    def evaluate_task_performance(self) -> Dict:
        success = (
            self.agent_holding is not None
            and self.agent_holding["name"] == "lettuce"
            and self.agent_holding["state"] == "chopped"
        )
        return {
            "outcome": {
                "final_position": (self.agent_x, self.agent_y),
                "agent_holding": (
                    f"{self.agent_holding['name']} ({self.agent_holding['state']})"
                    if self.agent_holding else None
                ),
                "total_time": self.current_time,
                "done": self.done,
            },
            "query": self.task_description,
            "task_completion": 1 if success else 0,
            "performance_rating": 1.0 if success else 0.0,
        }

    def __repr__(self):
        return f"CoParaCookEnv(grid={self.grid_w}x{self.grid_h}, members={self.team_members})"
