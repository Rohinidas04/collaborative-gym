"""
Phase 3: Multi-agent ParaCook env targeting salad_advanced.

Custom 4x3 kitchen, supports 1 or 2 agents:
  y=2: DL .  .  DT     <- dispenser_lettuce, dispenser_tomato
  y=1: A1 C1 C2 A2     <- agents start, chopping_board1, chopping_board2
  y=0: P  .  .  SW     <- plate_stack, serving_window
        0  1  2  3

Actions per agent:
  MOVE_TO(x, y)              time +1; cannot enter station tile or other agent
  INTERACT(target)           time +1; context-dependent at adjacent station
  PROCESS(target)            time +4; chops raw item on a chopping board
  SEND_MESSAGE(message)      time +0; appears in shared chat history (Co-Gym only)
  FINISH()                   ends episode

Items:
  raw / chopped ingredient: {"name": "lettuce", "state": "raw"|"chopped"}
  plate (container):        {"name": "plate", "contents": [list of ingredients]}

Recipe: salad_advanced = plate containing chopped lettuce + chopped tomato.

Metrics tracked (matches ParaCook tmp.json fields):
  total_time, finished_orders, remaining_orders, agent_progress,
  agent_moving_time, agent_processing_time, agent_interact_time,
  agent_message_count, agent_errors.
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
    SEND_MESSAGE = "SEND_MESSAGE"
    FINISH = "FINISH"

    def __str__(self):
        return self.value


SALAD_ADVANCED = {
    "name": "salad_advanced",
    "ingredients": [
        {"item": "lettuce", "state": "chopped"},
        {"item": "tomato", "state": "chopped"},
    ],
}


@EnvFactory.register("paracook")
class CoParaCookEnv(CoEnv):
    """Phase 3 multi-agent ParaCook env, salad_advanced target."""

    def __init__(
        self,
        team_members: List[str],
        env_id: str,
        max_time_budget: int = 60,
        disable_messaging: bool = False,
    ):
        super().__init__(team_members=team_members, env_id=env_id)

        if not (1 <= len(team_members) <= 2):
            raise ValueError("CoParaCookEnv Phase 3 supports 1 or 2 agents")

        self.max_time_budget = max_time_budget
        self.disable_messaging = disable_messaging

        # Map
        self.grid_w = 4
        self.grid_h = 3
        self.stations = {
            "dispenser_lettuce": {"x": 0, "y": 2, "type": "dispenser",
                                  "provides": "lettuce", "item": None},
            "dispenser_tomato":  {"x": 3, "y": 2, "type": "dispenser",
                                  "provides": "tomato", "item": None},
            "chopping_board1":   {"x": 1, "y": 1, "type": "chopping_board",
                                  "item": None},
            "chopping_board2":   {"x": 2, "y": 1, "type": "chopping_board",
                                  "item": None},
            "plate_stack":       {"x": 0, "y": 0, "type": "plate_stack",
                                  "item": None},
            "serving_window":    {"x": 3, "y": 0, "type": "serving_window",
                                  "item": None},
        }
        self.recipes = [SALAD_ADVANCED]
        self.recipes_by_name = {r["name"]: r for r in self.recipes}
        self.initial_orders = ["salad_advanced"]

        # Agent start positions
        self.start_positions = {team_members[0]: (0, 1)}
        if len(team_members) > 1:
            self.start_positions[team_members[1]] = (3, 1)

        # Runtime state (initialized in reset)
        self.agents: Dict[str, Dict] = {}
        self.pending_orders: List[str] = []
        self.finished_orders: List[str] = []
        self.chat_history: List[Dict] = []
        self.current_time = 0
        self.done = False
        self.agent_metrics: Dict[str, Dict[str, int]] = {}

        # Task description
        msg_line = "" if disable_messaging else (
            "  SEND_MESSAGE(message=<text>) -- coordinate with teammate (no time cost)\n"
        )
        self.task_description = (
            "You are working in a 4x3 kitchen to cook orders.\n\n"
            "Kitchen layout:\n"
            "  dispenser_lettuce at (0,2) -- provides raw lettuce\n"
            "  dispenser_tomato  at (3,2) -- provides raw tomato\n"
            "  chopping_board1   at (1,1) -- chop things placed here\n"
            "  chopping_board2   at (2,1) -- chop things placed here\n"
            "  plate_stack       at (0,0) -- empty plates\n"
            "  serving_window    at (3,0) -- serve completed dishes\n\n"
            "Order: salad_advanced (plate with chopped lettuce + chopped tomato).\n\n"
            "Actions:\n"
            "  MOVE_TO(x=<int>, y=<int>)\n"
            "  INTERACT(target=<station_name>)\n"
            "  PROCESS(target=<station_name>)\n"
            + msg_line +
            "  FINISH()\n\n"
            "Rules: hold at most one item; must be adjacent (Manhattan dist 1) to "
            "interact; cannot stand on station tiles or on another agent."
        )

        self.additional_task_info = {
            tm: {"role_info": f"You control agent '{tm}'. Coordinate to serve the order."}
            for tm in team_members
        }

        self._build_action_spaces()

        self.example_question = "Make and serve a salad_advanced."
        self.example_trajectory = [
            {"thought": "Pick up lettuce.", "action": "INTERACT(target=dispenser_lettuce)",
             "observation": "holding raw lettuce"},
            {"thought": "Put on board to chop.", "action": "INTERACT(target=chopping_board1)",
             "observation": "board has raw lettuce"},
            {"thought": "Chop it.", "action": "PROCESS(target=chopping_board1)",
             "observation": "board has chopped lettuce"},
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
            human_readable_description="Chop a raw item on the board. Pattern: PROCESS(target=<station>)",
        )
        send_msg = UnicodeWithRegexPattern(
            min_length=0, max_length=MAX_UNICODE_LENGTH,
            regex_pattern=re.compile(r"^SEND_MESSAGE\(message=(.*)\)$", re.DOTALL),
            params=["message"],
            machine_readable_identifier=ParaCookActions.SEND_MESSAGE,
            human_readable_name="Send Message",
            human_readable_description="Send a message to teammate (no time cost). Pattern: SEND_MESSAGE(message=<text>)",
        )
        finish = UnicodeWithRegexPattern(
            min_length=0, max_length=MAX_UNICODE_LENGTH,
            regex_pattern=re.compile(r"^FINISH\(\)$", re.DOTALL),
            params=[],
            machine_readable_identifier=ParaCookActions.FINISH,
            human_readable_name="Finish",
            human_readable_description="End the episode. Pattern: FINISH()",
        )
        if self.disable_messaging:
            self.action_space = MultiSpace((move_to, interact, process, finish))
        else:
            self.action_space = MultiSpace((move_to, interact, process, send_msg, finish))
        self.private_action_space = MultiSpace(())

    # ---------- helpers ----------
    @staticmethod
    def _item_str(item):
        if item is None:
            return None
        if item.get("name") == "plate":
            contents = item.get("contents", [])
            if not contents:
                return "plate (empty)"
            cs = ", ".join(f"{c['name']}({c['state']})" for c in contents)
            return f"plate [{cs}]"
        return f"{item['name']} ({item['state']})"

    def _is_adjacent(self, ax, ay, sx, sy) -> bool:
        return abs(ax - sx) + abs(ay - sy) == 1

    def _tile_station(self, x, y) -> Optional[str]:
        for name, s in self.stations.items():
            if s["x"] == x and s["y"] == y:
                return name
        return None

    def _tile_other_agent(self, x, y, this_role) -> Optional[str]:
        for tm, a in self.agents.items():
            if tm == this_role:
                continue
            if a["x"] == x and a["y"] == y:
                return tm
        return None

    # ---------- observation ----------
    def get_obs(self) -> Dict:
        return {
            "public": {
                "agents": {
                    tm: {"position": f"({a['x']}, {a['y']})",
                         "holding": str(self._item_str(a["holding"]))}
                    for tm, a in self.agents.items()
                },
                "stations": {
                    name: {"position": f"({s['x']}, {s['y']})",
                           "type": s["type"],
                           "item": str(self._item_str(s["item"]))}
                    for name, s in self.stations.items()
                },
                "pending_orders": list(self.pending_orders),
                "finished_orders": list(self.finished_orders),
                "chat_history": [
                    f"[t={m['at_time']}] {m['role']}: {m['message']}"
                    for m in self.chat_history
                ],
                "current_time": self.current_time,
            },
            "private": {tm: {} for tm in self.team_members},
        }

    def obs_type(self) -> Dict[str, ObservationTypes]:
        return {
            "agents": ObservationTypes.NO_RENDER,
            "stations": ObservationTypes.NO_RENDER,
            "pending_orders": ObservationTypes.NO_RENDER,
            "finished_orders": ObservationTypes.NO_RENDER,
            "chat_history": ObservationTypes.NO_RENDER,
            "current_time": ObservationTypes.NO_RENDER,
        }

    def reset(self, options: dict[str, Any] | None = None):
        self.agents = {tm: {"x": x, "y": y, "holding": None}
                       for tm, (x, y) in self.start_positions.items()}
        for s in self.stations.values():
            s["item"] = None
        self.pending_orders = list(self.initial_orders)
        self.finished_orders = []
        self.chat_history = []
        self.current_time = 0
        self.done = False
        self.agent_metrics = {
            tm: {"actions": 0, "moving_time": 0, "processing_time": 0,
                 "interact_time": 0, "messages_sent": 0, "errors": 0}
            for tm in self.team_members
        }
        return self.get_obs(), {}

    # ---------- action handlers ----------
    def _do_move_to(self, role: str, new_x: int, new_y: int) -> Optional[str]:
        if not (0 <= new_x < self.grid_w and 0 <= new_y < self.grid_h):
            return f"({new_x},{new_y}) out of grid {self.grid_w}x{self.grid_h}."
        st = self._tile_station(new_x, new_y)
        if st:
            return f"({new_x},{new_y}) is occupied by station '{st}'."
        other = self._tile_other_agent(new_x, new_y, role)
        if other:
            return f"({new_x},{new_y}) is occupied by teammate '{other}'."
        self.agents[role]["x"] = new_x
        self.agents[role]["y"] = new_y
        self.current_time += 1
        self.agent_metrics[role]["moving_time"] += 1
        return None

    def _do_interact(self, role: str, target: str) -> Optional[str]:
        if target not in self.stations:
            return f"Unknown station '{target}'. Known: {list(self.stations.keys())}"
        s = self.stations[target]
        agent = self.agents[role]
        if not self._is_adjacent(agent["x"], agent["y"], s["x"], s["y"]):
            return (f"Not adjacent to '{target}'. {role} at ({agent['x']},{agent['y']}), "
                    f"station at ({s['x']},{s['y']}).")

        if s["type"] == "dispenser":
            if agent["holding"] is not None:
                return (f"Cannot pick from '{target}', already holding "
                        f"{self._item_str(agent['holding'])}.")
            agent["holding"] = {"name": s["provides"], "state": "raw"}
            self.current_time += 1
            self.agent_metrics[role]["interact_time"] += 1
            return None

        if s["type"] == "chopping_board":
            held = agent["holding"]
            board = s["item"]
            # Plate held + chopped item on board: transfer to plate
            if held and held.get("name") == "plate" and board and board.get("state") == "chopped":
                held.setdefault("contents", []).append(board)
                s["item"] = None
                self.current_time += 1
                self.agent_metrics[role]["interact_time"] += 1
                return None
            # Empty hand + board has item: pick up
            if held is None and board is not None:
                agent["holding"] = board
                s["item"] = None
                self.current_time += 1
                self.agent_metrics[role]["interact_time"] += 1
                return None
            # Raw item in hand + empty board: place
            if held and held.get("name") != "plate" and board is None:
                s["item"] = held
                agent["holding"] = None
                self.current_time += 1
                self.agent_metrics[role]["interact_time"] += 1
                return None
            if held and held.get("name") == "plate" and board is None:
                return f"'{target}' is empty; nothing to add to plate."
            if held and held.get("name") == "plate" and board and board.get("state") != "chopped":
                return f"'{target}' has raw {board['name']}; chop it before transferring."
            if held and board:
                return (f"Holding {self._item_str(held)} but '{target}' already has "
                        f"{self._item_str(board)}.")
            return f"Cannot INTERACT with '{target}' in current state."

        if s["type"] == "plate_stack":
            if agent["holding"] is not None:
                return (f"Cannot grab plate, already holding "
                        f"{self._item_str(agent['holding'])}.")
            agent["holding"] = {"name": "plate", "contents": []}
            self.current_time += 1
            self.agent_metrics[role]["interact_time"] += 1
            return None

        if s["type"] == "serving_window":
            held = agent["holding"]
            if held is None:
                return f"Nothing to serve at '{target}'."
            if held.get("name") != "plate":
                return f"Can only serve a plate, not {self._item_str(held)}."
            matched = self._match_plate_to_order(held)
            if matched is None:
                contents = ", ".join(
                    f"{c['name']}({c['state']})" for c in held.get("contents", [])
                ) or "empty"
                return (f"Plate contents [{contents}] do not match any pending order "
                        f"{self.pending_orders}.")
            self.pending_orders.remove(matched)
            self.finished_orders.append(matched)
            agent["holding"] = None
            self.current_time += 1
            self.agent_metrics[role]["interact_time"] += 1
            if not self.pending_orders:
                self.done = True
            return None

        return f"Don't know how to INTERACT with type '{s['type']}'."

    def _do_process(self, role: str, target: str) -> Optional[str]:
        if target not in self.stations:
            return f"Unknown station '{target}'."
        s = self.stations[target]
        agent = self.agents[role]
        if not self._is_adjacent(agent["x"], agent["y"], s["x"], s["y"]):
            return f"Not adjacent to '{target}'."
        if s["type"] != "chopping_board":
            return f"Can only PROCESS at a chopping_board."
        if s["item"] is None:
            return f"Nothing on '{target}' to chop."
        if s["item"]["state"] != "raw":
            return f"Item on '{target}' already {s['item']['state']}."
        s["item"]["state"] = "chopped"
        self.current_time += 4
        self.agent_metrics[role]["processing_time"] += 4
        return None

    def _do_send_message(self, role: str, message: str) -> Optional[str]:
        if self.disable_messaging:
            return "Messaging is disabled in this configuration."
        self.chat_history.append({
            "role": role, "message": message, "at_time": self.current_time,
        })
        self.agent_metrics[role]["messages_sent"] += 1
        return None

    def _match_plate_to_order(self, plate) -> Optional[str]:
        plate_set = {(c["name"], c["state"]) for c in plate.get("contents", [])}
        for order in list(self.pending_orders):
            recipe = self.recipes_by_name.get(order)
            if not recipe:
                continue
            required = {(ing["item"], ing["state"]) for ing in recipe["ingredients"]}
            if required.issubset(plate_set):
                return order
        return None

    # ---------- step ----------
    def step(self, role: str, action: str):
        info = {"action_start_time": time.time(), "action_error": None}

        parsed_action, private, action_id, err_msg = self.parse_and_validate_action(role, action)
        if err_msg:
            if role in self.agent_metrics:
                self.agent_metrics[role]["errors"] += 1
            return self.handle_action_error(err_msg, private)

        for k in parsed_action:
            parsed_action[k] = post_process_parsed_function_arg(parsed_action[k])

        info["action"] = action_id
        terminated = False

        try:
            if action_id == ParaCookActions.MOVE_TO:
                err = self._do_move_to(role, int(parsed_action["x"]), int(parsed_action["y"]))
            elif action_id == ParaCookActions.INTERACT:
                err = self._do_interact(role, parsed_action["target"].strip())
            elif action_id == ParaCookActions.PROCESS:
                err = self._do_process(role, parsed_action["target"].strip())
            elif action_id == ParaCookActions.SEND_MESSAGE:
                err = self._do_send_message(role, parsed_action["message"])
            elif action_id == ParaCookActions.FINISH:
                terminated = True
                err = None
            else:
                err = f"Unknown action {action_id}"

            if err:
                self.agent_metrics[role]["errors"] += 1
                return self.handle_action_error(err, private)

            self.agent_metrics[role]["actions"] += 1
            logger.info(
                f"[t={self.current_time}] {role} did {action} -> "
                f"agents={ {tm: (a['x'],a['y'],self._item_str(a['holding'])) for tm,a in self.agents.items()} }"
            )

            if self.done:
                terminated = True
            if self.current_time >= self.max_time_budget:
                terminated = True
        except Exception as e:
            self.agent_metrics[role]["errors"] += 1
            return self.handle_action_error(f"Step error: {e}", private)
        finally:
            info["action_end_time"] = time.time()

        return self.get_obs(), 0, terminated, False, info

    def close(self):
        pass

    def evaluate_task_performance(self) -> Dict:
        total = max(1, len(self.initial_orders))
        finished = len(self.finished_orders)
        correctness = finished / total
        total_time = self.current_time or 1
        efficiency = max(0.0, min(1.0, self.max_time_budget / max(total_time, 1) / 2))
        return {
            "outcome": {
                "finished_orders": list(self.finished_orders),
                "remaining_orders": list(self.pending_orders),
                "total_orders": list(self.initial_orders),
                "total_time": self.current_time,
            },
            "agent_progress": {tm: f"{m['actions']}/{m['actions']}"
                               for tm, m in self.agent_metrics.items()},
            "agent_moving_time": {tm: m["moving_time"]
                                  for tm, m in self.agent_metrics.items()},
            "agent_processing_time": {tm: m["processing_time"]
                                      for tm, m in self.agent_metrics.items()},
            "agent_interact_time": {tm: m["interact_time"]
                                    for tm, m in self.agent_metrics.items()},
            "agent_message_count": {tm: m["messages_sent"]
                                    for tm, m in self.agent_metrics.items()},
            "agent_errors": {tm: m["errors"]
                             for tm, m in self.agent_metrics.items()},
            "query": self.task_description,
            "task_completion": 1 if correctness == 1.0 else 0,
            "performance_rating": (correctness + efficiency) / 2,
            "correctness": correctness,
            "efficiency": efficiency,
        }

    def __repr__(self):
        return (f"CoParaCookEnv(grid={self.grid_w}x{self.grid_h}, "
                f"members={self.team_members}, msg={'off' if self.disable_messaging else 'on'})")
