"""
WebShop task environment.

Owns the *execution* side of running WebShop tasks (separate from the dataset,
which only holds the task rows): starting the WebShop gym server (a Flask wrapper
around ``WebAgentTextEnv``), exposing the agent's shopping tools
(``search`` / ``click`` / ``get_available_actions``), running the agent against a
task, and turning the final purchase reward into an eval-result dict.

The WebShop gym (``app_simple.py``) runs in a separate virtualenv (``VENV_AUX``,
pinned to the WebShop dependency stack) and is reached over HTTP — the agent side
(strands + newer deps) never imports the gym, mirroring the internal dual-venv
design. Each task ``reset``s the gym to a goal index, and the goal's *instruction*
becomes the per-task user message.
"""

import json
import logging
import os
from contextlib import contextmanager

import httpx
from strands import tool

from _local import ManagedProcess, make_eval_result

logger = logging.getLogger(__name__)

# The per-task USER message: the shopping goal fetched from the environment.
# The operating instructions (tool protocol + strategy) are the SYSTEM prompt —
# the thing the optimizer tunes (see prompts/webshop_instructions.yaml).
_USER_MESSAGE_TEMPLATE = "Shopping task:\n{instruction}"


class WebShopEnvironment:
    """Runs WebShop tasks against a WebShop gym server.

    Usage:
        env = WebShopEnvironment()
        agent = create_strands_agent(cfg, additional_tools=env.get_tools())
        result = env.get_execute_fn()(agent, data_sample)   # runs + evaluates the task
        eval_result = env.get_evaluate_fn()(data_sample, agent.messages, result)
    """

    def __init__(self, flask_url: str | None = None):
        # The WebShop gym (app_simple.py) lives in this venv (set by the image),
        # and is served over HTTP at flask_url.
        self.flask_url = flask_url or os.environ.get("WEBSHOP_FLASK_URL", "http://localhost:3000")
        self.webshop_aux_env = os.environ.get("WEBSHOP_EXECUTE_ENV", "")
        self.webshop_path = os.environ.get("WEBSHOP_PATH", "/app/webshop")
        # Per-task interaction state, set by the execute fn while a task runs.
        self._session_id: str | None = None
        # The final reward WebShop returns on a `buy now` (done=True) step.
        self._last_reward = 0.0
        self._done = False
        self._server: ManagedProcess | None = None

    # --- gym server lifecycle -------------------------------------------------

    def _ensure_server(self) -> None:
        """Start the WebShop gym Flask server once (loads products + search index)."""
        if self._server and self._server.is_running():
            return
        try:
            with httpx.Client(timeout=2.0) as c:
                if c.get(f"{self.flask_url}/ping").status_code == 200:
                    logger.info("WebShop gym already running")
                    return
        except httpx.HTTPError:
            pass

        python_bin = os.path.join(self.webshop_aux_env, "bin/python")
        app_simple = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app_simple.py")
        logger.info("Starting WebShop gym server (app_simple.py) ...")
        # Loading the 20k product index takes a while — allow a generous startup window.
        self._server = ManagedProcess(
            [python_bin, app_simple, "--port", "3000"],
            ready_pattern="Loaded",  # app_simple prints "[app_simple] Loaded N goals"
            startup_timeout=float(os.environ.get("WEBSHOP_STARTUP_TIMEOUT", "900")),
        )
        self._server.start()
        logger.info("WebShop gym server ready")

    # --- agent tools (search / click / get_available_actions) -----------------

    def get_tools(self) -> list:
        """The shopping tools the agent uses (search / click / get_available_actions).

        These POST to the gym server's /step and /get_actions endpoints, tracking the
        final reward the environment emits when the agent clicks "buy now".
        """

        def _step(action: str) -> dict:
            with httpx.Client(timeout=120.0) as c:
                r = c.post(
                    f"{self.flask_url}/step",
                    json={"session_id": self._session_id, "action": action},
                )
                r.raise_for_status()
                result = r.json()
            # Capture the terminal reward (WebShop only pays out on the buy-now step).
            if "reward" in result:
                self._last_reward = float(result.get("reward", 0.0) or 0.0)
            if result.get("done"):
                self._done = True
            return result

        @tool
        def search(query: str) -> str:
            """Search WebShop for products with space-separated keywords.

            Returns a text observation listing matching products (names, prices, ASINs).
            Example: search("blue wireless headphones")
            """
            if not query or not query.strip():
                return json.dumps({"error": "Empty search query", "success": False})
            result = _step(f"search[{query.strip()}]")
            return json.dumps({
                "observation": result.get("observation", ""),
                "reward": result.get("reward", 0.0),
                "done": result.get("done", False),
                "available_actions": result.get("available_actions", {}),
                "success": True,
            })

        @tool
        def click(element: str) -> str:
            """Click an element in WebShop.

            Valid elements: product ASINs (e.g. "b09kqnh5c6"), "buy now" (purchase),
            "back to search", "< prev", "next >", product options (e.g. "blue",
            "large"), and info tabs ("description", "features", "reviews").
            Returns a text observation of the resulting page.
            """
            if not element or not element.strip():
                return json.dumps({"error": "Empty element", "success": False})
            result = _step(f"click[{element.strip().lower()}]")
            return json.dumps({
                "observation": result.get("observation", ""),
                "reward": result.get("reward", 0.0),
                "done": result.get("done", False),
                "available_actions": result.get("available_actions", {}),
                "success": True,
            })

        @tool
        def get_available_actions() -> str:
            """List all clickable elements on the current WebShop page.

            Returns whether a search bar is available and the list of clickable elements.
            """
            with httpx.Client(timeout=120.0) as c:
                r = c.post(f"{self.flask_url}/get_actions", json={"session_id": self._session_id})
                r.raise_for_status()
                result = r.json()
            return json.dumps({
                "has_search_bar": result.get("has_search_bar", False),
                "clickables": result.get("clickables", []),
                "success": True,
            })

        return [search, click, get_available_actions]

    # --- per-task run ---------------------------------------------------------

    @contextmanager
    def task_context(self, task_id: str):
        """Reset the gym to ``task_id`` and yield the task instruction."""
        self._ensure_server()
        self._session_id = f"session_{task_id}"
        self._last_reward = 0.0
        self._done = False
        with httpx.Client(timeout=900.0) as c:  # first reset may build/load the index
            r = c.post(
                f"{self.flask_url}/reset",
                json={"task_id": int(task_id), "session_id": self._session_id},
            )
            r.raise_for_status()
            reset = r.json()
        try:
            yield reset.get("instruction", "")
        finally:
            with httpx.Client(timeout=30.0) as c:
                try:
                    c.post(f"{self.flask_url}/close", json={"session_id": self._session_id})
                except httpx.HTTPError:
                    pass
            self._session_id = None

    def get_execute_fn(self):
        """Return fn(agent, data) that runs one task against the WebShop gym."""
        def fn(agent, data):
            with self.task_context(data["task_id"]) as instruction:
                logger.info(f"Task {data['task_id']}: {instruction[:120]}")
                user_message = _USER_MESSAGE_TEMPLATE.format(instruction=instruction)
                response = agent(user_message)
                return {
                    "id": data["task_id"],
                    "message": response.message,
                    "messages": agent.messages,
                    "reward": self._last_reward,
                    "done": self._done,
                    "success": self._last_reward > 0,
                }
        return fn

    def get_evaluate_fn(self):
        """Return fn(data, rollout_data, execute_result) -> eval-result dict."""
        def fn(data, rollout_data, execute_result):
            reward = float(execute_result.get("reward", 0.0) or 0.0)
            return make_eval_result(
                reward=reward,
                metrics={
                    # Persist success so the RewardFunction reads it directly.
                    # WebShop reward is a [0, 1] match score; "success" = a full match.
                    "success": reward >= 1.0,
                    "score": reward,
                    "done": bool(execute_result.get("done", False)),
                },
                turn_rewards=[],
                messages=rollout_data,
            )
        return fn
