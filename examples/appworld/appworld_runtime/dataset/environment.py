"""
AppWorld task environment.

Owns the *execution* side of running AppWorld tasks (separate from the dataset,
which only holds the task rows): spinning up a fresh `appworld serve environment`
server per task, exposing the agent's `execute` tool, running the agent against a
task, and turning the outcome into an eval-result dict.
"""

import logging
import os
import random
from contextlib import contextmanager

from _local import ManagedProcess, make_eval_result

from .appworld_strands_tool import AppWorldExecutor
from .appworld_utils import ExitHook, get_free_port

logger = logging.getLogger(__name__)

# The per-task USER message — the trailing block of AppWorld's canonical prompt
# (the instructions + worked example are the SYSTEM prompt, seeded/tuned by the
# optimizer; see prompts/appworld_code_instructions.yaml and the client).
_USER_MESSAGE_TEMPLATE = (
    "Using these APIs, now generate code to solve the actual task:\n\n"
    "My name is: {first_name} {last_name}. My personal email is {email} and "
    "phone number is {phone_number}.\n"
    "Task: {instruction}"
)


class AppWorldEnvironment:
    """Runs AppWorld tasks against a per-task environment server.

    Usage:
        env = AppWorldEnvironment()
        agent = create_strands_agent(cfg, additional_tools=env.get_tools())
        result = env.get_execute_fn()(agent, data_sample)   # runs + evaluates the task
        eval_result = env.get_evaluate_fn()(data_sample, agent.messages, result)
    """

    def __init__(self):
        # `appworld serve environment` lives in this venv (set by the image).
        self.appworld_execute_env_path = os.environ.get("APPWORLD_EXECUTE_ENV", "")
        self.appworld_context = AppWorldExecutor()

    def _render_user_message(self, world) -> str:
        """The per-task user message: supervisor identity + the task instruction.

        The operating instructions + worked example are the *system* prompt (the
        thing the optimizer tunes); this only carries the per-task bits.
        """
        task = world.task
        sup = task.supervisor if isinstance(task.supervisor, dict) else {}
        return _USER_MESSAGE_TEMPLATE.format(
            first_name=sup.get("first_name", ""),
            last_name=sup.get("last_name", ""),
            email=sup.get("email", ""),
            phone_number=sup.get("phone_number", ""),
            instruction=task.instruction,
        )

    def get_tools(self) -> list:
        """The agent tool(s) this environment exposes (the AppWorld `execute` tool)."""
        return [self.appworld_context.get_execute_tool()]

    @contextmanager
    def task_context(self, task_id: str):
        """Start a fresh AppWorld environment server and yield a client for the task."""
        try:
            from appworld import AppWorld
        except ImportError:
            raise ImportError("The 'appworld' package is required for the AppWorld environment.")

        port = get_free_port(6000 + random.randint(0, 100) * 10)
        appworld_bin = os.path.join(self.appworld_execute_env_path, "bin/appworld")
        server = ManagedProcess(
            [appworld_bin, "serve", "environment", "--port", str(port)],
            "Uvicorn running on",
            10,
        )
        server.start()
        logger.info(f"Started AppWorld environment server at port {port}")

        with ExitHook(
            AppWorld(
                task_id=task_id,
                remote_environment_url=f"http://localhost:{port}",
                experiment_name=f"eval_{task_id}",
            ),
            lambda *exc: server.stop(),
        ) as world:
            yield world

    def get_execute_fn(self):
        """Return fn(agent, data) that runs one task against a fresh AppWorld env."""
        def fn(agent, data):
            with self.task_context(data["task_id"]) as world:
                user_message = self._render_user_message(world)
                self.appworld_context.set_world(world)
                response = agent(user_message)
                try:
                    world.save()
                except Exception as e:
                    logger.warning(f"Failed to save world state: {e}")

                evaluation: dict = world.evaluate().to_dict()
                logger.info(evaluation)
                return {
                    "id": world.task.id,
                    "message": response.message,
                    "messages": agent.messages,
                    **evaluation,
                }
        return fn

    def get_evaluate_fn(self):
        """Return fn(data, rollout_data, execute_result) -> eval-result dict."""
        def fn(data, rollout_data, execute_result):
            success = bool(execute_result.get("success", False))
            return make_eval_result(
                reward=1.0 if success else 0.0,
                metrics={
                    # Persist success so the RewardFunction reads it directly.
                    "success": success,
                    "difficulty": execute_result["difficulty"],
                    "num_tests": execute_result["num_tests"],
                    "passes": execute_result["passes"],
                    "failures": execute_result["failures"],
                },
                turn_rewards=[],
                messages=rollout_data,
            )
        return fn
