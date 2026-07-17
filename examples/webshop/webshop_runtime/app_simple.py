"""
Simple Flask wrapper for the WebShop gym environment.

Wraps ``WebAgentTextEnv`` (from the public princeton-nlp/webshop repo) and exposes
plain JSON endpoints (no HTML rendering). Runs in VENV_AUX (the WebShop dependency
stack); the agent side (VENV_MAIN) reaches it over HTTP.

Dataset size is configurable via env so the same file serves the public small set
(1000 products) or a larger one:
    WEBSHOP_NUM_PRODUCTS  number of products to load            (default: 1000)
    WEBSHOP_DATA_FILE     path to the items_shuffle_*.json file (default: the 1000 set)
    WEBSHOP_HUMAN_GOALS   "true"/"false" — use human-written goals (default: true)
The Lucene index directory is chosen by WebShop's engine from num_products
(1000 -> indexes_1k), built at image-build time.
"""
import os
import sys

# Add webshop to path (the repo checkout lives one dir up from search_engine).
webshop_path = os.environ.get("WEBSHOP_PATH", "/app/webshop")
if webshop_path not in sys.path:
    sys.path.insert(0, webshop_path)

from flask import Flask, request, jsonify
from web_agent_site.envs import WebAgentTextEnv

app = Flask(__name__)

# Store env instances per session.
envs = {}

# Shared server for efficiency (loads products/search engine once).
_shared_server = None

# Configuration (env-overridable).
WEBSHOP_NUM_PRODUCTS = int(os.environ.get("WEBSHOP_NUM_PRODUCTS", "1000"))
WEBSHOP_DATA_FILE = os.environ.get(
    "WEBSHOP_DATA_FILE",
    os.path.join(webshop_path, "data", "items_shuffle_1000.json"),
)
WEBSHOP_HUMAN_GOALS = os.environ.get("WEBSHOP_HUMAN_GOALS", "true").lower() == "true"


def get_shared_server():
    """Lazily initialize the shared WebShop server (loads data + index once)."""
    global _shared_server
    if _shared_server is None:
        from web_agent_site.envs.web_agent_text_env import SimServer

        print("[app_simple] Configuration:", flush=True)
        print(f"[app_simple]   num_products: {WEBSHOP_NUM_PRODUCTS}", flush=True)
        print(f"[app_simple]   data_file: {WEBSHOP_DATA_FILE}", flush=True)
        print(f"[app_simple]   human_goals: {WEBSHOP_HUMAN_GOALS}", flush=True)
        print("[app_simple] Loading products and search engine...", flush=True)

        _shared_server = SimServer(
            base_url="http://127.0.0.1:3000",
            file_path=WEBSHOP_DATA_FILE,
            human_goals=WEBSHOP_HUMAN_GOALS,
            num_products=WEBSHOP_NUM_PRODUCTS,
        )
        print(f"[app_simple] Loaded {len(_shared_server.goals)} goals", flush=True)
    return _shared_server


@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@app.route("/reset", methods=["POST"])
def reset():
    """Initialize/reset the environment for a task.

    Request JSON: {"task_id": 0, "session_id": "abc123"}
    Response JSON: {"observation", "instruction", "available_actions", "session_id"}
    """
    data = request.json
    task_id = data.get("task_id", 0)
    session_id = data.get("session_id", f"session_{task_id}")

    env = WebAgentTextEnv(
        observation_mode="text",
        server=get_shared_server(),
        human_goals=WEBSHOP_HUMAN_GOALS,
    )
    obs, _ = env.reset(session=task_id)
    envs[session_id] = env

    return jsonify({
        "observation": obs,
        "instruction": env.instruction_text,
        "available_actions": env.get_available_actions(),
        "session_id": session_id,
    })


@app.route("/step", methods=["POST"])
def step():
    """Execute an action.

    Request JSON: {"session_id": "abc123", "action": "search[blue headphones]"}
    Response JSON: {"observation", "reward", "done", "available_actions"}
    """
    data = request.json
    session_id = data.get("session_id")
    action = data.get("action")

    if session_id not in envs:
        return jsonify({"error": f"Session {session_id} not found. Call /reset first."}), 400
    if not action:
        return jsonify({"error": "action is required"}), 400

    env = envs[session_id]
    obs, reward, done, info = env.step(action)

    response = {
        "observation": obs,
        "reward": reward,
        "done": done,
        "available_actions": env.get_available_actions(),
    }
    if done:
        del envs[session_id]
    return jsonify(response)


@app.route("/get_actions", methods=["POST"])
def get_actions():
    """Get available actions for the current state.

    Request JSON: {"session_id": "abc123"}
    Response JSON: {"has_search_bar", "clickables"}
    """
    data = request.json
    session_id = data.get("session_id")
    if session_id not in envs:
        return jsonify({"error": f"Session {session_id} not found"}), 400
    env = envs[session_id]
    return jsonify(env.get_available_actions())


@app.route("/close", methods=["POST"])
def close():
    """Close a session and free resources.

    Request JSON: {"session_id": "abc123"}
    """
    data = request.json
    session_id = data.get("session_id")
    if session_id in envs:
        del envs[session_id]
        return jsonify({"status": "closed", "session_id": session_id})
    return jsonify({"status": "not_found", "session_id": session_id})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple WebShop Flask Server")
    parser.add_argument("--port", type=int, default=3000, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    print(f"[app_simple] Starting server on {args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, threaded=True)
