import socket


class ExitHook:
    def __init__(self, cm, hook):
        self._cm = cm
        self._hook = hook

    def __enter__(self):
        self._cm.__enter__()
        return self._cm   # note: return original

    def __exit__(self, exc_type, exc, tb):
        try:
            return self._cm.__exit__(exc_type, exc, tb)
        finally:
            self._hook(exc_type, exc, tb)

def get_free_port(start_port=6000, max_port=65535):
    """Scan from start_port upward until a free port is found."""
    for port in range(start_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                return port
            except OSError:
                continue
    raise RuntimeError("No free ports available in range!")


