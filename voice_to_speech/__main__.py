"""Run the loopback-only transcription server."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys

import uvicorn

from .config import Config
from .server import create_app, socket_activation_fd


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Локальний сервер транскрипції")
    try:
        default_port = int(os.environ.get("VTS_PORT", "8765"))
    except ValueError:
        parser.error("VTS_PORT має бути номером порту.")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--open", action="store_true", help="Відкрити сторінку після запуску")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Порт має бути від 1 до 65535.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    inherited_fd = socket_activation_fd(os.environ, os.getpid())
    if inherited_fd is not None:
        listening_socket = socket.socket(fileno=inherited_fd)
    else:
        listening_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listening_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listening_socket.bind(("127.0.0.1", args.port))
            listening_socket.listen(128)
        except OSError:
            logging.error("Порт %s зайнятий. Сервер, імовірно, вже працює: http://127.0.0.1:%s/", args.port, args.port)
            listening_socket.close()
            return 1
    config = Config(port=args.port)
    app = create_app(config, open_browser=args.open)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="info", lifespan="on"))
    app.state.vts.uvicorn_server = server
    try:
        server.run(sockets=[listening_socket])
    finally:
        listening_socket.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
