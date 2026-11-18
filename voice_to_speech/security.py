"""Loopback HTTP and WebSocket request checks."""

from __future__ import annotations


class LocalOnlyMiddleware:
    def __init__(self, app, port: int):
        self.app = app
        self.port = port
        self.hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        self.origins = {f"http://{host}" for host in self.hosts}

    async def __call__(self, scope, receive, send):
        kind = scope["type"]
        if kind not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        pairs = [(key.decode("latin-1").lower(), value.decode("latin-1"))
                 for key, value in scope.get("headers", [])]
        headers = dict(pairs)
        host = headers.get("host", "")
        origin = headers.get("origin")
        method = scope.get("method", "GET").upper()
        invalid = sum(key == "host" for key, _ in pairs) != 1 or sum(key == "origin" for key, _ in pairs) > 1 or host not in self.hosts or (origin is not None and
            (origin not in self.origins) and (kind == "websocket" or method not in ("GET", "HEAD")))
        if invalid:
            if kind == "websocket":
                await send({"type": "websocket.close", "code": 4403})
            else:
                body = "Доступ заборонено.".encode("utf-8")
                await send({"type": "http.response.start", "status": 403, "headers": [
                    (b"content-type", b"text/plain; charset=utf-8"),
                    (b"content-length", str(len(body)).encode())]})
                await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)
