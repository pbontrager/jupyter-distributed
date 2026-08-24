"""Jupyter Server extension registration and lightweight health endpoint."""

from __future__ import annotations

from typing import Any

from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join
from tornado import web

from . import __version__


class HealthHandler(APIHandler):
    """Report that the server-side control extension loaded successfully."""

    @web.authenticated
    def get(self) -> None:
        self.finish(
            {
                "ok": True,
                "extension": "spmd_jupyter",
                "version": __version__,
            }
        )


def _jupyter_server_extension_points() -> list[dict[str, str]]:
    return [{"module": "spmd_jupyter.server_extension"}]


def _load_jupyter_server_extension(server_app: Any) -> None:
    """Register HTTP routes and initialize the distributed session manager."""

    base_url = server_app.web_app.settings.get("base_url", "/")
    route = url_path_join(base_url, "spmd-jupyter", "health")
    server_app.web_app.add_handlers(".*$", [(route, HealthHandler)])
    server_app.log.info("SPMD Jupyter server extension loaded")


# Jupyter Server 1.x compatibility; current releases call the underscored hook.
load_jupyter_server_extension = _load_jupyter_server_extension
