"""Jupyter Server extension and distributed-kernel lifecycle API."""

from __future__ import annotations

from typing import Any

from jupyter_server.auth.decorator import authorized
from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join
from tornado import web

from . import __version__
from .coordinator import DistributedKernelCoordinator


class HealthHandler(APIHandler):
    """Report that the server-side control extension loaded successfully."""

    @web.authenticated
    def get(self) -> None:
        self.finish(
            {
                "ok": True,
                "extension": "jupyter_distributed",
                "version": __version__,
            }
        )


class KernelWorldSizeHandler(APIHandler):
    """Inspect or change the process count for an ordinary kernel session."""

    auth_resource = "kernels"

    def initialize(self, coordinator: DistributedKernelCoordinator) -> None:
        self.coordinator = coordinator

    @web.authenticated
    @authorized
    def get(self, kernel_id: str) -> None:
        try:
            model = self.coordinator.describe(kernel_id)
        except KeyError as error:
            raise web.HTTPError(404, f"Kernel does not exist: {kernel_id}") from error
        self.finish(model)

    @web.authenticated
    @authorized
    async def post(self, kernel_id: str) -> None:
        body = self.get_json_body() or {}
        try:
            world_size = body["world_size"]
            model = await self.coordinator.set_world_size(kernel_id, world_size)
        except KeyError as error:
            if error.args and error.args[0] == "world_size":
                raise web.HTTPError(400, "world_size is required") from error
            raise web.HTTPError(404, f"Kernel does not exist: {kernel_id}") from error
        except (TypeError, ValueError) as error:
            raise web.HTTPError(400, str(error)) from error
        self.finish(model)


def _jupyter_server_extension_points() -> list[dict[str, str]]:
    return [{"module": "jupyter_distributed.server_extension"}]


def _load_jupyter_server_extension(server_app: Any) -> None:
    """Register the server-side integration routes."""

    base_url = server_app.web_app.settings.get("base_url", "/")
    coordinator = DistributedKernelCoordinator(server_app.kernel_manager)
    server_app.web_app.settings["jupyter_distributed_coordinator"] = coordinator
    health_route = url_path_join(base_url, "jupyter-distributed", "health")
    kernel_route = url_path_join(
        base_url,
        "jupyter-distributed",
        "kernels",
        r"(?P<kernel_id>[A-Za-z0-9-]+)",
    )
    server_app.web_app.add_handlers(
        ".*$",
        [
            (health_route, HealthHandler),
            (kernel_route, KernelWorldSizeHandler, {"coordinator": coordinator}),
        ],
    )
    server_app.log.info("Jupyter Distributed server extension loaded")


# Jupyter Server 1.x compatibility; current releases call the underscored hook.
load_jupyter_server_extension = _load_jupyter_server_extension
