from jupyterlab.galata import configure_jupyter_server

configure_jupyter_server(c)  # noqa: F821
c.ServerApp.ip = "127.0.0.1"  # noqa: F821
c.ServerApp.port = 9988  # noqa: F821
c.ServerApp.port_retries = 0  # noqa: F821
