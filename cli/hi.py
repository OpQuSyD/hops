"""A cli wrapper for the HI class to integrate locally and start a server.

Clients may be started with the client script.
"""

from enum import Enum
import typer
from typing import Optional
from hops.core.integration import HOPSSupervisor
from hops.util.logging_setup import logging_setup
import logging
from pathlib import Path
from . import common
import ray

app = typer.Typer()


class State:
    clear_pd: bool
    hierarchy: HOPSSupervisor


@app.command()
def integrate(
    server: Optional[str] = typer.Option(
        None,
        help="""The server and port to connect to.  This is passed straight
        to ray init.
        """,
    ),
    node_ip_address: Optional[str] = typer.Option(
        None,
        help="""The ip address of the current node.  This may be
             helpful if there is already a head node running on the
             current node.
             """,
    ),
):
    """Integrate the HOPS equations using multiple processes in a ray
    cluster.

    Such a cluster is automatically started on the local machine if
    `server` is `None`.  Otherwise the `server` argument can be
    specified to connect to a ray cluster.  You most like want to use
    the ray client by passing `ray://<ip>:10001`.
    """

    ray.init(address=server, _node_ip_address=node_ip_address or "127.0.0.1")
    State.hierarchy.integrate(State.clear_pd)


@app.command()
def integrate_single_process():
    """Integrate the HOPS equations in a single process.  For a single
    machine this might well be faster (scipy uses multithreading) than
    the ray cluster integration.

    This is useful for debugging purposes.
    """

    State.hierarchy.integrate_single_process(State.clear_pd)


levels = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


class LogLevels(str, Enum):
    critical = "critical"
    error = "error"
    warning = "warning"
    info = "info"
    debug = "debug"


@app.callback()
def entry(
    num_samples: int = typer.Argument(
        ..., help="How many samples are to be calculated."
    ),
    config: Path = common.config_opt,
    data_name: str = common.data_name_opt,
    data_path: Path = common.data_path_opt,
    minimum_index: int = typer.Option(
        0, help="The sample smallest index to begin with."
    ),
    clear_pd: bool = typer.Option(False, help="Whether to clear the result database."),
    hide_progress: bool = typer.Option(
        False,
        help="""Whether to hide the progress bar.
        Note that the progress bar is usually clever enough to hide itself.""",
    ),
    log_level: LogLevels = typer.Option(
        LogLevels.warning,
        help=f"The log level for the application.",
    ),
    show_stocproc_logs: bool = typer.Option(
        False, help="Whether to show the logs of the stochastic process generation."
    ),
):
    """The HOPS integration command line interface."""

    level = levels[log_level]
    logging_setup(level, show_stocproc_logs)

    State.hierarchy = HOPSSupervisor(
        common.load_config(config),
        num_samples,
        data_name,
        str(data_path),
        minimum_index,
        hide_progress,
    )

    State.clear_pd = clear_pd


def main():
    app()
    ray.shutdown()


if __name__ == "__main__":
    main()
