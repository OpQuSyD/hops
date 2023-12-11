"""Common functionality for the cli utilities."""

from pathlib import Path
from hops.core.hierarchy_parameters import HIParams

from importlib.abc import Loader
from importlib.machinery import ModuleSpec
import importlib.util
import typer
import os
import stocproc
import logging

stocproc.logging_setup(
    logging.WARNING, logging.WARNING, logging.WARNING, logging.WARNING
)


def load_config(path: Path) -> HIParams:
    """Load the configuration from a python module under ``path``."""
    spec = importlib.util.spec_from_file_location("config", path)

    if not isinstance(spec, ModuleSpec) or not isinstance(spec.loader, Loader):
        typer.echo(f"Invalid configuration file specified.", err=True)
        exit(-1)

    config_module = importlib.util.module_from_spec(spec)

    typer.secho(
        f"Loading the configuration from {path}. This might take a while...",
        fg=typer.colors.BRIGHT_BLUE,
        err=True,
    )

    spec.loader.exec_module(config_module)

    if not hasattr(config_module, "params") or not isinstance(
        config_module.params, HIParams
    ):
        typer.echo(
            f"The config must export a HIParams instance called `params`.", err=True
        )
        exit(-1)

    typer.echo()
    return config_module.params


config_opt: Path = typer.Option(
    Path("./config.py"),
    help="The config module that exports a configured hierarchy.",
    dir_okay=False,
    exists=True,
    file_okay=True,
)
data_name_opt: str = typer.Option(
    "data", help="The name under which to store the results."
)

data_path_opt: Path = typer.Option(
    Path(os.getcwd()),
    help="The path under which to store the results.",
    dir_okay=True,
    exists=True,
    file_okay=False,
)
