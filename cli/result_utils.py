"""A small script that plots the integration result (density matrix)."""

import matplotlib

from . import common
import typer
from pathlib import Path
from hops.core.hierarchy_data import HIData, HIMetaData
import matplotlib.pyplot as plt
import click_spinner
from typing import Optional
from prettytable import PrettyTable
import os
from IPython import embed

app = typer.Typer()


multi_file_app = typer.Typer()
app.add_typer(multi_file_app, name="results")


class MultiState:
    data_name: str
    data_path: Path


@multi_file_app.command()
def list_data_files():
    """Lists all result databases."""

    results = MultiState.data_path.rglob("*.h5")
    overview = PrettyTable()
    overview.field_names = [
        "File",
        "Samples",
        "Description",
        "Size [Bytes]",
        "Time [Span, Steps]",
        "Type",
    ]
    for result_path in results:
        typer.echo(result_path)
        try:
            with HIData(str(result_path), True) as data:
                overview.add_row(
                    [
                        result_path.relative_to(os.getcwd()),
                        data.samples,
                        data.params.SysP.__non_key__["desc"],
                        result_path.stat().st_size,
                        f"{data.get_time()[0]} - {data.get_time()[-1]}, {len(data.get_time())}",
                        data.result_type,
                    ]
                )
        except:
            overview.add_row(
                [
                    result_path.relative_to(os.getcwd()),
                    "unreadable",
                    "unreadable",
                    result_path.stat().st_size,
                    "unreadable",
                    "unreadable",
                ]
            )
    typer.echo(overview)


data_file_app = typer.Typer()
app.add_typer(data_file_app, name="result")


class DataFileState:
    file: Path


@data_file_app.callback()
def data_file_entry(
    file: Path = typer.Argument(
        ...,
        help="The path of the HDF5 file",
        dir_okay=False,
        exists=True,
        file_okay=True,
    )
):
    """Utilities for working with a single HOPS result database."""
    DataFileState.file = file


@data_file_app.command()
def print_config():
    """
    Prints the HOPS config (HIParams) of the contained in the `file`.
    """
    typer.echo(HIData(str(DataFileState.file), True).params)


@multi_file_app.callback()
def entry(
    data_name: str = common.data_name_opt,
    data_path: Path = common.data_path_opt,
):
    """
    Utilities for working with the multiple result databases in
    `data-path/data-name`.
    """

    MultiState.data_name = data_name
    MultiState.data_path = data_path


@multi_file_app.command()
def current_data_file(config: Path = common.config_opt):
    """
    Prints the bath of the data file corresponding to the specified configuration.
    """
    params = common.load_config(config)

    with HIMetaData(
        hid_name=MultiState.data_name, hid_path=str(MultiState.data_path)
    ).get_HIData(params, True) as data:
        typer.echo(Path(data.hdf5_name).relative_to(os.getcwd()))


@data_file_app.command()
def plot_results(
    out_file: Optional[Path] = typer.Argument(
        None,
        help="A file into which the plot is written.",
        dir_okay=False,
        file_okay=True,
    ),
    no_gui: bool = typer.Option(False, help="Do not show a gui window with the plot."),
):
    """Plot the density matrix elements of the result from the database."""

    with HIData(str(DataFileState.file), True) as data:
        if data.samples == 0:
            typer.echo("No samples found.", err=True)

        typer.echo(f"Found N={data.samples} samples.")
        typer.echo("Plotting... ", nl=False)

        with click_spinner.spinner():  # type: ignore
            plt.figure(figsize=(15, 10))
            ax_mean = plt.subplot(221)
            ax_std = plt.subplot(222)

            ρ = data.rho_t_accum.mean
            ρ_std = data.rho_t_accum.ensemble_std

            for i in range(ρ.shape[1]):
                for j in range(ρ.shape[1]):
                    ax_mean.plot(
                        data.get_time(),
                        ρ[:, i, j].real,
                        label=rf"$\Re\rho_{{{i},{j}}}$",
                    )

                    ax_mean.plot(
                        data.get_time(),
                        ρ[:, i, j].imag,
                        label=rf"$\Im\rho_{{{i},{j}}}$",
                    )

                    ax_std.plot(
                        data.get_time(),
                        ρ_std[:, i, j],
                        label=rf"$\sigma[\rho_{{{i},{j}}}]$",
                    )

            ax_mean.legend()
            ax_std.legend()

        if not no_gui:
            matplotlib.use("gtk3agg")
            plt.show()

        if out_file:
            plt.savefig(out_file)
            typer.secho(f"Wrote the plot to: {out_file}", fg="green")


@data_file_app.command()
def repl(
    read_only: bool = typer.Option(
        True, help="Whether to open the data file in read-only mode."
    )
):
    """
    Opens a repl with the data file opened and assigned to the `data`
    variable as a `HIData` instance.
    """

    with HIData(str(DataFileState.file), read_only) as data:
        typer.echo(
            f"Opened data file {DataFileState.file} as a HIData instance assigned to `data`."
        )
        embed()


def main():
    app()


if __name__ == "__main__":
    main()
