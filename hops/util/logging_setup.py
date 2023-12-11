"""A small collection of utilities to configure a nice logging output for hops.

It is currently being used by the CLI and custom client code.
"""


import coloredlogs
import stocproc


def logging_setup(level: int, show_stocproc: bool = False):
    """
    Installs colored logging via the ``coloredlogs`` module for the
    logging level ``level``.

    :param level: The :any:`logging` level.
    :param show_stocproc: Whether to show logs from :any:`stocproc`.
    """

    if show_stocproc:
        stocproc.logging_setup(level, level, level, level)
    else:
        stocproc.logging_setup(level, 99, 99, 99)

    coloredlogs.install(
        level=level,
        fmt="[%(levelname)-7s %(name)-25s %(process)d] %(message)s",
    )
