# HOPS
Solving Open Quantum System Dynamics with the Hierarchy of Pure States
(HOPS).

This readme is somewhat inadequate and needs some love.  The sphinx
documentation (see next section for an build instructions) is the
place to consult for usage instructions. The API documentation in
there is pretty complete but it lacks on the side of high-level
explaintations.

# Rules for Developers :P
Keeping the code of modestly good quality requires some discipline on
the side of the developers. If code is to be merged into the `main`
branch, it should preferably fullfill the requirements detailed below.

## Workflow
For every major change a new branch is created (either here or in a
forked repo).  As soon as you want to share you work (which hasn't to
be finished yet), you can create a merge request (optionally marked as
work-in-progress). This allows the CI to check for common problems and
other developers to give you hints.

When the CI is happy and you're ready, the code can be merged.

*Test coverage should not decrease*.

After merging the development branch should be deleted if it was
created in this repo.

## Style
We use [black](https://black.readthedocs.io/en/stable/).  No
compromises :P.

## Documentation
Every function (yes also the small ones) gets a docstring detailing
what it does (not how it does it) and what the arguments mean.

Each public property of a class shall be documented. The initializer
is documented along with the class in the class docstring.

The format is restructured text in sphinx style. Check you editor for
nice highlighting and formatting support.

See [this
page](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html)
for a tutorial and example.

The docs for the latest commit can be downloaded
[here](https://gitlab.hrz.tu-chemnitz.de/api/v4/projects/10157/jobs/artifacts/master/download?job=docs).

## Typing
Python type hints make code very much easier to reason about (although
they can be a pain in the neck).

The rule is: type annotate *everything*. No exceptions. (Except when
there is no other way.)

You encounter experiences similar to
 - the code working on the first try.
 - you editor knowing what the heck that variable is and where it
   comes from and which attributes it has.
 - the realization that the code couldn't possibly work that way.
 - the thing you're currently doing being *evil*.

As type hints are still evolving so you should begin with reading
 - https://www.python.org/dev/peps/pep-0484/
and then look at
 - https://www.python.org/dev/peps/pep-0585/ for updates.

Or just google `python type hints in python 3.9, 3.10`.

A rule of thumb is: if you import from the `typing` module, check if
there is an altrenative. (For example `typing.Callable ->
collections.abc.Callable`.)

In python 3.11 we will be able to type-annotate the numpy array
dimensions. Yay!

## Abstraction Zen

If you implement new functionality, then ask yourself

 - if it fits within the scope of the existing code or should be a
   project of its own.
 - if other parts of the code can use this functionality later or right now.
 - if this functionality has to be useful to other parts in the code later on.
 - if the functionality can be implemented by generalizing existing code.
 - how you can use existing code.
 - how you can implement the functionality with minimal impact on of
   the existing code.
 - how you can implement the functionality without changing the
   semantics of the existing code.

# Installation

## Poetry
Installing works as usual with `poetry install`.  For development use
`poetry shell` and for installing just add this repo to the depencies
of your project. You can also build a standalone package with `poetry
build`.


## Nix
For developing use `nix develop` and for installing use the default
package from the flake.

If you plan to use this package in another poetry2nix project you have
to include the overrides from `lib.overrides` in the flake
`github:vale981/hiro-flake-utils`.

# Usage
## Running Tests
You can run tests by executing `pytest` in the terminal. You may want
to disable the rather slow coverage analysis with `pytest
--no-cov`. If you want to execute the **very** slow tests as well you
may run `pytest --run-optional-tests=slow`.

## CLI
The scripts in the `cli` directory are being made available as
executables by poetry (or poetry2nix).

They are self-documenting via the `--help` flag.  Check out the
`examples` directory as well.

The `hi` script starts hops integration and the `result-utils` helps
you with managing the integration result database files.

The configuration file has to export a global variale that contains a
`HIParams` instance. See the sphinx docs for information on what it
contains.  The rest is up to you.

These scripts are by no means the exclusive entrypoint for the HOPS
code, but most convenient for the generic use case. The programatic
interface (see `HOPSSupervisor`) is also well supported. In fact, the
`cli` scripts are so simple that you should have a look at them as
examples anyway (see the `cli` folder).
