# About this Example
This is the simplest use case of HOPS. One bath and a qubit system.



# Integrate HOPS

There are several ways to trigger the solver.
Either the python interface (see documentation about `hops.core.integration.HOPSSupervisor`)
or the command line interface with a configuration file can be used.

The `cli` accepts a `--help` argument and is self-documenting.
In the following, we integrate`10` trajectories. It requires a configuration
module that exports an instance of `HIParams` named `params`.

## Simple Integration (for debugging only)

Sequential integration, i.e., a single stochastic pure state at each
time in one process:

```shell
$ hi --config config.py 10 integrate-single-process
```

## Local Integration

Integrate the HOPS equations in parrallel:

```shell
$ hi --config config.py 10 integrate
```

## Distributed Integration on Multiple Nodes

Set up a ray cluster with hops installed.

Then, the computation can be started with.
```shell
$ hi --config config.py 10 --server ray://<ip>:<port>
```

See the output of the ray cluster start command (the part about the
client). The `<port>` is most likely `10001`.


# Access to the HOPS Data

The easiest way to access the data is via `hops.core.hierarchy_data.HIMetaData ` and the config file
assumed to be importable as `config`.

```python
from config import params
from hops.core.hierarchy_data import HIMetaData

with hid.HIMetaData(hid_name="data", hid_path=".").get_data(params) as data:
    ...
```

See the docs for the `hierarchy_data` module for further information.
