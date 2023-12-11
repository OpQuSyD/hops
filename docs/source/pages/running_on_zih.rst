Running HOPS on the ZIH cluster
===============================

When running HOPS on the ZIH cluster, a proper environment should be
prepared.

The manual route described below contains the minimal information to
get you started. It assumes that you're already familiar with the ZIH
infrastructure.

The automatic route uses a script to prepare an environment that you
can base your efforts on. This route is recommended.

Manual Route
------------

It is advisable to read the `ZIH HPC compendium
<https://hpc-wiki.zih.tu-dresden.de/>`_ beforehand.

The basic steps are:

1. Allocating a workspace. (See `the ZIH docs
   <https://doc.zih.tu-dresden.de/data_lifecycle/workspaces/#allocate-a-workspace>`_.)

2. Installing you projct with HOPS and poetry. It is recommended to
   create a separate project and add HOPS as git dependency.

   Make sure to run

   .. code-block:: shell

      $ export TMPDIR=<SOME DIRECTORY IN YOUR SCRATCH SPACE>


   otherwise ``poetry install`` will fail because ``/tmp/`` is mounted
   with the option ``noexec`` which prevents cython extensions from
   being built.

3. Submitting `batch jobs
   <https://hpc-wiki.zih.tu-dresden.de/jobs_and_resources/overview/>`_
   that start the cluster nodes and run the actual HOPS client. See
   the `ray docs <https://docs.ray.io/en/latest/cluster/slurm.html>`_
   for more information.

   Each node should a ray worker that has access to a configured number
   of CPUs. You should check how many CPUs are available per node.

   Make sure to use ``poetry run`` to execute commands that involve python.

Automatic Route
---------------

This route assumes that you have a git repository containing you
project with HOPS declared as a git repository. Your account on the
ZIH login node must have access to the HOPS repostory as well as you
project repository. This guide provides intput and output from an
example session.

To prepare a workspace with you project setup run the
``tools/set_up_zih.sh`` script.

.. code-block:: shell-session

   s4498638@tauruslogin6:~/src/hops/tools> ./set_up_zih.sh
   You have to specify a project git repo with '-p'.
   Usage: ./set_up_zih.sh -p <PROJECT GIT REPO> [-n] [-t : 1] [-w <NAME>]

       -p <str>   Project git repo.
       -n         No interaction mode.
       -t <int>   Scratch space retention time.
       -w <str>   Scratch space/project name.

The only required option is ``-p``, but you should also think about
the others. Setting a good workspace name will make things easier.

.. code-block:: shell-session

   s4498638@tauruslogin6:~/src/hops/tools> ./set_up_zih.sh -p git@gitlab.hrz.tu-chemnitz.de:s4498638--tu-dresden.de/hops-example.git -w tutorial
   Loading Python Module: Python/3.9.6-GCCcore-11.2.0.
   You may want to adjust this in the script.
   Continue? [Yy/Nn] y
   Module Python/3.9.6-GCCcore-11.2.0 and 12 dependencies loaded.
   Allocating workspace 'tutorial'.
   Info: creating workspace.
   remaining extensions  : 2
   remaining time in days: 1
   Created /beegfs/ws/0/s4498638-tutorial
   Cloning Project from git@gitlab.hrz.tu-chemnitz.de:s4498638--tu-dresden.de/hops-example.git
   Cloning into 'project'...
   remote: Enumerating objects: 9, done.
   remote: Counting objects: 100% (9/9), done.
   remote: Compressing objects: 100% (9/9), done.
   remote: Total 9 (delta 1), reused 0 (delta 0), pack-reused 0
   Receiving objects: 100% (9/9), 32.49 KiB | 0 bytes/s, done.
   Resolving deltas: 100% (1/1), done.
   Installing the Project
   Creating virtualenv hops-ex-ujgHzsXl-py3.9 in /beegfs/ws/0/s4498638-tutorial/.cache/virtualenvs
   Installing dependencies from lock file

   Package operations: 61 installs, 0 updates, 0 removals

     • Installing numpy (1.22.2)
     • Installing pyparsing (3.0.7)

     ... a lot more packages

   A template slurm batch file can be found in /beegfs/ws/0/s4498638-tutorial/project/slurm.sh
   Dropping you into a poetry shell.
   Spawning shell within /beegfs/ws/0/s4498638-tutorial/.cache/virtualenvs/hops-ex-ujgHzsXl-py3.9
   bash-4.2$ . /beegfs/ws/0/s4498638-tutorial/.cache/virtualenvs/hops-ex-ujgHzsXl-py3.9/bin/activate
   (hops-ex-ujgHzsXl-py3.9) bash-4.2$

The script has created a scratch space with a lifetime of one day
under ``/beegfs/ws/0/s4498638-tutorial``, has cloned and installed you
project (configuring poetry to use the scratch space), created a
simple slurm batch file
``/beegfs/ws/0/s4498638-tutorial/project/slurm.sh`` and dropped you
into a ``poetry shell`` where you can interact with your poetry
environment.

The top of the slurm batch file contains the slurm configuration.

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=hops-test
   #SBATCH --tasks-per-node=1
   #SBATCH --mem-per-cpu=3GB
   #SBATCH --nodes=4
   #SBATCH --mail-type=end
   #SBATCH --mail-user=<your-mail>@tu-dresden.de
   #SBATCH --job-name=hops
   #SBATCH --output=/beegfs/ws/0/s4498638-tutorial/project/logs/out_slurm.txt
   #SBATCH --error=/beegfs/ws/0/s4498638-tutorial/project/logs/err_slurm.txt
   #SBATCH --time=00:10:00
   #SBATCH -D /beegfs/ws/0/s4498638-tutorial/project
   #SBATCH --cpus-per-task=24

You should configure ``--cpus-per-task`` to match the cpu count of one
node. Here we'll likely use the ``haswell64`` cluster partition which
gives us ``24`` threads per node (see `this
<https://doc.zih.tu-dresden.de/jobs_and_resources/partitions_and_limits/#memory-limits>`_
or more info and use ``--partition`` to choose a partition). The
``-D`` option sets the workspace to our workspace project
directory. You should set the ``--mail-user`` option if you want to be
notified via email when a job fails or ends. The ``--time`` option
sets the maximum time for your job. opFor the other options see `the
ZIH documentation
<https://hpc-wiki.zih.tu-dresden.de/jobs_and_resources/overview/>`_

After this, there follows some code to set up the cluster which you
shouldn't have to touch.

The end of the file looks like:

.. code-block:: bash

   ## EXAMPLE, CHANGE FOR YOU USE-CASE
   poetry run hi 1000 integrate --server "auto" --node-ip-address "$head_node_ip"

If you use the generic hops cli, this is what you want. Otherwise you
may want to create you own entrypoint that creates a
:any:`hops.core.integration.HOPSSupervisor` programatically. Make sure
to connect to the ray head node under the ip ``$ip_head`` (see :any:`ray.init`).

When you're ready, you can run ``sbatch slurm.sh`` and you're off to
the races. You can check the job status with ``sacct`` and cancel the
job with ``scancel <job-id>``.

.. code-block:: shell-session

   (hops-ex-ujgHzsXl-py3.9) bash-4.2$ sbatch slurm.sh
   Submitted batch job 23693083

   (hops-ex-ujgHzsXl-py3.9) bash-4.2$ sacct
          JobID    JobName  Partition    Account  AllocCPUS      State ExitCode
   ------------ ---------- ---------- ---------- ---------- ---------- --------
   23693083       tutorial haswell128    p_eflow         24    RUNNING      0:0
   23693083.ba+      batch               p_eflow         12    RUNNING      0:0
   23693083.ex+     extern               p_eflow         24    RUNNING      0:0
   23693083.0     hostname               p_eflow         12  COMPLETED      0:0
   23693083.1       poetry               p_eflow         12  COMPLETED      0:0
   23693083.2     ray-head               p_eflow         12    RUNNING      0:0
   23693083.3   ray-worke+               p_eflow         12    RUNNING      0:0

If your job is pending for a long time you can try to lower the
``cpus-per-task`` setting. You can view the output of the diffrent
processes via ``tail -f logs/*``.

This should get you going for now.
Enjoy.
