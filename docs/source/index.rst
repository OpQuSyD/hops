
Welcome to TUD HOPS's documentation!
====================================

.. _notation:

Notation
--------

The whole HOPS implementation features a lot of different index
labels.  For hierarchy states, denoted by :math:`ψ`, an index of the
form :math:`ψ^{\underline{k}}` always means a hierarchy index.
Underlined quantities are matrix indices. Vector indices have vector
arrows like :math:`\vec{k}`. For more information about hierarchy
indexing see :ref:`index handling` and :ref:`hops integration`.

For all other quantities an upper index like :math:`X^{(n)}` always
labels the bath and a lower index :math:`X^{(n)}_i` the component of
the quantity.

For example, the BCF expansions can be written compactly in this form

.. math::

   α^{(n)}(t)\approx ∑_{μ=1}^{N_n} G^{(n)}_μ \exp(-W^{(n)}_μ t).

The quantities with an upper and a lower index will often be
represented by :any:`list`\ s of :any:`numpy.ndarray`\ s. The upper
index will then label the list element while the lower index labels
the element in the array.


.. toctree::
   :caption: API Docs
   :maxdepth: 3

   modules/core
   modules/utilities

.. toctree::
   :caption: Notes
   :maxdepth: 3

   pages/tutorial
   pages/running_on_zih


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
