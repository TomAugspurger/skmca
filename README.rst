Use https://github.com/MaxHalford/Prince instead

skmca
-----

A scikit-learn pipeline API compatible implementation of
Multiple Correspondence Analysis (MCA).

Usage
~~~~~

.. code-block:: python

   import pandas as pd
   from skmca import MCA

   df = pd.read_csv('http://www.statoek.wiso.uni-goettingen.de/'
                    'CARME-N/download/wg93.txt',
                    sep='\t', dtype='category')
   mca = MCA()
   mca.fit(df)


Crucially, the input to ``MCA.fit`` must be a ``pandas.DataFrame``
where all the columns have a ``category`` dtype. This is necessary
to ensure that the dummy encoding of the columns is consistent across
training and test datasets.

Background
~~~~~~~~~~

MCA is like `PCA`_, but for categorical data.
You can use it to visualize high-dimensional datasets.
It can also be useful as a pre-processing step for clustering,
to avoid the curse of dimensionality.

`skmca` requires pandas and scikit-learn.

References
~~~~~~~~~~

This library follows the setup in `Nenadic and Greenacre (2005)`_.

.. PCA: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
.. Nenadic and Greenacre (2005): https://core.ac.uk/download/pdf/6591520.pdf
