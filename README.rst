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

References
~~~~~~~~~~

This library follows the setup in `Nenadic and Greenacre (2005)`_.

.. Nenadic and Greenacre (2005): https://core.ac.uk/download/pdf/6591520.pdf
