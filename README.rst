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
