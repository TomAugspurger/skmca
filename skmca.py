# TODO: do 3 point scale
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class MCA(BaseEstimator):
    r"""
    Multiple Correspondence Analysis

    Parameters
    ----------

    method : {'indicator', 'burt'}
    n_components : int
    benzecri_correction : bool

    Attributes
    ----------

    TODO

    Notes
    -----
    We use the notation of Nenadic and Greenacre (2005)

    @article{nenadic2005computation,
      title={Computation of multiple correspondence analysis, with code in R},
      author={Nenadic, Oleg and Greenacre, Michael},
      year={2005},
      publisher={UPF Working Paper}
    }
    """
    def __init__(self, method='indicator', n_components=None,
                 benzecri_correction=True):
        self.benzecri_correction = benzecri_correction
        self.method = method
        self.n_components = n_components

    @property
    def method(self):
        """
        Matrix to do computations on `{'indicator', 'burt'}`
        """
        return self._method

    @method.setter
    def method(self, method):
        allowed = ['burt', 'indicator']
        if method not in allowed:
            raise TypeError(allowed)
        self._method = method

    def fit(self, X, y=None):
        """
        ``X`` should be a DataFrame of Categoricals.
        """
        df = X.copy()
        X = pd.get_dummies(df).values
        self.I_, self.Q_ = df.shape

        if self.method == 'indicator':
            result = self._fit_indicator(X)
        elif self.method == 'burt':
            result = self._fit_burt(X)
        else:
            raise TypeError
        return result

    def _fit_burt(self, Z: np.array):
        raise ValueError("Haven't verified this yet")

        J = Z.shape[1]                 # Total number of levels
        C = Z.T @ Z                    # Burt matrix

        P = C / C.sum()                # Correspondence matrix
        r = P.sum(1)                   # equals row and column masses

        # Marginals
        cm = np.outer(r, r)

        # Residual (TODO verify)
        S = (P - cm) / np.sqrt(cm)

        u, s, v = np.linalg.svd(S)  # paper seems to claim that U = V.T? Typo?
        Σ = np.diag(s)

        A = v / cm
        F = A * s

        inertia = np.sum(s**2)

        self.J_ = J
        self.Σ_ = Σ
        self.F_ = F
        self.inertia_ = inertia
        self.σ_adj_ = self.adjust_inertia(s, self.Q_)
        self.cm_ = self.rm_ = cm

        self.s_ = s
        self.u_ = u
        self.expl_ = self.b_ = self.g_ = self.v_ = self.λ_ = None

    def _fit_indicator(self, Z: np.array):
        Q = self.Q_
        J = Z.shape[1]
        N = self.n_components if self.n_components is not None else J - Q

        P = Z / Z.sum()
        cm = P.sum(0)
        rm = P.sum(1)
        eP = np.outer(rm, cm)
        S = (P - eP) / np.sqrt(eP)

        u, s, v = np.linalg.svd(S, full_matrices=False)

        lam = s[:N]**2
        expl = lam / lam.sum()

        b = (v / np.sqrt(cm))[:N]  # colcoord
        g = (b.T * np.sqrt(lam)).T        # colpcoord

        u_red = u[:, :N]

        f = ((u_red * np.sqrt(lam)).T / np.sqrt(rm)).T
        a = f / np.sqrt(lam)

        self.u_ = u
        self.s_ = s
        self.v_ = v
        self.b_ = b
        self.g_ = g
        self.expl_ = expl
        self.J_ = J
        self.Z_ = Z
        self.P_ = P
        self.cm_ = cm
        self.rm_ = rm
        self.lam_ = lam
        self.f_ = f
        self.a_ = a

    def transform(self, X, y=None):
        """
        Transform the DataFrame `X` into the reduced space.

        Parameters
        -----------
        X : DataFrame

            .. warning::

            This *must* have the same categories as the original
            data

        y : object
            Ignored. For compatability with scikit-learn pipelines

        Returns
        -------
        trn : numpy.array
            Has shape (N x K) where N is the number of observations
            and K is the minimum of `self.n_components` or `J - Q`
            where `Q` is the number of columns in the original DataFrame
            and `J` is the number of columns in the dummy-encoded DataFrame,
            the total number of levels. If `self.n_compoments` is None,
            the default, then `J - Q` is used.
        """
        # TODO: Verify!
        return pd.get_dummies(X).values @ self.v_[:, :self.n_components]

    @staticmethod
    def adjust_inertia(σ, Q):
        σ_ = σ.copy()
        mask = σ_ >= 1 / Q
        σ_[mask] = ((Q / (Q - 1)) * (σ_[mask] - 1 / Q)) ** 2
        σ[~mask] = 0
        return σ_
