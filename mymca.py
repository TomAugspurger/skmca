# TODO: do 3 point scale
import numpy as np
import pandas as pd
from scipy.linalg import diagsvd
from sklearn.base import BaseEstimator


class MCA(BaseEstimator):

    def __init__(self, benzecri_correction=True, method='indicator',
                 n_components=None):
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
        self.Q_ = df.shape[1]
        self.J_ = X.shape[1]

        if self.method == 'indicator':
            result = self._fit_indicator(X)
        elif self.method == 'burt':
            result = self._fit_burt(X)
        else:
            raise TypeError
        self._validate_fit()
        return result

    def _fit_burt(self, Z: np.array):

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
        v = v.T                     # Symmetrical, but w/e

        A = v / cm
        F = A * s

        inertia = np.sum(s**2)

        self.J_ = J
        self.Σ_ = Σ
        self.F_ = F
        self.inertia_ = inertia
        self.σ_adj_ = self.inertia_adjustment(s, self.Q_)
        self.cm_ = self.rm_ = cm

    def _fit_indicator(self, Z: np.array):
        J = Z.shape[1]                 # Total number of levels
        P = Z / Z.sum()

        # Marginals
        cm = P.sum(0)                  # col margin
        rm = P.sum(1)                  # row margin

        # Residual (TODO: Verify)
        eP = np.outer(rm, cm)
        S = (P - eP) / np.sqrt(eP)

        # we match python mca thru here...
        # TODO: Verify full_matricies
        u, s, v = np.linalg.svd(S, full_matrices=False)
        # urghhhhhhhhhhhhhhhhhhhhhhhh
        # so s matches R's mca$svd$vs
        s2 = s ** 2

        # TODO: Where to adjust s?
        if self.benzecri_correction:
            s = self.adjust_inertia(s, self.Q_)
        λ = s**2
        expl = λ / λ.sum()

        # this is broken, maybe also em/rm and the retval of svd
        b = v / np.sqrt(cm)
        g = b.T * np.sqrt(λ)

        self.u_ = u
        self.s_ = s
        self.s2_ = s2
        self.v_ = v
        self.λ_ = λ
        self.b_ = b
        self.g_ = g
        self.expl_ = expl
        self.J = J

    def transform(self, X, y=None):
        # TODO: verify!
        return pd.get_dummies(X).values @ self.v_[:, :self.n_components]

    @staticmethod
    def adjust_inertia(σ, Q):
        σ_ = σ.copy()
        mask = σ_ >= 1 / Q
        σ_[mask] = ((Q / (Q - 1)) * (σ_[mask] - 1 / Q)) ** 2
        σ[~mask] = 0
        return σ_

    def _validate_fit(self):
        need = [
            'u_',
            's_',
            'v_',
            'λ_',
            'b_',
            'g_',
            'expl_',
        ]
        have = dir(self)

        for attr in need:
            assert attr in have

class MCA2(object):
    """Run MCA on selected columns of a pd DataFrame.

    If the column are specified, assume that they hold
    categorical variables that need to be replaced with
    dummy indicators, otherwise process the DataFrame as is.

    'cols': The columns of the DataFrame to process.
    'ncols': The number of columns before dummy coding. To be passed if cols isn't.
    'benzecri': Perform Benzécri correction (default: True)
    'TOL': value below which to round eigenvalues to zero (default: 1e-4)
    """

    def __init__(self, ncols=None, benzecri=True, TOL=1e-4):
        self.ncols = ncols
        self.benzecri = benzecri
        self.TOL = TOL

    def fit(self, X):
        K = X.shape[1]
        X = pd.get_dummies(X).values
        J = X.shape[1]

        S = X.sum()
        Z = X / S  # correspondence matrix
        r = Z.sum(axis=1)
        c = Z.sum()
        D_r = np.diag(1 / np.sqrt(r))
        Z_c = Z - np.outer(self.r, c)  # standardized residuals matrix
        D_c = np.diag(1 / np.sqrt(c))

        # another option, not pursued here, is sklearn.decomposition.TruncatedSVD
        P, s, Q = np.linalg.svd(D_r @ Z_c @ D_c)

        E = self._benzecri() if self.benzecri else s**2
        inertia = sum(E)
        rank = np.argmax(E < self.TOL)
        L = E[:rank]

        self.X = X
        self.S = S
        self.Z = Z
        self.K = K
        self.J = J
        self.inertia = inertia
        self.L = L

    def _benzecri(self):
        if self.E is None:
            self.E = np.array([(self.K / (self.K - 1.) * (_ - 1. / self.K))**2
                              if _ > 1. / self.K else 0 for _ in self.s**2])
        return self.E

    def fs_r(self, percent=0.9, N=None):
        """Get the row factor scores (dimensionality-reduced representation),
        choosing how many factors to retain, directly or based on the explained
        variance.

        'percent': The minimum variance that the retained factors are required
                                to explain (default: 90% = 0.9)
        'N': The number of factors to retain. Overrides 'percent'.
                If the rank is less than N, N is ignored.
        """
        if not 0 <= percent <= 1:
                raise ValueError("Percent should be a real number between 0 and 1.")
        if N:
                if not isinstance(N, (int, np.int64)) or N <= 0:
                        raise ValueError("N should be a positive integer.")
                N = min(N, self.rank)
                # S = np.zeros((self._numitems, N))
        # else:
        self.k = 1 + np.flatnonzero(np.cumsum(self.L) >= sum(self.L) * percent)[0]
        #  S = np.zeros((self._numitems, self.k))
        # the sign of the square root can be either way; singular value vs. eigenvalue
        # np.fill_diagonal(S, -np.sqrt(self.E) if self.cor else self.s)
        num2ret = N if N else self.k
        s = -np.sqrt(self.L) if self.cor else self.s
        S = diagsvd(s[:num2ret], self._numitems, num2ret)
        self.F = self.D_r @ self.P @ S
        return self.F

    def fs_c(self, percent=0.9, N=None):
        """Get the column factor scores (dimensionality-reduced representation),
        choosing how many factors to retain, directly or based on the explained
        variance.

        'percent': The minimum variance that the retained factors are required
                                to explain (default: 90% = 0.9)
        'N': The number of factors to retain. Overrides 'percent'.
                If the rank is less than N, N is ignored.
        """
        if not 0 <= percent <= 1:
            raise ValueError("Percent should be a real number between 0 and 1.")
        if N:
            if not isinstance(N, (int, np.int64)) or N <= 0:
                raise ValueError("N should be a positive integer.")
            N = min(N, self.rank)  # maybe we should notify the user?
        # else:
        self.k = 1 + np.flatnonzero(np.cumsum(self.L) >= sum(self.L) * percent)[0]
        #  S = np.zeros((self._numitems, self.k))
        # the sign of the square root can be either way; singular value vs. eigenvalue
        # np.fill_diagonal(S, -np.sqrt(self.E) if self.cor else self.s)
        num2ret = N if N else self.k
        s = -np.sqrt(self.L) if self.cor else self.s
        S = diagsvd(s[:num2ret], len(self.Q), num2ret)
        self.G = self.D_c @ self.Q.T @ S  # important! note the transpose on Q
        return self.G

    def cos_r(self, N=None):  # percent=0.9
        """Return the squared cosines for each row."""

        if not hasattr(self, 'F') or self.F.shape[1] < self.rank:
                self.fs_r(N=self.rank)  # generate F
        self.dr = np.linalg.norm(self.F, axis=1)**2
        # cheaper than np.diag(self.F.dot(self.F.T))?

        return np.apply_along_axis(lambda _: _ / self.dr, 0, self.F[:, :N]**2)

    def cos_c(self, N=None):  # percent=0.9,
        """Return the squared cosines for each column."""

        if not hasattr(self, 'G') or self.G.shape[1] < self.rank:
                self.fs_c(N=self.rank)  # generate
        self.dc = np.linalg.norm(self.G, axis=1)**2
        # cheaper than np.diag(self.G.dot(self.G.T))?

        return np.apply_along_axis(lambda _: _ / self.dc, 0, self.G[:, :N]**2)

    def cont_r(self, percent=0.9, N=None):
        """Return the contribution of each row."""

        if not hasattr(self, 'F'):
            self.fs_r(N=self.rank)  # generate F
        return np.apply_along_axis(lambda _: _ / self.L[:N], 1,
                                   np.apply_along_axis(lambda _: _ * self.r, 0, self.F[:, :N]**2))

    def cont_c(self, percent=0.9, N=None):  # bug? check axis number 0 vs 1 here
        """Return the contribution of each column."""

        if not hasattr(self, 'G'):
            self.fs_c(N=self.rank)  # generate G
        return np.apply_along_axis(lambda _: _ / self.L[:N], 1,
                                   np.apply_along_axis(lambda _: _ * self.c, 0, self.G[:, :N]**2))

    def expl_var(self, greenacre=True, N=None):
        """
        Return proportion of explained inertia (variance) for each factor.

        :param greenacre: Perform Greenacre correction (default: True)
        """
        if greenacre:
            greenacre_inertia = (self.K / (self.K - 1.) * (sum(self.s**4) -
                                 (self.J - self.K) / self.K**2.))
            return (self._benzecri() / greenacre_inertia)[:N]
        else:
            E = self._benzecri() if self.cor else self.s**2
            return (E / sum(E))[:N]

    def fs_r_sup(self, DF, N=None):
        """Find the supplementary row factor scores.

        ncols: The number of singular vectors to retain.
        If both are passed, cols is given preference.
        """
        if not hasattr(self, 'G'):
            self.fs_c(N=self.rank)  # generate G

        if N and (not isinstance(N, int) or N <= 0):
                raise ValueError("ncols should be a positive integer.")
        s = -np.sqrt(self.E) if self.cor else self.s
        N = min(N, self.rank) if N else self.rank
        S_inv = diagsvd(-1 / s[:N], len(self.G.T), N)
        # S = scipy.linalg.diagsvd(s[:N], len(self.tau), N)
        return (DF.div(DF.sum(1), 0).values @ self.G @ S_inv)[:, :N]

    def fs_c_sup(self, DF, N=None):
        """Find the supplementary column factor scores.

        ncols: The number of singular vectors to retain.
        If both are passed, cols is given preference.
        """
        if not hasattr(self, 'F'):
            self.fs_r(N=self.rank)  # generate F

        if N and (not isinstance(N, int) or N <= 0):
                raise ValueError("ncols should be a positive integer.")
        s = -np.sqrt(self.E) if self.cor else self.s
        N = min(N, self.rank) if N else self.rank
        S_inv = diagsvd(-1 / s[:N], len(self.F.T), N)
        # S = scipy.linalg.diagsvd(s[:N], len(self.tau), N)
        return (DF / DF.sum().values.T @ self.F @ S_inv)[:, :N]

