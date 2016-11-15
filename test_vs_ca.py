import os

import numpy as np
import pandas as pd
import pytest
import skmca

HERE = os.path.dirname(__file__)


@pytest.fixture
def data():
    """The WG93 dataset of just the categoricals [871x4]"""
    df = pd.read_csv(
        os.path.join(HERE, 'data', 'wg93.txt'),
        sep='\t',
        dtype='category',
        usecols=['A', 'B', 'C', 'D'])
    return df


class TestMCA:
    @staticmethod
    def check_attrs(mca):
        need = [
            'u_',
            's_',
            'v_',
            'b_',
            'g_',
            'expl_',
        ]
        have = dir(mca)
        missing = set(need) - set(have)
        if missing:
            raise AssertionError("Missing {}".format(', '.join(missing)))

    @pytest.mark.parametrize('method', ['indicator', 'burt'])
    def test_fit(self, data, method):
        mca = skmca.MCA(method=method)
        mca.fit(data)
        self.check_attrs(mca)

    def test_values(self, data):
        mca = skmca.MCA()
        mca.fit(data)

        assert mca.Z_.shape == (871, 20)
        assert mca.J_ == 20
        assert mca.Q_ == 4
        assert mca.I_ == 871

        eP0 = np.array([
            0, 0.0002870264, 0.0000000000, 0, 0, 0, 0, 0.0002870264,
            0.0000000000, 0, 0, 0.0000000000, 0, 0.0002870264, 0, 0, 0,
            0.0002870264, 0, 0
        ])
        assert np.allclose(mca.P_[0], eP0)
        ecm = np.array([
            0.03415614, 0.09242250, 0.05855339, 0.05109070, 0.01377727,
            0.02037887, 0.04994259, 0.05884041, 0.08065442, 0.04018370,
            0.04362801, 0.09070034, 0.05654420, 0.04420207, 0.01492537,
            0.01722158, 0.06659013, 0.05797933, 0.06486797, 0.04334099
        ])
        assert np.allclose(mca.cm_, ecm)
        assert np.allclose(mca.rm_[0], 0.001148106)
        assert len(mca.rm_) == 871

        e_s = np.array([
            6.762981e-01, 6.564798e-01, 5.673850e-01, 5.536002e-01,
            5.250473e-01, 5.019243e-01, 4.925029e-01, 4.847171e-01,
            4.748347e-01, 4.697117e-01, 4.580803e-01, 4.440140e-01,
            4.217622e-01, 4.112322e-01, 3.909208e-01, 3.539014e-01,
            1.087841e-15, 5.147218e-16, 3.379731e-16, 2.542894e-16
        ])
        assert np.allclose(mca.s_, e_s)

        e_lam = np.array([
            .4573792, 0.4309658, 0.3219257, 0.3064732, 0.2756747, 0.2519280,
            0.2425591, 0.2349506, 0.2254680, 0.2206291, 0.2098376, 0.1971485,
            0.1778833, 0.1691119, 0.1528191, 0.1252462
        ])
        assert np.allclose(mca.lam_, e_lam)

        e_expl = np.array([
            .11434479, 0.10774145, 0.08048143, 0.07661830, 0.06891868,
            0.06298199, 0.06063978, 0.05873766, 0.05636700, 0.05515728,
            0.05245940, 0.04928711, 0.04447083, 0.04227798, 0.03820477,
            0.03131155
        ])
        assert np.allclose(mca.expl_, e_expl)

        e_b1 = np.array([
            1.8366266, 0.5462399, -0.4467973, -1.1659032, -1.9952169,
            2.9243212, 0.6415160, 0.3460504, -0.7141260, -1.3537252, 2.1577825,
            0.2468277, -0.6189958, -1.3488576, -1.4675816, 1.2037823,
            -0.2211514, -0.3846555, -0.2216352, 0.7077495
        ])
        assert np.allclose(mca.b_[0], e_b1)

        e_g1 = np.array([
            1.2421072, 0.3694211, -0.3021682, -0.7884981, -1.3493615,
            1.9777130, 0.4338561, 0.2340332, -0.4829620, -0.9155218, 1.4593042,
            0.1669291, -0.4186257, -0.9122299, -0.9925227, 0.8141157,
            -0.1495643, -0.2601418, -0.1498915, 0.4786497
        ])
        assert np.allclose(mca.g_[0], e_g1)

        e_f0 = np.array([-.210, -0.325, 0.229, 0.303, -0.276])
        e_f1 = np.array([0.443, 0.807, 0.513, 0.387, 1.092])
        assert np.allclose(mca.f_[:5, 0], e_f0, atol=1e-2)
        assert np.allclose(mca.f_[:5, 1], e_f1, atol=1e-2)

    def test_n_components(self, data):
        mca = skmca.MCA(n_components=2)
        mca.fit(data)

        assert mca.f_.shape == (871, 2)
        assert mca.g_.shape == (2, 20)
        assert mca.a_.shape == (871, 2)
