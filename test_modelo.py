import unittest
import numpy as np
import pandas as pd
from modelo import information_clock, synthetic_data, scale_exponent, robust_past_z, confirmed_pivots, best_fibonacci_impulses, hurst_spectra

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=synthetic_data(1300, seed=2)

    def test_time_monotonic(self):
        result=information_clock(self.df)
        self.assertTrue((result.clock_speed>0).all())
        self.assertTrue((np.diff(result.tau_minutes)>0).all())

    def test_z_score_is_past_only(self):
        original=np.arange(400,dtype=float)*0.005 + np.sin(np.arange(400)/8)
        alternate=original.copy(); alternate[-1]=100000
        za=robust_past_z(pd.Series(original))
        zb=robust_past_z(pd.Series(alternate))
        np.testing.assert_allclose(za[:-1],zb[:-1])

    def test_clock_ignores_future_data(self):
        df=self.df.copy()
        altered=df.copy()
        altered.loc[len(df)-1,'num_trades']=10000000
        v0=information_clock(df).clock_speed.to_numpy()
        v1=information_clock(altered).clock_speed.to_numpy()
        np.testing.assert_allclose(v0[:-1],v1[:-1])

    def test_hurst_random_walk(self):
        rng=np.random.default_rng(11)
        x=np.cumsum(rng.normal(size=15000))
        h,r=scale_exponent(x,np.arange(len(x)),np.array([1,2,3,5,8,13,21,34,55,89]))
        self.assertAlmostEqual(h,.5,delta=.12)
        self.assertGreater(r,.9)

    def test_spectra_three_bands(self):
        df=information_clock(self.df)
        spec=hurst_spectra(df)
        self.assertEqual(spec.scale.nunique(),3)
        self.assertTrue(np.isfinite(spec.H_calendar).all())
        self.assertTrue(np.isfinite(spec.H_information).all())

    def test_impulse_confirmation_never_precedes_pivot(self):
        seq=np.log(information_clock(self.df).close.to_numpy())
        pivots=confirmed_pivots(seq)
        self.assertTrue(all(p.index<=p.confirmed_at for p in pivots))
        candidates=best_fibonacci_impulses(seq,pivots)
        self.assertTrue(all(p['known_at']>=p['pivot_indices'][-1] for p in candidates))

if __name__=='__main__':
    unittest.main(verbosity=2)
