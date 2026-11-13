import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from features import RAW,build_feature_sets
from a_log_mixture import LogTailMixtureSFR


class FeatureTests(unittest.TestCase):
    def catalogue(self):
        rng=np.random.default_rng(42)
        df=pd.DataFrame(rng.normal(size=(80,13)),columns=RAW)
        df['log10_halo_mass']=rng.uniform(12,15,len(df))
        df[['pos_x','pos_y','pos_z']]=rng.uniform(0,200,(len(df),3))
        # A close pair straddles the periodic boundary.
        df.loc[0,['pos_x','pos_y','pos_z']]=[.1,10,10]
        df.loc[1,['pos_x','pos_y','pos_z']]=[199.9,10,10]
        return df

    def test_features_target_independent_and_row_order_invariant(self):
        df=self.catalogue();a=build_feature_sets(df)
        df['sfr_linear']=np.arange(len(df))*100
        b=build_feature_sets(df.sample(frac=1,random_state=7))
        for name,n in [('A13_baseline',13),('A17_mass',17),('A30_environment',30),('A34_both',34)]:
            self.assertEqual(a[name].shape,(80,n))
            np.testing.assert_allclose(a[name],b[name].loc[df.index],rtol=1e-10,atol=1e-10,equal_nan=True)
        np.testing.assert_allclose(a['A17_mass'].dlogMs_growth,df.log10_stellar_mass-df.log10_main_prog_stellar_mass)

    def test_periodic_translation_and_boundary_neighbors(self):
        df=self.catalogue();a=build_feature_sets(df)['A30_environment']
        df[['pos_x','pos_y','pos_z']]=(df[['pos_x','pos_y','pos_z']]+[7,13,29])%200
        b=build_feature_sets(df)['A30_environment']
        np.testing.assert_allclose(a.iloc[:,13:],b.iloc[:,13:],rtol=1e-9,atol=1e-9,equal_nan=True)
        self.assertGreaterEqual(a.log1p_N_r1.iloc[0],np.log(2))

    def test_nan_allowed_but_infinity_rejected_by_model(self):
        rng=np.random.default_rng(4);X=rng.normal(size=(300,5));X[:70,2]=np.nan
        y=np.tile([0.,3.,20.],100)
        with threadpool_limits(limits=2):
            m=LogTailMixtureSFR(tail_threshold=10,hgb_params={'max_iter':5}).fit(X,y)
            p=m.predict(X[:20])
        self.assertTrue(np.isfinite(p).all() and (p>=0).all())
        X[0,2]=np.inf
        with self.assertRaises(ValueError):m.predict(X)

if __name__=='__main__':unittest.main()
