import unittest
from unittest.mock import patch
import numpy as np
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
import run_experiment as run
from models import labels,fit_part,part,mix,BASE

class ModelTests(unittest.TestCase):
    def test_log_mix(self):
        p={'gate':np.array([[1.,0.,0.],[0.,.5,.5]]),'mid':np.array([1.,np.log(4.)]),'tail':np.array([3.,np.log(16.)])}
        np.testing.assert_allclose(mix(p),[0.,7.],atol=1e-14)

    def test_all_families_handle_missing_values_and_regime_targets(self):
        rng=np.random.default_rng(4);X=rng.normal(size=(400,34));X[:100,30]=np.nan
        y=np.where(X[:,1]<-.4,0.,np.where(X[:,1]>.4,20.,3.))
        y[y>0]+=rng.uniform(0,.001,(y>0).sum())
        with threadpool_limits(limits=2):
            for slot in BASE:
                for family,params in [('cat',dict(iterations=3,depth=3)),('lgb',dict(n_estimators=3,num_leaves=7)),('extra',dict(n_estimators=3,min_samples_leaf=3))]:
                    if family=='extra' and slot!='tail':continue
                    model=fit_part(slot,dict(family=family,params=params),X,y)
                    p=part(model,slot,X[:30]);self.assertTrue(np.isfinite(p).all())
                    if slot=='gate':self.assertEqual(p.shape,(30,3))
                    else:
                        expected=np.log(4.) if slot=='mid' else np.log(21.)
                        np.testing.assert_allclose(p,expected,atol=1e-3)

    def test_inner_selection_does_not_train_on_inner_validation(self):
        rng=np.random.default_rng(5);X=rng.normal(size=(400,34));X[:,0]=np.arange(400)
        y=np.tile([0.,3.,20.,3.],100)
        y[y>0]+=rng.uniform(0,.001,(y>0).sum())
        a,b=train_test_split(np.arange(len(y)),test_size=.2,random_state=1001,stratify=labels(y))
        calls=[]
        def tracked(slot,spec,xx,yy):
            calls.append(set(xx[:,0].astype(int)))
            return fit_part(slot,spec,xx,yy)
        small={'cat':[dict(iterations=2,depth=2)],'lgb':[dict(n_estimators=2,num_leaves=3)],'extra':[dict(n_estimators=2,min_samples_leaf=3)]}
        with threadpool_limits(limits=2),patch.object(run,'GRID',small),patch.object(run,'fit_part',tracked):
            records=[];best,sel=run.tune(X,y,1001,'test',records)
        self.assertEqual(len(best),7)
        self.assertTrue(all(c==set(a) and not(c&set(b)) for c in calls))
        self.assertLessEqual(sel['combo_rmse'],sel['baseline_rmse']+1e-12)

if __name__=='__main__':unittest.main()
