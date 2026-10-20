"""第11章一般相关设计、零列与KKT，以及岭的平移性质。"""
import numpy as np
from sklearn.linear_model import Lasso
from ch11_regression import ridge
from ch11_lasso import coordinate_descent,soft

def main():
    X=np.arange(3.)[:,None]; y=np.array([2.,3.,5.]); w,b=ridge(X,y,0)
    np.testing.assert_allclose([b,w[0]],[11/6,1.5])
    for lam in [0,.3,2]:
        w,b=ridge(X,y,lam); wp,bp=ridge(X,y+100,lam)
        np.testing.assert_allclose(w,wp,atol=1e-12); np.testing.assert_allclose(bp-b,100)
    np.testing.assert_allclose(soft(np.array([-2,.3,1.5]),.5),[-1.5,0,1])
    rng=np.random.default_rng(1103); Z=rng.normal(size=(120,4)); Z[:,1]=.8*Z[:,0]+.2*Z[:,1]
    Z=np.c_[Z,np.zeros(len(Z))]; Z-=Z.mean(0)
    q=Z@np.array([2.,0.,-1.,0.,0.])+rng.normal(0,.1,len(Z)); q-=q.mean()
    for lam in [.01,.1,1.]:
        w=coordinate_descent(Z,q,lam); ref=Lasso(alpha=lam,fit_intercept=False,tol=1e-12,max_iter=100000).fit(Z,q)
        np.testing.assert_allclose(Z@w,Z@ref.coef_,atol=1e-7)
        g=Z.T@(Z@w-q)/len(q); active=abs(w)>1e-8
        assert np.max(abs(g[active]+lam*np.sign(w[active])),initial=0)<1e-7
        assert np.max(abs(g[~active]),initial=0)<lam+1e-7
        assert w[-1]==0
    for function in [ridge,coordinate_descent]:
        for xx,yy,ll in [(np.ones((2,1)),[1],.1),(np.array([[np.inf]]),[1],.1),(np.zeros((2,1)),[0,0],-1)]:
            try:function(xx,yy,ll)
            except ValueError:pass
            else:raise AssertionError('invalid input accepted')
    print('hand answers, ridge translation, correlated Lasso/library/KKT, zero column, invalid inputs: all checks passed')
if __name__=='__main__':main()
