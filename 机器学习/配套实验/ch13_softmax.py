"""实验13.2：稳定多分类交叉熵、梯度、共同平移不变性。"""
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

def objective(t,X,y,K,lam):
    n,d=X.shape; W=t[:d*K].reshape(d,K); b=t[d*K:]
    Z=X@W+b
    logp=Z-logsumexp(Z,axis=1,keepdims=True)
    loss=-logp[np.arange(n),y].mean()+lam/2*np.sum(W*W)
    R=np.exp(logp); R[np.arange(n),y]-=1
    g=np.r_[(X.T@R/n+lam*W).ravel(),R.mean(0)]
    return loss,g

def main():
    rng=np.random.default_rng(1302)
    centers=np.array([[-2.,0.],[2.,0.],[0.,2.]])
    X=np.concatenate([c+rng.normal(0,.7,(40,2)) for c in centers]); y=np.repeat(np.arange(3),40)
    t=rng.normal(0,.1,9); h=1e-5; lam=.05
    g=objective(t,X,y,3,lam)[1]
    ng=np.array([(objective(t+np.eye(9)[j]*h,X,y,3,lam)[0]-objective(t-np.eye(9)[j]*h,X,y,3,lam)[0])/(2*h) for j in range(9)])
    np.testing.assert_allclose(g,ng,rtol=1e-6,atol=1e-8)
    res=minimize(objective,np.zeros(9),args=(X,y,3,lam),jac=True,method='L-BFGS-B',options={'gtol':1e-11,'ftol':1e-14,'maxiter':2000})
    assert res.success,res.message
    W=res.x[:6].reshape(2,3); b=res.x[6:]; Z=X@W+b
    p=np.exp(Z-logsumexp(Z,axis=1,keepdims=True))
    lib=LogisticRegression(C=1/(len(y)*lam),tol=1e-11,max_iter=2000).fit(X,y)
    np.testing.assert_allclose(p,lib.predict_proba(X),atol=2e-5,rtol=0)
    z=np.array([[1000.,1001.,999.]])
    stable=z-logsumexp(z,axis=1,keepdims=True)
    shifted=z+3000
    np.testing.assert_allclose(stable,shifted-logsumexp(shifted,axis=1,keepdims=True),atol=1e-12)
    print('extreme_probability',np.round(np.exp(stable),6))
    print('class_1_loss',-stable[0,1]); print('train_accuracy',np.mean(p.argmax(1)==y))
    print('gradient_check_max_error',np.max(abs(g-ng)))
    print('library_probability_max_error',np.max(abs(p-lib.predict_proba(X))))
    print('all checks passed')
if __name__ == '__main__': main()
