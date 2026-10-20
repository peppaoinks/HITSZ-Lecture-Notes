"""实验13.1：平均二元交叉熵+lambda/2||w||²，截距不罚。"""
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

def objective(theta, X, y, lam):
    X = np.asarray(X, float); y = np.asarray(y, float)
    theta = np.asarray(theta, float)
    if X.ndim != 2 or len(X) == 0 or y.shape != (len(X),) or theta.shape != (X.shape[1]+1,):
        raise ValueError('invalid shape')
    if not all(np.isfinite(a).all() for a in [X,y,theta]) or not np.isin(y,[0,1]).all() or not np.isfinite(lam) or lam < 0:
        raise ValueError('invalid values')
    w, b = theta[:-1], theta[-1]
    z = X@w+b
    loss = np.mean(np.logaddexp(0,z)-y*z)+lam/2*(w@w)
    r = expit(z)-y
    grad = np.r_[X.T@r/len(y)+lam*w, r.mean()]
    return float(loss), grad

def fit(X,y,lam=.1,steps=12000,rate=.1):
    theta = np.zeros(X.shape[1]+1)
    history = []
    for step in range(steps):
        loss,g = objective(theta,X,y,lam)
        if step % 1000 == 0: history.append(loss)
        theta -= rate*g
    return theta, np.array(history)

def main():
    X = np.array([[-2.,0.],[-1.,1.],[0.,-1.],[0.,1.],[1.,-1.],[2.,0.]])
    y = np.array([0,0,0,1,1,1]); lam=.1
    t = np.array([.2,-.3,.1]); h=1e-5
    analytic=objective(t,X,y,lam)[1]
    numeric=np.array([(objective(t+np.eye(3)[j]*h,X,y,lam)[0]-objective(t-np.eye(3)[j]*h,X,y,lam)[0])/(2*h) for j in range(3)])
    np.testing.assert_allclose(analytic,numeric,rtol=1e-6,atol=1e-8)
    theta,history=fit(X,y,lam)
    lib=LogisticRegression(C=1/(len(y)*lam),solver='lbfgs',tol=1e-12,max_iter=5000).fit(X,y)
    p=expit(X@theta[:-1]+theta[-1]); q=lib.predict_proba(X)[:,1]
    np.testing.assert_allclose(p,q,atol=2e-6,rtol=0)
    assert np.all(np.diff(history)<=1e-12)
    extreme=objective(np.array([1000.,0.,0.]),X,y,lam)[0]
    assert np.isfinite(extreme)
    print('theta',np.round(theta,6)); print('probability',np.round(p,6))
    print('objective',objective(theta,X,y,lam)[0])
    print('gradient_check_max_error',np.max(abs(analytic-numeric)))
    print('library_probability_max_error',np.max(abs(p-q)))
    print('all checks passed')
if __name__ == '__main__': main()
