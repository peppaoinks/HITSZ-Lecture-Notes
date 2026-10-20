"""实验11.2：坐标下降、软阈值与KKT核对。输入必须已中心化。"""
import numpy as np
from sklearn.linear_model import Lasso

def soft(z,t):
    return np.sign(z)*np.maximum(np.abs(z)-t,0.)

def coordinate_descent(X,y,lam,tol=1e-10,max_iter=20000):
    X=np.asarray(X,float); y=np.asarray(y,float)
    if X.ndim!=2 or len(X)==0 or y.shape!=(len(X),) or not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isfinite(lam) or lam<0:
        raise ValueError('invalid input')
    if not np.allclose(X.mean(0),0,atol=1e-10) or abs(y.mean())>1e-10:
        raise ValueError('center X and y first')
    w=np.zeros(X.shape[1]); norm=np.mean(X**2,axis=0)
    for _ in range(max_iter):
        old=w.copy()
        for j in range(X.shape[1]):
            r=y-X@w+X[:,j]*w[j]
            w[j]=soft(np.mean(X[:,j]*r),lam)/norm[j] if norm[j]>0 else 0.
        if np.max(abs(w-old),initial=0.) < tol:
            return w
    raise RuntimeError('coordinate descent did not converge')

def main():
    X=np.array([[1.,1.],[1.,-1.],[-1.,1.],[-1.,-1.]])
    y=X@np.array([3.,.4]); lam=.5
    w=coordinate_descent(X,y,lam); lib=Lasso(alpha=lam,fit_intercept=False,tol=1e-12).fit(X,y)
    np.testing.assert_allclose(w,[2.5,0]); np.testing.assert_allclose(w,lib.coef_)
    g=X.T@(X@w-y)/len(y)
    active=np.abs(w)>1e-9
    assert np.max(np.abs(g[active]+lam*np.sign(w[active])),initial=0)<1e-8
    assert np.max(np.abs(g[~active]),initial=0)<=lam+1e-8
    print('lasso',w,'gradient',g)
    print('ridge_same_lambda',np.linalg.solve(X.T@X+len(y)*lam*np.eye(2),X.T@y))
    print('all checks passed')
if __name__ == '__main__': main()
