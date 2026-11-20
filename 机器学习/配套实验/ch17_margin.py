"""实验17.1：小规模软间隔对偶、KKT及原始-对偶差距。"""
import numpy as np
from scipy.optimize import minimize
from sklearn.svm import SVC


def solve_dual(X,y,C=1.):
    X=np.asarray(X,float); y=np.asarray(y,float)
    if X.ndim!=2 or y.shape!=(len(X),) or not np.isfinite(X).all() or not np.isin(y,[-1,1]).all() or len(np.unique(y))!=2:
        raise ValueError('finite X and both labels -1,+1 required')
    if not np.isfinite(C) or C<=0: raise ValueError('C must be positive')
    gram=X@X.T; Q=y[:,None]*y[None,:]*gram
    result=minimize(lambda a:.5*a@Q@a-a.sum(),np.zeros(len(X)),jac=lambda a:Q@a-1,
                    bounds=[(0,C)]*len(X),constraints={'type':'eq','fun':lambda a:a@y,'jac':lambda a:y},
                    method='SLSQP',options={'ftol':1e-12,'maxiter':2000})
    if not result.success: raise RuntimeError(result.message)
    a=result.x; w=(a*y)@X; free=(a>1e-7)&(a<C-1e-7)
    if not free.any(): raise RuntimeError('teaching solver needs a free support vector for b')
    b=np.mean(y[free]-X[free]@w)
    return a,w,b


def audit(X,y,a,w,b,C):
    margins=y*(X@w+b); xi=np.maximum(0,1-margins)
    primal=.5*w@w+C*xi.sum()
    dual=a.sum()-.5*w@w
    residual=max(abs(a@y),np.max(abs(a*(margins-1+xi))),np.max(abs((C-a)*xi)))
    return margins,xi,primal-dual,residual


def main():
    cases=[(np.array([[-2.,0],[-1,0],[1,0],[2,0]]),np.array([-1,-1,1,1]),10.),
           (np.array([[-2.,0],[-1,0],[1,0],[2,0],[0,0],[0,0]]),np.array([-1,-1,1,1,-1,1]),1.)]
    for X,y,C in cases:
        a,w,b=solve_dual(X,y,C); m,xi,gap,res=audit(X,y,a,w,b,C)
        lib=SVC(kernel='linear',C=C,tol=1e-10).fit(X,y)
        q=np.array([[-3.,1],[-.3,0],[.4,0],[3,1]])
        np.testing.assert_allclose(q@w+b,lib.decision_function(q),atol=1e-6)
        assert abs(gap)<1e-6 and res<1e-6
        print('C / alpha',C,np.round(a,6))
        print('w / b',np.round(w,6),round(float(b),6))
        print('margins / xi',np.round(m,6),np.round(xi,6))
        print('primal-dual gap / KKT residual',gap,res)
        print('decision max error',np.max(abs(q@w+b-lib.decision_function(q))))
    print('all checks passed')

if __name__ == '__main__': main()
