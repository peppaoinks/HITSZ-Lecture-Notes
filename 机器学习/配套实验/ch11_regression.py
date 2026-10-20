"""实验11.1：OLS、岭的目标系数以及秩亏预测。"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

def ridge(X,y,lam):
    X=np.asarray(X,float); y=np.asarray(y,float)
    if X.ndim!=2 or X.shape[0]==0 or y.shape!=(len(X),) or not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isfinite(lam) or lam<0:
        raise ValueError('invalid input')
    xm=X.mean(0); ym=y.mean(); Z=X-xm; q=y-ym
    # 增广最小二乘比显式求逆更稳定；截距不罚。
    aug=np.vstack([Z,np.sqrt(len(y)*lam)*np.eye(X.shape[1])])
    target=np.r_[q,np.zeros(X.shape[1])]
    w=np.linalg.lstsq(aug,target,rcond=None)[0]
    return w,float(ym-xm@w)

def main():
    X=np.array([[0.],[1.],[2.]]); y=np.array([1.,2.,2.])
    w,b=ridge(X,y,0); print('OLS',b,w)
    np.testing.assert_allclose([b,w[0]],[7/6,.5])
    for lam in [0.,.2,1.]:
        w,b=ridge(X,y,lam)
        lib=Ridge(alpha=len(y)*lam).fit(X,y) if lam else LinearRegression().fit(X,y)
        np.testing.assert_allclose(w,lib.coef_,atol=1e-12)
        np.testing.assert_allclose(b,lib.intercept_,atol=1e-12)
        print(f'lambda={lam} intercept={b:.6f} slope={w[0]:.6f}')
    singular=np.c_[X,X]
    w,b=ridge(singular,y,0)
    np.testing.assert_allclose(w,[.25,.25]); print('duplicate_columns',w)
    pred=b+singular@w; residual=y-pred
    print('MAE MSE RMSE R2',np.mean(abs(residual)),np.mean(residual**2),np.sqrt(np.mean(residual**2)),1-np.sum(residual**2)/np.sum((y-y.mean())**2))
    print('all checks passed')
if __name__ == '__main__': main()
