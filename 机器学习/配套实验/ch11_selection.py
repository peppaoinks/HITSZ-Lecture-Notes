"""实验11.3：多项式与正则强度只在验证集选择，最终测试一次。"""
from pathlib import Path
import json
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

def main():
    rng=np.random.default_rng(1104)
    def generate(n):
        X=rng.uniform(-2,2,size=(n,1)); y=1+.5*X[:,0]+.8*X[:,0]**2+rng.normal(0,.35,n)
        return X,y
    X,y=generate(60); V,v=generate(60); T,t=generate(400)
    records=[]; best=None
    for degree in [1,2,5,10]:
        for lam in [0.,.001,.1,1.]:
            pipe=Pipeline([('poly',PolynomialFeatures(degree,include_bias=False)),('scale',StandardScaler()),('ridge',Ridge(alpha=len(y)*lam,solver='svd'))]).fit(X,y)
            train=float(np.mean((pipe.predict(X)-y)**2)); valid=float(np.mean((pipe.predict(V)-v)**2))
            records.append([degree,lam,train,valid])
            if best is None or valid<best[0]:best=(valid,degree,lam,pipe)
    valid,degree,lam,pipe=best
    prediction=pipe.predict(T); err=t-prediction
    score={'degree':degree,'lambda':lam,'valid_mse':valid,'test_mae':float(np.mean(abs(err))),'test_rmse':float(np.sqrt(np.mean(err**2))),'test_r2':float(1-np.sum(err**2)/np.sum((t-t.mean())**2))}
    root=Path(__file__).resolve().parent/'ch11_results';root.mkdir(exist_ok=True)
    np.savetxt(root/'selection.csv',records,delimiter=',',header='degree,lambda,train_mse,valid_mse',comments='')
    np.savetxt(root/'residuals.csv',np.c_[T[:,0],t,prediction,err],delimiter=',',header='x,y,prediction,residual',comments='')
    (root/'selection.json').write_text(json.dumps(score,indent=2)+'\n')
    assert score['test_rmse']<.6
    print(np.array(records)); print(json.dumps(score,indent=2));print('all checks passed')
if __name__=='__main__':main()
