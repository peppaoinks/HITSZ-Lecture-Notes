"""实验12.2：小批量训练、加权日志和最佳验证状态恢复。"""
from pathlib import Path
import json
import numpy as np
from ch12_optimizers import Optimizer

def train(X,y,Xv,yv,*,batch=16,epochs=120,lr=.05,patience=12,seed=1202):
    if X.ndim!=2 or Xv.ndim!=2 or len(X)==0 or len(Xv)==0 or X.shape[1]!=Xv.shape[1] or y.shape!=(len(X),) or yv.shape!=(len(Xv),):
        raise ValueError('invalid data shape')
    if not all(np.isfinite(a).all() for a in [X,y,Xv,yv]) or batch<1 or epochs<1 or patience<1:
        raise ValueError('invalid training input')
    rng=np.random.default_rng(seed); w=np.zeros(X.shape[1]); opt=Optimizer('adam',len(w),lr)
    history=[]; best=float('inf'); saved=w.copy(); best_epoch=0; stale=0
    for epoch in range(1,epochs+1):
        order=rng.permutation(len(y))
        online_sum=0.
        for start in range(0,len(y),batch):
            ix=order[start:start+batch]; r=X[ix]@w-y[ix]
            online_sum+=np.sum(r*r)/2
            g=X[ix].T@r/len(ix)
            w=opt.step(w,g)
        tr=np.mean((X@w-y)**2)/2; va=np.mean((Xv@w-yv)**2)/2
        history.append([epoch,online_sum/len(y),tr,va])
        if va < best-1e-8:
            best=va; saved=w.copy(); best_epoch=epoch; stale=0
        else:
            stale+=1
            if stale>=patience: break
    return saved, np.array(history),best_epoch

def main():
    rng=np.random.default_rng(1202)
    x=rng.normal(size=(103,2)); xv=rng.normal(size=(47,2))
    X=np.c_[np.ones(len(x)),x]; Xv=np.c_[np.ones(len(xv)),xv]
    truth=np.array([1.,2.,-1.]); y=X@truth+rng.normal(0,.3,len(x)); yv=Xv@truth+rng.normal(0,.3,len(xv))
    w,h,epoch=train(X,y,Xv,yv)
    value=np.mean((Xv@w-yv)**2)/2
    np.testing.assert_allclose(value,h[epoch-1,3])
    assert value<.2 and np.isfinite(h).all()
    result=Path(__file__).resolve().parent/'ch12_results'; result.mkdir(exist_ok=True)
    np.savetxt(result/'training.csv',h,delimiter=',',header='epoch,online_half_mse,train_half_mse,valid_half_mse',comments='')
    report={'best_epoch':epoch,'epochs_run':len(h),'steps_per_epoch':7,'best_valid_half_mse':value,'weights':w.tolist()}
    (result/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2)); print('all checks passed')
if __name__ == '__main__': main()
