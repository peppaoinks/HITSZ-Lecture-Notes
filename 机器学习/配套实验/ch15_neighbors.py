"""实验15.1：稳定同分、零距离和分块精确近邻。"""
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

def neighbors(X,Q,k,block=32):
    X=np.asarray(X,float); Q=np.asarray(Q,float)
    if X.ndim!=2 or Q.ndim!=2 or len(X)==0 or X.shape[1]==0 or Q.shape[1]!=X.shape[1]: raise ValueError('shape')
    if not np.isfinite(X).all() or not np.isfinite(Q).all(): raise ValueError('nonfinite')
    if isinstance(k,(bool,np.bool_)) or not isinstance(k,(int,np.integer)) or not 1<=k<=len(X): raise ValueError('k')
    if isinstance(block,(bool,np.bool_)) or not isinstance(block,(int,np.integer)) or block<1: raise ValueError('block')
    ids=[]; distances=[]
    for start in range(0,len(Q),block):
        with np.errstate(over='raise',invalid='raise'):
            D=np.sum((Q[start:start+block,None,:]-X[None,:,:])**2,axis=2)
        order=np.argsort(D,axis=1,kind='stable')[:,:k]
        ids.append(order); distances.append(np.sqrt(np.take_along_axis(D,order,axis=1)))
    if not ids: return np.empty((0,k),int),np.empty((0,k))
    return np.vstack(ids),np.vstack(distances)

def weights(dist):
    zero=dist==0; has=zero.any(1)
    a=np.zeros_like(dist)
    a[has]=zero[has]
    # 距离先按每行最小值缩放，与1/d成比例，避免极小距离取倒数溢出。
    a[~has]=dist[~has].min(1,keepdims=True)/dist[~has]
    return a/a.sum(1,keepdims=True)

def main():
    X=np.array([[0.],[0.],[2.],[4.]])
    y=np.array([1.,3.,8.,10.]); Q=np.array([[0.],[1.],[3.]])
    ix,d=neighbors(X,Q,3,block=1); pred=np.sum(weights(d)*y[ix],axis=1)
    lib=KNeighborsRegressor(n_neighbors=3,weights='distance',algorithm='brute').fit(X,y)
    np.testing.assert_allclose(pred[:2],lib.predict(Q[:2]))
    # 第三条查询的第3/4名同距且标签不同，库未承诺稳定索引规则。
    print('boundary_tie_library',lib.predict(Q[2:]))
    ix4,d4=neighbors(X,Q,4)
    lib4=KNeighborsRegressor(n_neighbors=4,weights='distance',algorithm='brute').fit(X,y)
    np.testing.assert_allclose(np.sum(weights(d4)*y[ix4],axis=1),lib4.predict(Q))
    np.testing.assert_allclose(pred,[2.,4.,55/7])
    for block in [1,2,10]: np.testing.assert_array_equal(neighbors(X,Q,3,block)[0],ix)
    labels=np.array([0,1,1,0]); vote=np.array([np.bincount(labels[i],minlength=2).argmax() for i in ix])
    print('indices',ix); print('weighted_regression',pred); print('uniform_classification',vote)
    for bad in [0,5,1.5,True]:
        try: neighbors(X,Q,bad)
        except ValueError: pass
        else: raise AssertionError('invalid k accepted')
    print('all checks passed')
if __name__ == '__main__': main()
