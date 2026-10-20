"""第15章编程核对：NCA解析梯度、平移/重命名不变性、近邻边界。"""
import numpy as np
from scipy.special import logsumexp
from ch15_neighbors import neighbors, weights

def nca(L,X,y):
    delta=X[:,None,:]-X[None,:,:]
    C=delta[:,:,:,None]*delta[:,:,None,:]
    dist=np.sum((delta@L.T)**2,axis=2)
    score=-dist;np.fill_diagonal(score,-np.inf)
    p=np.exp(score-logsumexp(score,axis=1,keepdims=True))
    same=(y[:,None]==y[None,:]);np.fill_diagonal(same,False)
    correct=np.sum(p*same,axis=1)
    first=np.einsum('i,ij,ijab->ab',correct,p,C)
    second=np.einsum('ij,ij,ijab->ab',p,same,C)
    return correct.sum(),2*L@(first-second),p

def main():
    X=np.array([[0.,0.],[1.,.2],[0.,2.],[1.,2.5]])
    y=np.array([0,0,1,1]);L=np.array([[.7,.2],[-.1,.8]])
    value,g,p=nca(L,X,y);numeric=np.zeros_like(L);h=1e-5
    for ix in np.ndindex(L.shape):
        e=np.zeros_like(L);e[ix]=h
        numeric[ix]=(nca(L+e,X,y)[0]-nca(L-e,X,y)[0])/(2*h)
    np.testing.assert_allclose(g,numeric,atol=1e-8,rtol=1e-6)
    np.testing.assert_allclose(p.sum(1),1);np.testing.assert_array_equal(np.diag(p),0)
    np.testing.assert_allclose(nca(L,X+np.array([8.,-3.]),y)[0],value)
    np.testing.assert_allclose(nca(L,X,9-y)[0],value)
    assert neighbors(X,np.empty((0,2)),2)[0].shape==(0,2)
    for Q in [np.ones((2,3)),np.array([[np.nan,0]])]:
        try:neighbors(X,Q,1)
        except ValueError:pass
        else:raise AssertionError('invalid query accepted')
    try:neighbors(np.array([[1e308]]),np.array([[-1e308]]),1)
    except FloatingPointError:pass
    else:raise AssertionError('overflow accepted')
    np.testing.assert_allclose(weights(np.array([[0.,0.,1.],[1.,2.,4.]])),[[.5,.5,0],[4/7,2/7,1/7]])
    print('NCA_objective',value);print('NCA_gradient',g);print('max_gradient_error',np.max(abs(g-numeric)))
    print('all checks passed')
if __name__=='__main__': main()
