"""第13章答案核对：一阶更新、加权概率、复制样本正则对应与稳定端点。"""
import numpy as np
from scipy.special import expit
from ch13_logistic import objective,fit

def main():
    X=np.array([[0.],[2.]]);y=np.array([0,1])
    loss,g=objective(np.array([0.,0.]),X,y,.1)
    np.testing.assert_allclose(g,[-.5,0]);np.testing.assert_allclose(objective(-g,X,y,.1)[0],.515704434,atol=1e-9)
    g=objective(np.array([1.,0.]),X,y,.2)[1]
    print('exercise_13_4_update',np.array([1.,0.])-.1*g)
    for z in [-1000.,1000.]:
        for label in [0,1]:
            v=objective(np.array([z,0.]),np.array([[1.]]),np.array([label]),0)[0]
            assert np.isfinite(v)
    t,_=fit(X,y,.2);u,_=fit(np.repeat(X,2,axis=0),np.repeat(y,2),.2)
    np.testing.assert_allclose(t,u,atol=1e-10)
    for eta in [.1,.25,.8]:
        q=3*eta/(3*eta+1-eta)
        restored=q/(3*(1-q)+q)
        np.testing.assert_allclose(restored,eta)
    try:objective(np.zeros(2),X,y[:,None],.1)
    except ValueError:pass
    else:raise AssertionError('column labels accepted')
    print('duplicate_data_mean_objective_same_parameters',t)
    print('all checks passed')
if __name__=='__main__':main()
