"""第16章手算与边界核对；与主实验共同使用。"""
import numpy as np
from scipy.special import logsumexp
from ch16_models import count_fit,count_log_proba,gaussian_fit


def main():
    # 例16.1，类1先验.4，两个症状条件概率.8/.7和.2/.3。
    joint=np.array([.6*.2*.7, .4*.8*.3])
    np.testing.assert_allclose(joint/joint.sum(),[7/15,8/15])
    # 例16.2的伯努利完整事件。
    theta=np.array([[3/4,1/2],[1/4,3/4]])
    p=np.exp(count_log_proba([[1,0]],np.log([.5,.5]),theta,True))
    np.testing.assert_allclose(p,[[6/7,1/7]])
    # 例16.3，多项式计数。
    theta=np.array([[1/2,3/8,1/8],[1/8,1/4,5/8]])
    p=np.exp(count_log_proba([[1,0,1]],np.log([.5,.5]),theta))
    np.testing.assert_allclose(p,[[4/9,5/9]])
    # 复制同一个证据使模型过度自信。
    one=np.array([.8,.2]); duplicate=one**2; duplicate/=duplicate.sum()
    np.testing.assert_allclose(duplicate,[16/17,1/17])
    rejected=0
    bad=[(np.array([[-1.,0],[0,1]]),np.array([0,1]),1.,False),
         (np.array([[1.,0],[0,1]]),np.array([0,1]),0.,False),
         (np.array([[2.,0],[0,1]]),np.array([0,1]),1.,True)]
    for X,y,a,b in bad:
        try: count_fit(X,y,a,b)
        except ValueError: rejected+=1
    assert rejected==3
    try: gaussian_fit(np.ones((4,2)),np.array([0,0,1,1]))
    except ValueError: rejected+=1
    assert rejected==4
    # log-sum-exp平移不变，长文普通连乘下溢。
    s=np.array([-1000.,-1002.]); shifted=s+5000
    np.testing.assert_allclose(s-logsumexp(s),shifted-logsumexp(shifted),atol=1e-12)
    assert .01**200==0
    print('posterior examples: 8/15, 6/7, 5/9; duplicate: 16/17')
    print('invalid cases rejected',rejected)
    print('all checks passed')

if __name__ == '__main__': main()
