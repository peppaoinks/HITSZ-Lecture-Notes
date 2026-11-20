"""第17章手算、次梯度、核映射及参数换算核对。"""
import numpy as np
from sklearn.svm import SVC
from ch17_margin import solve_dual,audit
from ch17_kernel import rbf


def hinge_objective(theta,X,y,lam):
    w,b=theta[:-1],theta[-1]; m=y*(X@w+b); active=m<1
    loss=lam/2*(w@w)+np.maximum(0,1-m).mean()
    grad=np.r_[lam*w-X[active].T@y[active]/len(X),-y[active].sum()/len(X)]
    return loss,grad


def main():
    X=np.array([[-2.,0],[-1,0],[1,0],[2,0]]); y=np.array([-1,-1,1,1])
    a,w,b=solve_dual(X,y,10.)
    np.testing.assert_allclose(a,[0,.5,.5,0],atol=1e-6)
    np.testing.assert_allclose(w,[1,0],atol=1e-6)
    # 远离hinge拐点才使用中心差分。
    theta=np.array([.2,-.1,.15]); lam=.3; h=1e-6
    numeric=np.array([(hinge_objective(theta+np.eye(3)[j]*h,X,y,lam)[0]-hinge_objective(theta-np.eye(3)[j]*h,X,y,lam)[0])/(2*h) for j in range(3)])
    np.testing.assert_allclose(numeric,hinge_objective(theta,X,y,lam)[1],atol=1e-9)
    A=np.array([[1.,2.],[3.,-1.],[-2.,.5]])
    phi=np.c_[A[:,0]**2,np.sqrt(2)*A[:,0]*A[:,1],A[:,1]**2]
    np.testing.assert_allclose(phi@phi.T,(A@A.T)**2,atol=1e-12)
    K=rbf(A,A,.7); assert np.linalg.eigvalsh(K).min()>-1e-12
    # 负距离不是半正定核：取一个非零点时其自内积应非负。
    bad=-A@A.T; assert np.linalg.eigvalsh(bad).min()<0
    # 均值损失lambda固定时，复制样本应将SVC的C除以复制倍数。
    c=1/(len(X)*lam)
    m1=SVC(kernel='linear',C=c,tol=1e-10).fit(X,y)
    m2=SVC(kernel='linear',C=c/2,tol=1e-10).fit(np.repeat(X,2,axis=0),np.repeat(y,2))
    np.testing.assert_allclose(m1.decision_function(A),m2.decision_function(A),atol=1e-6)
    print('alpha',np.round(a,6),'w',np.round(w,6),'b',b)
    print('hinge gradient max error',np.max(abs(numeric-hinge_objective(theta,X,y,lam)[1])))
    print('polynomial map / PSD / copied-sample checks passed')
    print('all checks passed')

if __name__ == '__main__': main()
