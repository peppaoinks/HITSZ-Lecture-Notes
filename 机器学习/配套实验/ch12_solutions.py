"""第12章更新手算、差分、末批归一化与恢复核对。"""
import numpy as np
from ch12_optimizers import Optimizer
from ch12_training import train

def main():
    op=Optimizer('momentum',1,lr=.1,beta=.9); w=np.array([1.])
    w=op.step(w,np.array([2.])); np.testing.assert_allclose(w,[.8])
    w=op.step(w,np.array([-1.])); np.testing.assert_allclose(w,[.72])
    op=Optimizer('adagrad',1,lr=.1); w=np.zeros(1)
    w=op.step(w,np.array([2.])); np.testing.assert_allclose(w,[-.1],atol=1e-8)
    w=op.step(w,np.array([2.])); np.testing.assert_allclose(w,[-.1-.1/np.sqrt(2)],atol=1e-8)
    # 独立的一步手算，防止训练例程与答案复用同一错误梯度。
    X=np.array([[1.],[2.],[3.]]); y=np.array([1.,0.,2.]); w=np.array([.4])
    g=X.T@(X@w-y)/3; h=1e-6
    f=lambda a:np.mean((X@a-y)**2)/2
    numerical=(f(w+h)-f(w-h))/(2*h)
    np.testing.assert_allclose(g,[numerical],atol=1e-9)
    # 整批第一次Adam更新：全梯度为-7/3，更新向正方向移动0.1。
    saved,history,epoch=train(X,y,X,y,batch=10,epochs=1,lr=.1,patience=1)
    np.testing.assert_allclose(saved,[.1],atol=1e-8)
    np.testing.assert_allclose(history[0,1],5/6)
    assert [len(range(103)[i:i+16]) for i in range(0,103,16)]==[16]*6+[7]
    try: Optimizer('adam',1).step([0.],[float('nan')])
    except ValueError:pass
    else:raise AssertionError('nonfinite gradient accepted')
    print('momentum, AdaGrad, central difference, one-step Adam, online weighting, short batch: all checks passed')
if __name__=='__main__':main()
