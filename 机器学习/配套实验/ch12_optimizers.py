"""实验12.1：相同二次目标上的六种优化器；每种方法独立状态。"""
import numpy as np

class Optimizer:
    def __init__(self, name, size, lr=.1, beta=.9, beta2=.999, eps=1e-8, decay=0.):
        if name not in ['sgd','momentum','adagrad','rmsprop','adam','adamw']:
            raise ValueError('unknown optimizer')
        if not (np.isfinite(lr) and lr>0 and 0<=beta<1 and 0<=beta2<1 and eps>0 and decay>=0):
            raise ValueError('invalid optimizer settings')
        self.name=name; self.lr=lr; self.beta=beta; self.beta2=beta2
        self.eps=eps; self.decay=decay; self.t=0
        self.m=np.zeros(size); self.v=np.zeros(size)
    def step(self,w,g):
        w=np.asarray(w,float); g=np.asarray(g,float)
        if w.shape!=self.m.shape or g.shape!=w.shape or not np.isfinite(w).all() or not np.isfinite(g).all():
            raise ValueError('invalid state or gradient')
        self.t+=1; name=self.name
        if name=='sgd': direction=g
        elif name=='momentum':
            self.m=self.beta*self.m+g; direction=self.m
        elif name=='adagrad':
            self.v+=g*g; direction=g/(np.sqrt(self.v)+self.eps)
        elif name=='rmsprop':
            self.v=self.beta2*self.v+(1-self.beta2)*g*g
            direction=g/(np.sqrt(self.v)+self.eps)
        else:
            self.m=self.beta*self.m+(1-self.beta)*g
            self.v=self.beta2*self.v+(1-self.beta2)*g*g
            mh=self.m/(1-self.beta**self.t); vh=self.v/(1-self.beta2**self.t)
            direction=mh/(np.sqrt(vh)+self.eps)
        shrink=1-self.lr*self.decay if name=='adamw' else 1.
        return shrink*w-self.lr*direction

def main():
    rates={'sgd':.1,'momentum':.05,'adagrad':.3,'rmsprop':.05,'adam':.1,'adamw':.1}
    for name,lr in rates.items():
        w=np.array([2.,2.]); opt=Optimizer(name,2,lr=lr,beta2=.9 if name=='rmsprop' else .999,decay=.01 if name=='adamw' else 0.)
        for _ in range(200): w=opt.step(w,np.array([1.,9.])*w)
        value=.5*np.sum(np.array([1.,9.])*w*w)
        print(name,'w',np.round(w,6),'loss',f'{value:.8f}')
        assert np.isfinite(value) and value < 20
    w=np.array([1.]); g=np.array([2.]); adam=Optimizer('adam',1); adamw=Optimizer('adamw',1,decay=.2)
    np.testing.assert_allclose(adam.step(w,g),[.9],atol=1e-8)
    np.testing.assert_allclose(adamw.step(w,g),[.88],atol=1e-8)
    print('first_step Adam=.9 AdamW=.88; all checks passed')
if __name__ == '__main__': main()
