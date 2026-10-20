"""第9章数值答案、训练偏差恒等式与错误输入核对。"""
import numpy as np
from ch09_risk import risk

def main():
    assert risk([1,6],[2,2],[.8,.2]) == 4
    np.testing.assert_allclose([risk([1,3,4],[2,2,2]),np.mean(abs(np.array([1,3,4])-2))],[2,4/3])
    rng=np.random.default_rng(9); y=rng.normal(size=(200,7)); means=y.mean(1)
    np.testing.assert_allclose(np.sum((y-means[:,None])**2,1),np.sum(y*y,1)-7*means**2,atol=1e-12)
    for args in [([],[]),([1],[1,2]),([np.nan],[1]),([1],[1],[-1]),([1],[1],[0])]:
        try: risk(*args)
        except ValueError: pass
        else: raise AssertionError(args)
    print('9.3 risk=4; 9.4 MSE=2 MAE=4/3; 9.6 thresholds=.3/.4')
    print('sample-centering identity and five invalid inputs: all checks passed')
if __name__=='__main__':main()
