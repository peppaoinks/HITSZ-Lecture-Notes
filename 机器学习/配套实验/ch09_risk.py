"""实验9.1：固定函数的风险、数据选择偏差与成本阈值。"""
import numpy as np

def risk(y, pred, weights=None):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if y.ndim != 1 or y.size == 0 or y.shape != pred.shape:
        raise ValueError('expected matching nonempty vectors')
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('nonfinite input')
    if weights is None:
        return float(np.mean((y-pred)**2))
    weights = np.asarray(weights, float)
    if weights.shape != y.shape or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('invalid weights')
    return float(np.average((y-pred)**2, weights=weights))

def main():
    y = np.array([0., 2.]); p = np.array([.75, .25])
    for c in [0., .5, 1.]:
        print(f'constant={c:.1f} population_MSE={risk(y, np.full(2,c),p):.6f}')
    train = np.array([0., 0., 0., 2.])
    assert risk(train, np.zeros(4)) == 1
    assert risk(y, np.full(2,.5),p) == .75
    prob = np.array([.1,.3,.6,.9]); labels = np.array([0,1,0,1])
    for threshold in [.5, 1/6]:
        decision = prob >= threshold
        fp = np.sum(decision & (labels == 0)); fn = np.sum(~decision & (labels == 1))
        print(f'threshold={threshold:.6f} FP={fp} FN={fn} cost={fp+5*fn}')
    print('all checks passed')
if __name__ == '__main__': main()
