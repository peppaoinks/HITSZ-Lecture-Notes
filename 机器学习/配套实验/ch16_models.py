"""实验16.1：三种朴素贝叶斯、对数计算及库对照。固定教学数组。"""
import numpy as np
from scipy.special import logsumexp
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB


def validate(X, y=None, binary=False):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError('X must be a finite nonempty matrix')
    if binary and not np.isin(X, [0, 1]).all():
        raise ValueError('binary input required')
    if y is None:
        return X
    y = np.asarray(y)
    if y.shape != (len(X),) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('both classes 0 and 1 required')
    return X, y


def count_fit(X, y, alpha=1., binary=False):
    X, y = validate(X, y, binary)
    if (X < 0).any() or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('nonnegative X and positive alpha required')
    n = np.bincount(y.astype(int), minlength=2)
    counts = np.stack([X[y == c].sum(axis=0) for c in [0, 1]])
    denominator = n[:, None] + 2*alpha if binary else counts.sum(axis=1, keepdims=True) + X.shape[1]*alpha
    theta = (counts + alpha)/denominator
    return np.log(n/len(y)), theta


def count_log_proba(X, log_prior, theta, binary=False):
    X = validate(X, binary=binary)
    if X.shape[1] != theta.shape[1] or (X < 0).any():
        raise ValueError('feature mismatch or negative count')
    scores = X @ np.log(theta).T + log_prior
    if binary:
        scores += (1-X) @ np.log1p(-theta).T
    return scores - logsumexp(scores, axis=1, keepdims=True)


def gaussian_fit(X, y, var_smoothing=1e-9):
    X, y = validate(X, y)
    if not np.isfinite(var_smoothing) or var_smoothing <= 0:
        raise ValueError('positive variance smoothing required')
    eps = var_smoothing * np.var(X, axis=0).max()
    if eps == 0:
        raise ValueError('all features globally constant; remove them first')
    mu = np.stack([X[y == c].mean(axis=0) for c in [0, 1]])
    var = np.stack([X[y == c].var(axis=0) for c in [0, 1]]) + eps
    prior = np.array([(y == c).mean() for c in [0, 1]])
    return np.log(prior), mu, var


def gaussian_log_proba(X, log_prior, mu, var):
    X = validate(X)
    if X.shape[1] != mu.shape[1]:
        raise ValueError('feature mismatch')
    diff = X[:, None, :] - mu[None, :, :]
    scores = log_prior - .5*np.sum(np.log(2*np.pi*var)[None, :, :] + diff**2/var, axis=2)
    return scores-logsumexp(scores, axis=1, keepdims=True)


def main():
    X = np.array([[2,1,0], [1,1,0], [0,1,2], [0,0,2]])
    y = np.array([0,0,1,1]); Q = np.array([[1,0,1], [0,0,0], [2000,0,2000]])
    prior,theta = count_fit(X,y)
    own = count_log_proba(Q,prior,theta)
    lib = MultinomialNB(alpha=1).fit(X,y).predict_log_proba(Q)
    np.testing.assert_allclose(own,lib,atol=1e-10)
    np.testing.assert_allclose(np.exp(own).sum(axis=1),1,atol=1e-12)
    B = (X>0).astype(float); p,t = count_fit(B,y,binary=True)
    qb = (Q>0).astype(float)
    bp = count_log_proba(qb,p,t,binary=True)
    blib = BernoulliNB(alpha=1,binarize=None).fit(B,y).predict_log_proba(qb)
    np.testing.assert_allclose(bp,blib,atol=1e-12)
    G = np.array([[-2,0],[-1,1],[0,-1],[2,1],[3,0],[4,2]],float)
    gy = np.array([0,0,0,1,1,1]); qp = np.array([[0,0],[2,1],[1,0]],float)
    gp,mu,var = gaussian_fit(G,gy)
    pred = gaussian_log_proba(qp,gp,mu,var)
    glib = GaussianNB(var_smoothing=1e-9).fit(G,gy).predict_log_proba(qp)
    np.testing.assert_allclose(pred,glib,atol=1e-12)
    print('multinomial theta',theta)
    print('multinomial posterior',np.round(np.exp(own),6))
    print('bernoulli theta',t)
    print('bernoulli posterior',np.round(np.exp(bp),6))
    print('gaussian posterior',np.round(np.exp(pred),6))
    print('maximum log-probability errors',np.max(abs(own-lib)),np.max(abs(bp-blib)),np.max(abs(pred-glib)))
    print('all checks passed')

if __name__ == '__main__':
    main()
