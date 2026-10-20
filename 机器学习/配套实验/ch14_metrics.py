"""实验14.1：固定预测的阈值指标、AP、ROC AUC及配对bootstrap。"""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, log_loss, brier_score_loss

def main():
    y=np.array([1,0,1,0,0,1]); p=np.array([.9,.8,.7,.4,.3,.2])
    for threshold in [.5,.75]:
        pred=p>=threshold
        print('threshold',threshold,'matrix',confusion_matrix(y,pred,labels=[0,1]).tolist())
        print('precision recall f1',precision_score(y,pred,zero_division=0),recall_score(y,pred),f1_score(y,pred))
    auc=roc_auc_score(y,p); ap=average_precision_score(y,p)
    np.testing.assert_allclose([auc,ap],[5/9,13/18])
    print('ROC_AUC AP',auc,ap); print('logloss brier',log_loss(y,p),brier_score_loss(y,p))
    # 预测固定且保持配对；此小样本仅演示算法，不解释为可靠区间。
    q=np.array([.8,.7,.8,.3,.2,.4]); delta=(y-p)**2-(y-q)**2
    rng=np.random.default_rng(1401); ix=rng.integers(0,len(y),(10000,len(y)))
    diffs=delta[ix].mean(1); interval=np.quantile(diffs,[.025,.975])
    reverse=(-delta)[ix].mean(1)
    np.testing.assert_allclose(reverse,-diffs)
    print('Brier_A_minus_B',delta.mean(),'paired_interval',interval)
    assert np.isfinite(diffs).all()
    print('all checks passed')
if __name__ == '__main__': main()
