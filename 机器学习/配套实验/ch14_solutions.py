"""第14章答案核对：排序不变性、成本比例与汇总指标的区别。"""
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss

def main():
    y=np.array([1,0,1,0,0,1]);p=np.array([.9,.8,.7,.4,.3,.2]);q=p**2
    np.testing.assert_allclose(roc_auc_score(y,p),roc_auc_score(y,q))
    np.testing.assert_allclose(average_precision_score(y,p),average_precision_score(y,q))
    assert not np.array_equal(p>=.5,q>=.5)
    print('probability_transform_Brier',brier_score_loss(y,p),brier_score_loss(y,q))
    print('probability_transform_logloss',log_loss(y,p),log_loss(y,q))
    np.testing.assert_allclose(roc_auc_score([1,1,0,0],[.8,.5,.5,.2]),.875)
    np.testing.assert_allclose(average_precision_score([0,1,1,0,1],[5,4,3,2,1]),53/90)
    mean_f1=(18/19+2/11)/2; pooled_f1=20/30
    print('fold_mean_f1 pooled_f1',mean_f1,pooled_f1)
    assert abs(mean_f1-pooled_f1)>.1
    print('all checks passed')
if __name__=='__main__':main()
