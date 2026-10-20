"""实验14.2：嵌套交叉验证评估选择流程，锁定后一次测试。"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss, accuracy_score

def main():
    X,y=make_classification(n_samples=600,n_features=12,n_informative=5,n_redundant=2,weights=[.7,.3],flip_y=.06,random_state=1402)
    Xd,Xt,yd,yt=train_test_split(X,y,test_size=.2,random_state=14,stratify=y)
    inner=StratifiedKFold(4,shuffle=True,random_state=141)
    outer=StratifiedKFold(5,shuffle=True,random_state=142)
    pipe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))
    search=GridSearchCV(pipe,{'logisticregression__C':[.01,.1,1.,10.]},scoring='neg_log_loss',cv=inner)
    scores=-cross_val_score(search,Xd,yd,scoring='neg_log_loss',cv=outer)
    search.fit(Xd,yd)
    baseline=DummyClassifier(strategy='prior').fit(Xd,yd)
    p=search.predict_proba(Xt); bp=baseline.predict_proba(Xt)
    print('outer_logloss',np.round(scores,6)); print('outer_mean',scores.mean())
    print('selected_C',search.best_params_['logisticregression__C'])
    print('test_logloss baseline',log_loss(yt,p),log_loss(yt,bp))
    print('test_accuracy',accuracy_score(yt,p.argmax(1)))
    assert len(Xt)==120 and len(scores)==5 and np.isfinite(scores).all()
    print('all checks passed')
if __name__ == '__main__': main()
