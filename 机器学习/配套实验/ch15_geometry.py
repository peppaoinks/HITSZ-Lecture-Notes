"""实验15.2：无关特征与距离集中；全部转换在训练折内拟合。"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.metrics import accuracy_score

def main():
    rng=np.random.default_rng(1502)
    for d in [2,20,200]:
        A=rng.normal(size=(10000,d)); B=rng.normal(size=(10000,d))
        sq=np.sum((A-B)**2,axis=1)
        cv=sq.std()/sq.mean()
        print('dimension',d,'squared_distance_CV',cv,'theory',np.sqrt(2/d))
        assert abs(cv-np.sqrt(2/d))<.05
    X,y=make_classification(n_samples=260,n_features=8,n_informative=2,n_redundant=0,n_clusters_per_class=1,class_sep=1.3,random_state=1502)
    X[:,0]*=100
    A,B,a,b=train_test_split(X,y,test_size=.25,stratify=y,random_state=15)
    cv=StratifiedKFold(3,shuffle=True,random_state=151)
    for name,transform in [('scaled','passthrough'),('NCA',NeighborhoodComponentsAnalysis(n_components=2,max_iter=60,random_state=15))]:
        pipe=Pipeline([('scale',StandardScaler()),('metric',transform),('knn',KNeighborsClassifier())])
        search=GridSearchCV(pipe,{'knn__n_neighbors':[3,7,15]},cv=cv,scoring='accuracy').fit(A,a)
        print(name,'k',search.best_params_['knn__n_neighbors'],'cv',search.best_score_,'test',accuracy_score(b,search.predict(B)))
    print('all checks passed')
if __name__ == '__main__': main()
