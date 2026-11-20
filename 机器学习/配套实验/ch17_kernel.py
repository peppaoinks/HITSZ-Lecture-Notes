"""实验17.2：核分类、训练内标准化、C/gamma选择与核矩阵核对。"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,balanced_accuracy_score


def rbf(A,B,gamma):
    # 教学规模显式差数组，避免平方范数相减造成的小负距离。
    return np.exp(-gamma*np.sum((A[:,None,:]-B[None,:,:])**2,axis=2))


def main():
    X,y=make_circles(n_samples=240,factor=.4,noise=.10,random_state=17)
    Xt,Xe,yt,ye=train_test_split(X,y,test_size=.25,stratify=y,random_state=17)
    cv=StratifiedKFold(3,shuffle=True,random_state=17)
    pipe=Pipeline([('scale',StandardScaler()),('svc',SVC())])
    grid=[{'svc__kernel':['linear'],'svc__C':[.1,1,10]},
          {'svc__kernel':['rbf'],'svc__C':[.1,1,10],'svc__gamma':[.1,1,10]}]
    search=GridSearchCV(pipe,grid,cv=cv,scoring='balanced_accuracy',refit=True).fit(Xt,yt)
    best=search.best_estimator_; score=best.decision_function(Xe); pred=best.predict(Xe)
    svc=best.named_steps['svc']; scale=best.named_steps['scale']
    np.testing.assert_allclose(scale.mean_,Xt.mean(axis=0))
    Zt=scale.transform(Xt); Ze=scale.transform(Xe)
    if svc.kernel=='rbf':
        K=rbf(Zt,Zt,svc.gamma)
        assert np.linalg.eigvalsh(K).min()>-1e-9
        pre=SVC(kernel='precomputed',C=svc.C,tol=svc.tol).fit(K,yt)
        np.testing.assert_allclose(pre.decision_function(rbf(Ze,Zt,svc.gamma)),score,atol=1e-8)
        reconstructed=rbf(Ze,svc.support_vectors_,svc.gamma)@svc.dual_coef_[0]+svc.intercept_[0]
        np.testing.assert_allclose(reconstructed,score,atol=1e-10)
    out=Path(__file__).resolve().parent/'ch17_results'; out.mkdir(exist_ok=True)
    with (out/'cv_scores.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['params','mean_cv_balanced_accuracy','std_cv'])
        for p,m,s in zip(search.cv_results_['params'],search.cv_results_['mean_test_score'],search.cv_results_['std_test_score']):
            writer.writerow([str(p),m,s])
    fig,ax=plt.subplots(figsize=(6.4,5.0))
    axis=np.linspace(-1.6,1.6,220); xx,yy=np.meshgrid(axis,axis)
    z=best.decision_function(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx,yy,z>0,levels=[-.5,.5,1.5],colors=['#F2F6F8','#D5E2EA'])
    ax.contour(xx,yy,z,levels=[-1,0,1],colors='#35566E',linestyles=['--','-','--'])
    for c,marker in [(0,'o'),(1,'^')]:
        ax.scatter(Xt[yt==c,0],Xt[yt==c,1],s=18,marker=marker,facecolors='none',edgecolors='#35566E',label=f'class {c}, training')
    sv=Xt[svc.support_]; ax.scatter(sv[:,0],sv[:,1],s=65,facecolors='none',edgecolors='#777',linewidths=.7,label='support vectors')
    ax.set(xlabel='x1 (original coordinates)',ylabel='x2 (original coordinates)',title='Training data and selected SVM decision boundary',aspect='equal')
    ax.legend(fontsize=8,loc='upper right'); fig.tight_layout(); fig.savefig(out/'kernel_boundary.png',dpi=160); plt.close(fig)
    print('best params',search.best_params_)
    print('mean validation balanced accuracy',search.best_score_)
    print('test accuracy / balanced accuracy',accuracy_score(ye,pred),balanced_accuracy_score(ye,pred))
    print('support vectors',len(svc.support_),'of',len(Xt))
    print('all checks passed')

if __name__ == '__main__': main()
