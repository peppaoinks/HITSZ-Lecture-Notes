"""实验14.2补充：仅作开发诊断的学习曲线，所有尺度在子训练集拟合。"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def main():
    X,y=make_classification(n_samples=700,n_features=18,n_informative=5,n_redundant=3,flip_y=.08,random_state=144)
    A,B,a,b=train_test_split(X,y,test_size=200,stratify=y,random_state=144)
    rows=[]
    for C in [.01,10.]:
        for size in [50,150,400]:
            tr=[];va=[]
            for seed in [1,2,3]:
                ix,_=train_test_split(np.arange(len(a)),train_size=size,stratify=a,random_state=seed)
                pipe=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=2000)).fit(A[ix],a[ix])
                np.testing.assert_allclose(pipe[0].mean_,A[ix].mean(0),atol=1e-12)
                tr.append(log_loss(a[ix],pipe.predict_proba(A[ix])))
                va.append(log_loss(b,pipe.predict_proba(B)))
            row=[C,size,np.mean(tr),np.std(tr,ddof=1),np.mean(va),np.std(va,ddof=1)]
            rows.append(row); print('C n train_mean train_sd valid_mean valid_sd',*[round(v,6) for v in row])
    out=Path(__file__).resolve().parent/'ch14_results';out.mkdir(exist_ok=True)
    with (out/'learning_curves.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['C','n','train_mean','train_sd','valid_mean','valid_sd']);w.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
    for ax,C in zip(axes,[.01,10.]):
        a=np.array([r for r in rows if r[0]==C]);ax.set_title(f'C={C:g}')
        ax.errorbar(a[:,1],a[:,2],yerr=a[:,3],label='train',color='#35566E',marker='o')
        ax.errorbar(a[:,1],a[:,4],yerr=a[:,5],label='validation',color='#8C6545',marker='s')
        ax.set_xlabel('Training samples');ax.set_ylabel('Log loss');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=160);plt.close(fig)
    print('all checks passed')
if __name__=='__main__': main()
