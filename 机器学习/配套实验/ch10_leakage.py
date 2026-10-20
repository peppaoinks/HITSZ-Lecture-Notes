"""实验10.2：同一用户重复记录时，随机划分与新用户划分。"""
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score

def memorize(train_ids, train_y, ids):
    table = dict(zip(train_ids.tolist(),train_y.tolist()))
    fallback = int(np.mean(train_y) >= .5)
    return np.array([table.get(int(x),fallback) for x in ids])

def main():
    rng = np.random.default_rng(1002)
    group = np.repeat(np.arange(200),10)
    user_y = rng.integers(0,2,size=200); y = user_y[group]
    ix = np.arange(len(y))
    a,b = train_test_split(ix,test_size=.25,random_state=10,stratify=y)
    c,d = next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=10).split(ix,y,group))
    for label,tr,va in [('row',a,b),('group',c,d)]:
        overlap = len(set(group[tr]) & set(group[va]))
        score = accuracy_score(y[va],memorize(group[tr],y[tr],group[va]))
        print(f'{label} overlap_users={overlap} accuracy={score:.6f}')
    assert not (set(group[c]) & set(group[d]))
    assert len(set(group[a]) & set(group[b])) > 180
    print('all checks passed')
if __name__ == '__main__': main()
