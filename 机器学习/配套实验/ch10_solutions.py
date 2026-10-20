"""第10章状态不变、手算缩放与类别词表核对。"""
import numpy as np
import pandas as pd
from ch10_pipeline import make_pipeline

def main():
    x=np.array([2.,4.,6.]); np.testing.assert_allclose((8-x.mean())/x.std(),np.sqrt(6))
    np.testing.assert_allclose(np.nanmedian([1,np.nan,5,9]),5)
    model=make_pipeline().fit(pd.DataFrame({'weight':[1.,2.,np.nan,4.,5.,6.], 'zone':['A','A','B','B','A','B']}),[3,5,8,9,11,13])
    prep=model.named_steps['prep']; num=prep.named_transformers_['num']; cat=prep.named_transformers_['cat']
    med=num.named_steps['fill'].statistics_.copy()
    vocabulary=[a.copy() for a in cat.named_steps['encode'].categories_]
    model.predict(pd.DataFrame({'weight':[100.,np.nan], 'zone':['C','D']}))
    np.testing.assert_array_equal(med,num.named_steps['fill'].statistics_)
    for a,b in zip(vocabulary,cat.named_steps['encode'].categories_):np.testing.assert_array_equal(a,b)
    print('10.2 sqrt(6); 10.3 median=5; 10.4 scaled=1.5')
    print('10.6 IDF',1+np.log(101/10),'10.7 target encoding',(1+3*.2)/5)
    print('training median and category vocabulary unchanged; all checks passed')
if __name__=='__main__':main()
