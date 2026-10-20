"""实验10.1：列预处理、训练状态与未见类别；教学构造数据。"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

def make_pipeline():
    numeric = Pipeline([('fill',SimpleImputer(strategy='median')),
                        ('scale',StandardScaler())])
    category = Pipeline([('fill',SimpleImputer(strategy='most_frequent')),
                         ('encode',OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    prep = ColumnTransformer([('num', numeric, ['weight']), ('cat', category, ['zone'])])
    return Pipeline([('prep',prep),('model',Ridge(alpha=1.))])

def main():
    train = pd.DataFrame({'weight':[1.,2.,np.nan,4.,5.,6.],
                          'zone':['A','A','B','B','A','B']})
    y = np.array([3.,5.,8.,9.,11.,13.])
    valid = pd.DataFrame({'weight':[3.,np.nan,10.], 'zone':['A','C','B']})
    model = make_pipeline().fit(train,y)
    numeric = model.named_steps['prep'].named_transformers_['num']
    median = numeric.named_steps['fill'].statistics_.copy()
    means = numeric.named_steps['scale'].mean_.copy()
    z = model.named_steps['prep'].transform(valid)
    print('training_median',median); print('training_mean',means)
    print('valid_transformed',np.round(z,6)); print('prediction',np.round(model.predict(valid),6))
    np.testing.assert_allclose(median,[4.]); np.testing.assert_allclose(means,[22/6])
    assert np.all(z[1,1:] == 0)
    model.predict(pd.DataFrame({'weight':[1e6], 'zone':['new']}))
    np.testing.assert_array_equal(numeric.named_steps['scale'].mean_,means)
    print('all checks passed')
if __name__ == '__main__': main()
