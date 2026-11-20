"""实验16.2：训练内词表、分层交叉验证与一次最终测试。"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix


def data():
    # 人工构造两类短语；空格明确词元边界，省去中文分词工具。
    train0 = ['会议 时间 通知', '课程 作业 通知', '会议 地点 安排', '课程 时间 安排',
              '作业 提交 时间', '课程 资料 通知', '会议 资料 安排', '作业 时间 提醒',
              '课程 地点 通知', '会议 提交 资料', '作业 资料 安排', '课程 提交 提醒']
    train1 = ['优惠 活动 立即', '中奖 奖品 领取', '优惠 奖品 立即', '活动 领取 奖品',
              '中奖 立即 领取', '优惠 活动 领取', '奖品 立即 通知', '中奖 活动 提醒',
              '优惠 领取 通知', '活动 奖品 提醒', '中奖 优惠 奖品', '优惠 立即 提醒']
    test = ['会议 作业 时间', '课程 资料 地点', '会议 安排 明日专用词', '作业 提交 通知',
            '优惠 立即 领取', '中奖 奖品 活动', '优惠 奖品 明日专用词', '中奖 领取 通知']
    return train0+train1, np.repeat([0,1],12), test, np.repeat([0,1],4)


def main():
    train,y,test,yt = data()
    pipe = Pipeline([('vec',CountVectorizer(tokenizer=str.split,token_pattern=None,lowercase=False)),
                     ('nb',MultinomialNB())])
    cv = StratifiedKFold(n_splits=3,shuffle=True,random_state=16)
    search = GridSearchCV(pipe,{'nb__alpha':[.1,1.,10.]},cv=cv,scoring='neg_log_loss',refit=True)
    search.fit(train,y)
    model=search.best_estimator_; probs=model.predict_proba(test); labels=model.predict(test)
    vec=model.named_steps['vec']
    assert '明日专用词' not in vec.vocabulary_
    before=dict(vec.vocabulary_); vec.transform(test); assert before==vec.vocabulary_
    empty=vec.transform(['从未出现词']); assert empty.nnz==0
    np.testing.assert_allclose(model.predict_proba(['从未出现词']), [[.5,.5]],atol=1e-12)
    print('vocabulary size',len(vec.vocabulary_))
    print('best alpha',search.best_params_['nb__alpha'])
    print('mean validation log loss',-search.best_score_)
    print('test labels',labels)
    print('test accuracy / log loss',accuracy_score(yt,labels),log_loss(yt,probs))
    print('confusion matrix',confusion_matrix(yt,labels).tolist())
    print('class 1 posterior',np.round(probs[:,1],6))
    print('unseen-only posterior',model.predict_proba(['从未出现词']))
    print('all checks passed')

if __name__ == '__main__':
    main()
