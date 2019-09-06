import pandas as pd
import numpy as np
import random
from scipy import stats

from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   OneHotEncoder, Binarizer, KBinsDiscretizer)
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.pipeline import Pipeline

from sklearn.model_selection import (train_test_split, cross_validate, GroupShuffleSplit,
                                     GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit)
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from treeinterpreter import treeinterpreter as ti
from scipy.stats import chi2_contingency

import warnings
warnings.filterwarnings("ignore")


def get_missing_row_col(train):

    missing_col = pd.DataFrame(train.isnull().sum() / len(train),
                               columns=['missing_rate'])
    missing_col = missing_col.loc[missing_col['missing_rate'] >= 0.7]
    col_drop_list = missing_col.index.tolist()

    missing_row = pd.DataFrame(train.isnull().sum(axis=1) / train.shape[1],
                               columns=['missing_rate'])
    missing_row = missing_row.loc[missing_row['missing_rate'] >= 0.8]
    row_drop_list = missing_row.index.tolist()

    return row_drop_list, col_drop_list


def get_high_corr_list(train):
    cor_df = train.corr()
    cor_df = pd.DataFrame(cor_df.stack(-1), columns=['corr']).reset_index()
    cor_df = cor_df.loc[cor_df['corr'] >= 0.9]
    cor_df = cor_df.loc[cor_df['level_0'] != cor_df['level_1']]
    cor_df = cor_df.loc[cor_df['level_0'] != 'TARGET'].loc[cor_df['level_1'] != 'TARGET']
    cor_target = train.corr()[['TARGET']]
    cor_target = cor_target.reset_index()
    cor_df = cor_df.merge(cor_target, left_on='level_0', right_on='index', how='left')
    cor_df = cor_df.merge(cor_target, left_on='level_1', right_on='index', how='left')

    cor_df.TARGET_x = cor_df.TARGET_x.apply(lambda x: abs(x))
    cor_df.TARGET_y = cor_df.TARGET_y.apply(lambda x: abs(x))

    dro_list_1 = cor_df.loc[cor_df['TARGET_x'] >= cor_df['TARGET_y']].level_1.tolist()
    dro_list_2 = cor_df.loc[cor_df['TARGET_x'] <= cor_df['TARGET_y']].level_0.tolist()

    drop_list = list(set(dro_list_1 + dro_list_2))

    return drop_list


def get_binary_list(train):
    binary_list = []
    for col in train.columns.tolist():
        tmp_list = train[col].unique().tolist()
        if len(tmp_list) == 2:
            if 0 in tmp_list:
                if 1 in tmp_list:
                    binary_list = binary_list + [col]
        else:
            pass
    binary_list = [x for x in binary_list if x not in ['TARGET']]

    return binary_list


def get_binary_obj_list(train, binary_list):
    binary_obj_list = []
    for col in train.columns.tolist():
        tmp_list = train[col].unique().tolist()
        if (len(tmp_list) == 2) and (col not in binary_list):
            binary_obj_list = binary_obj_list + [col]
        else:
            pass
    binary_obj_list = [x for x in binary_obj_list if (
            x not in ['TARGET']) and 'FLAG' in x]

    return binary_obj_list


def get_cate_var_transformed(train, binary_list, binary_obj_list):
    cate_list = train.select_dtypes('object').columns.tolist()
    cate_list = [x for x in cate_list if (x not in binary_list)
                 and (x not in binary_obj_list)]
    train_cate = pd.get_dummies(train[cate_list].fillna('missing'))
    train = train.drop(columns=cate_list)
    train = pd.concat([train, train_cate], axis=1)
    binary_cate_list = train_cate.columns.tolist()

    return train, binary_cate_list


def get_num_var_transformed(train, binary_list, binary_obj_list, binary_cate_list):
    num_list = train.select_dtypes('number').columns.tolist()
    num_list = [x for x in num_list if x not in binary_list+binary_obj_list+binary_cate_list+['TARGET']]
    train[num_list] = train[num_list].apply(lambda x: x.fillna(x.median()), axis=1)

    log_df = pd.DataFrame(train[num_list].skew(axis=0))
    log_df = log_df.loc[log_df[0] > 0]
    log_list = log_df.index.tolist()
    train[log_list] = train[log_list].apply(lambda x: np.log(x+1), axis=1)
    train[log_list] = train[log_list].replace([-np.inf, np.inf], np.nan)
    train[log_list] = train[log_list].apply(lambda x: x.fillna(np.min(x)), axis=1)

    return train, num_list, log_list


def get_u_test_drop_list(train):
    feature_list = [x for x in train.columns.tolist() if x not in ['TARGET']]
    drop_list = []
    for col in feature_list:
        _, p_value = stats.mannwhitneyu(train[col], train['TARGET'])
        if p_value > 0.005:
            drop_list.append(col)
    return drop_list


def get_num_var_scaled(train):

    print('get numeric variable scaled!')

    rob = RobustScaler(with_scaling=False, with_centering=False,
                       quantile_range=(0.1, 0.9))
    train[num_list] = rob.fit_transform(train[num_list])

    minmax = MinMaxScaler()
    train[num_list] = minmax.fit_transform(train[num_list])

    return train


def check_for_null(train):
    null = pd.DataFrame(train.isnull().sum())
    null = null.loc[null[0] > 0]
    null_list = null.index.tolist()
    if len(null_list) > 0:
        raise ValueError('NA in df!!!')
    else:
        print('No NA in dataset!')


if __name__ == '__main__':

    train = pd.read_csv('data/application_train.csv')
    train = train.loc[train['FLAG_OWN_REALTY'] == 'Y']

    train = train.set_index(['SK_ID_CURR'])

    # missing
    row_drop_list, col_drop_list = get_missing_row_col(train)
    train = train.drop(index=row_drop_list)
    train = train.drop(columns=col_drop_list)

    # high correlation
    print('Start dealing with high corr variables...')
    drop_list = get_high_corr_list(train)
    train = train.drop(columns=drop_list)
    print('High corr variables dropped!')

    # get binary variables
    print('Start dealing with binary variables...')
    binary_list = get_binary_list(train)
    train[binary_list] = train[binary_list].fillna(0)

    binary_obj_list = get_binary_obj_list(train, binary_list)
    binary_obj_dict = {'N': 0, 'Y': 0}
    train[binary_obj_list] = train[binary_obj_list].replace(binary_obj_dict)
    train[binary_obj_list] = train[binary_obj_list].fillna(0)
    print('Binary variables done!')

    # get categorical variables
    print('Start dealing with categorical variables...')
    train, binary_cate_list = get_cate_var_transformed(
        train, binary_list, binary_obj_list)
    print('Categorical variables done!')

    # get numeric variables
    print('Start dealing with numeric variables...')
    train, num_list, log_list = get_num_var_transformed(
        train, binary_list, binary_obj_list, binary_cate_list)
    print('Numeric variables done!')

    train = get_num_var_scaled(train)

    u_test_drop_list = get_u_test_drop_list(train)
    train = train.drop(columns=u_test_drop_list)

    # with dummies generated, there must be other variables with high correlation
    drop_list_2 = get_high_corr_list(train)
    train = train.drop(columns=drop_list_2)

    X = train[[x for x in train.columns.tolist() if x != 'TARGET']]
    X = X.replace([np.inf, -np.inf], np.nan)
    y = train[['TARGET']]

    check_for_null(X)

    print('Start modeling...')
    rfc = RandomForestClassifier(
        bootstrap=True,
        n_estimators=182,
        max_depth=8,
        max_features='auto',
        criterion='gini',
        class_weight='balanced',
        oob_score=True
    )

    lgb = LGBMClassifier(boosting_type='goss',
                         max_depth=5,
                         learning_rate=0.05,
                         n_estimators=457,
                         class_weight='balanced',
                         )
    model = lgb
    model.fit(X, y)

    # para = {'model': [lgb],
    #         'model__n_estimators': range(50, 500),
    #         'model__max_depth': range(-1, 20),
    #         'model__criterion': ['gini', 'entropy'],
    #         'model__max_features': ['auto', 'log2']}

    para_lgb = {'n_estimators': range(50, 500),
                'max_depth': range(-1, 15),
                'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
                'boosting_type': ['gbdt', 'dart', 'goss']}

    # rand = RandomizedSearchCV(model, param_distributions=para, cv=8,
    #                           error_score='ignore', verbose=300, n_iter=30)
    cv = StratifiedShuffleSplit(n_splits=4, test_size=0.25)
    rand_lgb = RandomizedSearchCV(estimator=model, scoring='roc_auc',
                                  param_distributions=para_lgb, cv=cv,
                                  error_score='raise', verbose=300, n_iter=80)
    # rand.fit(X, y)
    # rand.best_score_
    # rand.best_params_

    rand_lgb.fit(X, y)
    print(rand_lgb.best_score_)
    # home + no home: 0.7515454919910335
    # home: 0.7511267288999272

    print(rand_lgb.best_params_)
    # home + no home: {'n_estimators': 472, 'max_depth': 10,
    # 'learning_rate': 0.1, 'boosting_type': 'dart'}
    # home: {'n_estimators': 457, 'max_depth': 5,
    # 'learning_rate': 0.05, 'boosting_type': 'goss'}

    y_pred = model.predict(X)
    precision = metrics.precision_score(y, y_pred)
    recall = metrics.recall_score(y, y_pred)
    f1 = 2*(precision*recall)/(precision+recall)
    auc = metrics.roc_auc_score(y, y_pred)
    print('precision: {} \n'.format(precision),
          'recall: {}\n'.format(recall),
          'f1: {}\n'.format(f1),
          'auc: {}\n'.format(auc))

    # cv_mode

    cv = StratifiedShuffleSplit(n_splits=4, test_size=0.25)
    cv_results = cross_validate(estimator=model, X=X, y=y,
                                scoring=('roc_auc', 'f1', 'recall', 'precision'),
                                cv=cv, verbose=100, error_score='raise_deprecating')
    cv_results_df = pd.DataFrame(cv_results)
    cv_results_mean = cv_results_df.apply(lambda x: x.mean(), axis=0)
    print(cv_results_mean)
    # lgb - home + no_home
    # test_roc_auc       0.751579
    # test_f1            0.267573
    # test_recall        0.655374
    # test_precision     0.168108

    # lgb - home
    # test_roc_auc       0.752301
    # test_f1            0.270546
    # test_recall        0.649258
    # test_precision     0.170876

    print('cv_mode done! Start checking cv results with Xsquare test...')

    xsq_data = pd.DataFrame(cv_results).iloc[:, 2:]
    chi2 = chi2_contingency(xsq_data)
    print('chi2_value: {}; p-value: {}; degree of freedom: {}'.format(chi2[0], chi2[1],
                                                                      chi2[2]))
    print('Fail to reject H0! cv_results accepted!')
    print('Xsquare test done!')

    # feature importance
    features = pd.DataFrame(X.columns.tolist(), columns=['feature'])
    features['score'] = model.feature_importances_
    features = features.sort_values(by=['score'], ascending=False)

    # let's do signal_score!
    #
    # idx_len = len(X)
    # rounds = len(X)//1000 + 1
    # cont = X.copy()
    # print('total rounds: {}'.format(rounds))
    #
    # for round in range(0, rounds):
    #     print('round {}'.format(round))
    #     x = X[round*1000: min((round + 1)*1000, idx_len)]
    #     pred, bias, contribution = ti.predict(model, x)
    #     contribution_df = np.array(contribution)[:, :, 1]
    #     cont.iloc[round*1000: min((round + 1)*1000, idx_len)] = contribution_df

    # print('contributions for each sample done!')