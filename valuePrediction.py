# Jon-Paul Boyd - Kaggle - Santander Customer Value Prediction 
# Importing the libraries
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb


def handle_missing(df):
    # Delete any column containing only 0's
    df = df.loc[:, (df != 0).any(axis=0)]
    
    # Impute nan with 0
    df.fillna(0, inplace=True)    
    return df

def add_stats(df):
    df['median'] = df.median(axis=1)
    df['mean'] = df.mean(axis=1)
    df['sum'] = df.sum(axis=1)
    df['std'] = df.std(axis=1)
    df['kur'] = df.kurtosis(axis=1)    
    return df


def get_selected_features():
    return [
        'f190486d6', 'c47340d97', 'eeb9cd3aa', '66ace2992', 'e176a204a',
        '491b9ee45', '1db387535', 'c5a231d81', '0572565c2', '024c577b9',
        '15ace8c9f', '23310aa6f', '9fd594eec', '58e2e02e6', '91f701ba2',
        'adb64ff71', '2ec5b290f', '703885424', '26fc93eb7', '6619d81fc',
        '0ff32eb98', '70feb1494', '58e056e12', '1931ccfdd', '1702b5bf0',
        '58232a6fb', '963a49cdc', 'fc99f9426', '241f0f867', '5c6487af1',
        '62e59a501', 'f74e8f13d', 'fb49e4212', '190db8488', '324921c7b',
        'b43a7cfd5', '9306da53f', 'd6bb78916', 'fb0f5dbfe', '6eef030c1'
    ]
#     
#def get_selected_features():
#    return [
#        '555f18bd3', '9fd594eec', '5bc7ab64f', 'cbbc9c431',
#       'f190486d6', '6b119d8ce', 'f74e8f13d', 'ac30af84a', '26fc93eb7',
#       '58e2e02e6', '429687d5a', 'e8d9394a0', '6eef030c1', 'f3cf9341c',
#       'e4159c59e', 'ba4ceabc5', '51707c671', '1702b5bf0', '38e6f8d32',
#       'f296082ec', '41bc25fef', 'f1851d155', '70feb1494', '0d5215715',
#       '6d2ece683', 'ad207f7bb', '174edf08a', '1fd0a1f2a', 'd79736965'
#    ]
    
# Importing the datasets
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_train = handle_missing(df_train)
y = np.log1p(df_train.target)

#corrmat = df_train.corr()
#k = 500 #number of variables for heatmap
#cols = corrmat.nlargest(k, 'target')['target'].index 
#cols = cols.drop(['target'])

cols = get_selected_features()

df_train = df_train[cols]    
df_train = add_stats(df_train)

cols = get_selected_features()
id_test = df_test.ID
df_test = handle_missing(df_test)
df_test = df_test[cols]    
df_test = add_stats(df_test)

# Garbage collect
gc.collect()


# Convert train to numpy array and delete index column
X = np.array(df_train)
X = np.delete(X, 0, axis=1)

test_errors_regr_gbr = []
test_errors_regr_lgb = []
test_errors_regr_rf = []
test_errors_regr_lasso_stacked = []

nFolds = 20
ifold = 1
models = []

kf = KFold(n_splits=nFolds, random_state=241, shuffle=True)

for train_index, test_index in kf.split(X):
    print('fold: ',ifold)
    ifold = ifold + 1    
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    
    # Gradient Boosting    
    regr_gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.02,
                                          max_depth=4, max_features='sqrt',
                                          min_samples_leaf=15, min_samples_split=50,
                                          loss='huber', random_state = 5)         
    regr_gbr.fit(X_train, y_train)
    regr_gbr_train_pred = regr_gbr.predict(X_train)
    regr_gbr_test_pred = regr_gbr.predict(X_test)

    
    # Light gradient boost
    regr_lgb = lgb.LGBMRegressor(objective = 'regression',
        num_leaves= 58,
        subsample = 0.6143,
        colsample_bytree = 0.6453,
        min_split_gain = np.power(10, -2.5988),
        reg_alpha = np.power(10, -2.2887),
        reg_lambda = np.power(10, 1.7570),
        min_child_weight = np.power(10, -0.1477),
        verbose = -1,
        seed = 3,
        boosting_type = 'gbdt',
        max_depth = -1,
        learning_rate = 0.05,
        metric = 'l2')
    regr_lgb.fit(X_train, y_train)
    regr_lgb_train_pred = regr_lgb.predict(X_train)
    regr_lgb_test_pred = regr_lgb.predict(X_test)           
       
    
    # Random Forest regressor    
    regr_rf = RandomForestRegressor(n_estimators = 500, random_state = 0)
    regr_rf.fit(X_train, y_train)
    regr_rf_train_pred = regr_gbr.predict(X_train)
    regr_rf_test_pred = regr_gbr.predict(X_test)
        
    
    # Stacking
    stacked_set = pd.DataFrame({'A' : []})
    stacked_set = pd.concat([stacked_set, pd.DataFrame(regr_gbr_test_pred)], axis=1) 
    stacked_set = pd.concat([stacked_set, pd.DataFrame(regr_rf_test_pred)], axis=1)
    stacked_set = pd.concat([stacked_set, pd.DataFrame(regr_lgb_test_pred)], axis=1)
    product = (regr_gbr_test_pred*regr_lgb_test_pred*regr_rf_test_pred) ** (1.0/3.0)
    stacked_set = pd.concat([stacked_set, pd.DataFrame(product)], axis=1)
    Xstack = np.array(stacked_set)
    Xstack = np.delete(Xstack, 0, axis=1)
    regr_lasso_stacked = Lasso(alpha = 0.0001,fit_intercept = True)
    regr_lasso_stacked.fit(Xstack, y_test)
    regr_lasso_stacked_Xstack_pred = regr_lasso_stacked.predict(Xstack)
    
    models.append([regr_gbr, regr_lgb, regr_rf, regr_lasso_stacked])
    
    test_errors_regr_gbr.append(np.square(regr_gbr_test_pred - y_test).mean() ** 0.5)
    test_errors_regr_lgb.append(np.square(regr_lgb_test_pred - y_test).mean() ** 0.5)
    test_errors_regr_rf.append(np.square(regr_rf_test_pred - y_test).mean() ** 0.5)
    test_errors_regr_lasso_stacked.append(np.square(regr_lasso_stacked_Xstack_pred - y_test).mean() ** 0.5)


print('Gradient Boosting test error: ', np.mean(test_errors_regr_gbr))
print('LGB test error: ', np.mean(test_errors_regr_lgb))
print('Random Forest test error: ', np.mean(test_errors_regr_rf))
print('Lasso stacked test error: ', np.mean(test_errors_regr_lasso_stacked))


# Convert test to numpy array and delete index column
X_score = np.array(df_test)
X_score = np.delete(X_score, 0, axis=1)
M = X_score.shape[0]
scores_final = 1+np.zeros(M)

for model in models:
    model_gbr = model[0]
    model_lgb = model[1]
    model_rf = model[2]
    model_lasso_stacked = model[3]
    
    model_gbr_scores = model_gbr.predict(X_score)
    model_lgb_scores = model_lgb.predict(X_score)
    model_rf_scores = model_rf.predict(X_score)
    
    stacked_sets = pd.DataFrame({'A' : []})
    stacked_sets = pd.concat([stacked_sets, pd.DataFrame(model_gbr_scores)],axis=1)
    stacked_sets = pd.concat([stacked_sets, pd.DataFrame(model_lgb_scores)],axis=1)
    stacked_sets = pd.concat([stacked_sets, pd.DataFrame(model_rf_scores)],axis=1)
    
    product = (model_gbr_scores*model_lgb_scores*model_rf_scores) ** (1.0/3.0)
    stacked_sets = pd.concat([stacked_sets, pd.DataFrame(product)], axis=1)    
    Xstacks = np.array(stacked_sets)
    Xstacks = np.delete(Xstacks, 0, axis=1)
    scores_final = scores_final * model_lasso_stacked.predict(Xstacks)


scores_final = scores_final ** (1/nFolds)

# Output    
fin_score = pd.DataFrame({'target': np.exp(scores_final)-1})
fin_data = pd.concat([id_test, fin_score],axis=1)
fin_data.to_csv('test_set_prediction.csv', sep=',', index = False)