# Jon-Paul Boyd - Kaggle - Santander Customer Value Prediction 
# Importing the libraries
import sys
import gc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

def handle_missing(df):
    
    # Delete any column containing only 0's
    df = df.loc[:, (df != 0).any(axis=0)]
    df.fillna(0, inplace=True)
    
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

corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
k = 30 #number of variables for heatmap
cols = corrmat.nlargest(k, 'target')['target'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

cols = cols.drop(['target'])

y = np.log1p(df_train.target)

## Possibly try a non-linear version of PCA - check thge udemy course ?????
cols = get_selected_features()

df_train = df_train[cols]    
df_train = handle_missing(df_train)

id_test = df_test.ID
df_test = df_test[cols]    
df_test = handle_missing(df_test)


df_train = np.log1p(df_train)
df_test = np.log1p(df_test)

gc.collect()

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso

# Convert train to numpy array and delete index column
X = np.array(df_train)
X = np.delete(X, 0, axis=1)

test_errors_regr_lasso = []
test_errors_regr_ridge = []
test_errors_regr_gbr = []
test_errors_regr_enet = []
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
    
    # lasso
    regr_lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0003, random_state=1, max_iter=50000))
    regr_lasso.fit(X_train, y_train)
    regr_lasso_train_pred = regr_lasso.predict(X_train)
    regr_lasso_test_pred = regr_lasso.predict(X_test)


    # Ridge
    regr_ridge = Ridge(alpha=9.0, fit_intercept = True)
    regr_ridge.fit(X_train, y_train)
    regr_ridge_train_pred = regr_ridge.predict(X_train)
    regr_ridge_test_pred = regr_ridge.predict(X_test)


    # Gradient Boosting    
    regr_gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.02,
                                          max_depth=4, max_features='sqrt',
                                          min_samples_leaf=15, min_samples_split=50,
                                          loss='huber', random_state = 5)         
    regr_gbr.fit(X_train, y_train)
    regr_gbr_train_pred = regr_gbr.predict(X_train)
    regr_gbr_test_pred = regr_gbr.predict(X_test)
        
       
    # Elastic Net
    regr_enet = make_pipeline(RobustScaler(), ElasticNet(alpha=4.0, l1_ratio=0.005, random_state=3))
    regr_enet.fit(X_train, y_train)
    regr_enet_train_pred = regr_enet.predict(X_train) 
    regr_enet_test_pred = regr_enet.predict(X_test) 
    
        
    # Stacking
    stacked_set = pd.DataFrame({'A' : []})
    stacked_set = pd.concat([stacked_set, pd.DataFrame(regr_lasso_test_pred)], axis=1)
    stacked_set = pd.concat([stacked_set, pd.DataFrame(regr_ridge_test_pred)], axis=1)
    stacked_set = pd.concat([stacked_set, pd.DataFrame(regr_gbr_test_pred)], axis=1) 
    stacked_set = pd.concat([stacked_set, pd.DataFrame(regr_enet_test_pred)], axis=1)
    product = (regr_lasso_test_pred*regr_ridge_test_pred*regr_gbr_test_pred*regr_enet_test_pred) ** (1.0/4.0)
    stacked_set = pd.concat([stacked_set, pd.DataFrame(product)], axis=1)
    Xstack = np.array(stacked_set)
    Xstack = np.delete(Xstack, 0, axis=1)
    regr_lasso_stacked = Lasso(alpha = 0.0001,fit_intercept = True)
    regr_lasso_stacked.fit(Xstack, y_test)
    regr_lasso_stacked_Xstack_pred = regr_lasso_stacked.predict(Xstack)
    
    models.append([regr_ridge, regr_lasso, regr_gbr, regr_enet, regr_lasso_stacked])
    #models.append([regr_ridge, regr_lasso, regr_gbr, regr_enet])
    
    test_errors_regr_lasso.append(np.square(regr_lasso_test_pred - y_test).mean() ** 0.5)
    test_errors_regr_ridge.append(np.square(regr_ridge_test_pred - y_test).mean() ** 0.5)
    test_errors_regr_gbr.append(np.square(regr_gbr_test_pred - y_test).mean() ** 0.5)
    test_errors_regr_enet.append(np.square(regr_enet_test_pred - y_test).mean() ** 0.5)
    test_errors_regr_lasso_stacked.append(np.square(regr_lasso_stacked_Xstack_pred - y_test).mean() ** 0.5)


print('Lasso test error: ', np.mean(test_errors_regr_lasso))
print('Ridge test error: ', np.mean(test_errors_regr_ridge))
print('Gradient Boosting test error: ', np.mean(test_errors_regr_gbr))
print('Elastic Net test error: ', np.mean(test_errors_regr_enet))
print('Lasso stacked test error: ', np.mean(test_errors_regr_lasso_stacked))


# Convert test to numpy array and delete index column
X_score = np.array(df_test)
X_score = np.delete(X_score, 0, axis=1)
M = X_score.shape[0]
scores_final = 1+np.zeros(M)

for model in models:
    model_lasso = model[0]
    model_ridge = model[1]
    model_gbr = model[2]
    model_enet = model[3]
    model_lasso_stacked = model[4]
    
    model_lasso_scores = model_lasso.predict(X_score)
    model_ridge_scores = model_ridge.predict(X_score)
    model_gbr_scores = model_gbr.predict(X_score)
    model_enet_scores = model_enet.predict(X_score)
    
    stacked_sets = pd.DataFrame({'A' : []})
    stacked_sets = pd.concat([stacked_sets, pd.DataFrame(model_lasso_scores)],axis=1)
    stacked_sets = pd.concat([stacked_sets, pd.DataFrame(model_ridge_scores)],axis=1)
    stacked_sets = pd.concat([stacked_sets, pd.DataFrame(model_gbr_scores)],axis=1)
    stacked_sets = pd.concat([stacked_sets, pd.DataFrame(model_enet_scores)],axis=1)
    
    product = (model_lasso_scores*model_ridge_scores*model_gbr_scores*model_enet_scores) ** (1.0/4.0)
    stacked_sets = pd.concat([stacked_sets, pd.DataFrame(product)], axis=1)    
    Xstacks = np.array(stacked_sets)
    Xstacks = np.delete(Xstacks, 0, axis=1)
    scores_final = scores_final * model_lasso_stacked.predict(Xstacks)


scores_final = scores_final ** (1/nFolds)

# Output    
fin_score = pd.DataFrame({'target': np.exp(scores_final)-1})
fin_data = pd.concat([id_test, fin_score],axis=1)
fin_data.to_csv('test_set_prediction.csv', sep=',', index = False)
      