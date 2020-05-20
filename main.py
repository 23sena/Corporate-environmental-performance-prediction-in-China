from XGBoost import plot_tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_csv, read_excel
import XGBoost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
from sklearn import svm
import shap
 
data= pd.read_excel("data/raw.xlsx")
data = data.drop(['NO.'],axis=1)
label = data.pop('Y2')

def func():
    
    # split data into train and test sets
    seed = 7
    test_size = .25

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_size, random_state=seed)
    original_col = X_train.columns
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(X_train)
    X_train = imp.transform(X_train)
    X_test = imp.transform(X_test)

    # Random Forest 
    regr_rf = RandomForestRegressor(max_depth=30, random_state=2)
    regr_rf.fit(X_train, y_train)
    y_pred_train1= regr_rf.predict(X_train)
    y_pred1 = regr_rf.predict(X_test)
    # random forest end

    # XGBoost 
    xgdmat=xgb.DMatrix(X_train,y_train)
    our_params={'eta':.03,'seed':0,'subsample':0.8,\
                'colsample_bytree':0.8,'objective':'reg:linear',\
                'max_depth':7,'min_child_weight':.5}
    final_gb=xgb.train(our_params,xgdmat,num_boost_round=1500)
    testmat = xgb.DMatrix(X_test)
    trainmat=xgb.DMatrix(X_train)
    y_pred2 = final_gb.predict(testmat)
    y_pred_train2= final_gb.predict(trainmat)
 
    # SVM
    clf = svm.SVR(kernel='rbf', degree = 3, gamma = 'auto', coef0=0.0, tol=0.1, C=1.0, epsilon=0.1, shrinking = True, cache_size=200, verbose=False, max_iter=-1)
    clf.fit(X_train, y_train)
    y_pred_train3 = clf.predict(X_train)
    y_pred3 = clf.predict(X_test)

    ###### Evaluation ######
    
    # Random Forest 
    mae = mean_absolute_error(y_test.values, y_pred1)
    print("MAE: %.5f" % mae)
    rmse =np.sqrt(mean_squared_error(y_test.values, y_pred1))
    print("RMSE: %.5f" % rmse)
    R = np.corrcoef(y_test.values,y_pred1)
    print("Correlation Coef: %.5f" % R[0,1])
    r2 = r2_score(y_test.values,y_pred1)
    print("r2 score: %.5f" % r2)

    # XGBoost
    mae = mean_absolute_error(y_test.values, y_pred2)
    print("MAE: %.5f" % mae)
    rmse =np.sqrt(mean_squared_error(y_test.values, y_pred2))
    print("RMSE: %.5f" % rmse)
    R = np.corrcoef(y_test.values,y_pred2)
    print("Correlation Coef: %.5f" % R[0,1])
    r2 = r2_score(y_test.values,y_pred2)
    print("r2 score: %.5f" % r2)

    # SVM
    mae = mean_absolute_error(y_test.values, y_pred3)
    print("MAE: %.5f" % mae)
    rmse =np.sqrt(mean_squared_error(y_test.values, y_pred3))
    print("RMSE: %.5f" % rmse)
    R = np.corrcoef(y_test.values,y_pred3)
    print("Correlation Coef: %.5f" % R[0,1])
    r2 = r2_score(y_test.values,y_pred3)
    print("r2 score: %.5f" % r2)

    ###### Visualization ######
    
    # plot predict error
    plt.gcf().set_size_inches((10, 4))
    plt.plot(((y_pred1-y_test.values)/y_test.values)[::8], color='g', marker='*', label='random forest')
    plt.plot(((y_pred2-y_test.values)/y_test.values)[::8], color='c', marker='s', markerfacecolor='none', label='XGBoost')
    plt.plot(((y_pred3-y_test.values)/y_test.values)[::8], color='y', marker='o', markerfacecolor='none', label='SVM')
    # plt.gca().legend()
    plt.legend(loc='upper right')
    plt.savefig('junk.jpg')

    # plot training error
    plt.gcf().set_size_inches((10, 4))
    plt.plot(((y_pred_train1-y_train.values)/y_train.values)[::20], color='g', marker='*', label='random forest')
    plt.plot(((y_pred_train2-y_train.values)/y_train.values)[::20], color='c', marker='s', markerfacecolor='none', label='XGBoost')
    plt.plot(((y_pred_train3-y_train.values)/y_train.values)[::20],color='y', marker='o', markerfacecolor='none', label='SVM')
    # plt.gca().legend()
    plt.legend(loc='upper right')
    plt.savefig('junk.jpg')

    # plot predictions on test split
    plt.gcf().set_size_inches((10, 4))
    plt.plot(y_test.values[::3], color='b', label='value')
    plt.plot(y_pred1[::3], color='g', marker='*', markerfacecolor='none', label='random forest',linestyle='None')
    plt.plot(y_pred2[::3], color='c', marker='s', markerfacecolor='none', label='XGBoost',linestyle='None')
    plt.plot(y_pred3[::3], color='y', marker='o', markerfacecolor='none', label='SVM',linestyle='None')
    # plt.gca().legend()
    plt.legend(loc='upper right')
    plt.savefig('junk.jpg')

    # plot predictions on training split
    plt.gcf().set_size_inches((10, 4))
    plt.plot(y_train.values[::10], color='b', label='value')
    plt.plot(y_pred_train1[::10], color='g', marker='*', markerfacecolor='none', label='random forest',linestyle='None')
    plt.plot(y_pred_train2[::10], color='c', marker='s', markerfacecolor='none', label='XGBoost',linestyle='None')
    plt.plot(y_pred_train3[::10], color='y', marker='o', markerfacecolor='none', label='SVM',linestyle='None')
    # plt.gca().legend()
    plt.legend(loc='upper right')
    plt.savefig('junk2.jpg')

    # shap the value for better visualization
    shap.initjs()
    shap_values = shap.TreeExplainer(final_gb).shap_values(X_train)
    X_train = pd.DataFrame(data=X_train, columns=original_col)
    X_train = X_train.rename(columns={
        "X2": "X7", "X3": "X6","X4":"X14","X5":"X4","X6":"X8","X7":"X9","X8":"X10","X9":"X12",
        "X10":"X11","X11":"X13","X12":"X5","X13":"X1","X14":"X2","X15":"X3"})
    shap.summary_plot(shap_values, X_train)
    
if __name__ == "__main__":
    func()
