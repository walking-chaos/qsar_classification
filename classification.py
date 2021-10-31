import pandas as pd
import numpy as np
from scipy.stats import mode
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import seaborn as sns
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def stats_data(data, cols_we_need):
    for i in cols_we_need:
        print(i)
        descr_stat(data[i])
        distplot(data[i])

def descr_stat(x):
    print('Max: {0}\nMin: {1}\nMode: {2}\nMedian: {3}\nMean: {4}\nShapiro-Wilk test: {5}\nVariance: {6}\nStandart deviation: {7}\nRange of values (max-min): {8}'\
          .format(np.max(x), np.min(x), mode(x).mode, np.median(x), np.mean(x), shapiro(x), np.var(x), np.std(x), np.ptp(x)))

def distplot(x):
    sns.displot(x)

def cor_table(data, descriptors):
    display(data[descriptors].corr())

def t_test(X_train, X_test, descriptors):
    for i in descriptors:
        print(i + ':')
        print(ttest_ind(X_train[i], X_test[i]))

def MW_test(X_train, X_test, descriptors):
    for i in descriptors:
        print(i + ':')
        print(mannwhitneyu(X_train[i], X_test[i]))

def train_test(data, descriptors, dist):
    y = data['pIC50']
    X = data[descriptors]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    if dist == "normal":
        t_test(X_train, X_test, descriptors)
    elif dist == "abnormal":
        MW_test(X_train, X_test, descriptors)
    else:
        print("Error")
    return X_train, X_test, y_train, y_test

class Classifier:
    
    def __init__(self, X_train, X_test, y_train_regr, y_test_regr, algorithm, threshold, hyperparameters):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train_regr = y_train_regr
        self.y_test_regr = y_test_regr
        self.algorithm = algorithm
        self.threshold = threshold
        self.hyperparameters = hyperparameters

    def random_forest_model(self):
        clf_rf = RandomForestClassifier()
        randomized_search_cv_clf_rf = RandomizedSearchCV(clf_rf, self.hyperparameters, cv=5, scoring='roc_auc')
        randomized_search_cv_clf_rf.fit(self.X_train, self.y_train)
        self.best_model = randomized_search_cv_clf_rf.best_estimator_

    def logistic_regression_model(self):
        log_reg = LogisticRegressionCV(cv=5)
        log_reg.fit(self.X_train, self.y_train)
        self.best_model = log_reg

    def ridge_classifier_model(self):
        ridge_clf = RidgeClassifier()
        randomized_search_cv_ridge = RandomizedSearchCV(ridge_clf, self.hyperparameters, cv=5, scoring='roc_auc')
        randomized_search_cv_ridge.fit(self.X_train, self.y_train)
        self.best_model = randomized_search_cv_ridge.best_estimator_

    def knn_classifier_model(self):
        knn_clf = KNeighborsClassifier()
        randomized_search_cv_knn_clf = RandomizedSearchCV(knn_clf, self.hyperparameters, cv=5, scoring='roc_auc')
        randomized_search_cv_knn_clf.fit(self.X_train, self.y_train)
        self.best_model = randomized_search_cv_knn_clf.best_estimator_

    def svm_classifier_model(self):
        svm_clf = SVC()
        randomized_search_cv_svm_clf = RandomizedSearchCV(svm_clf, self.hyperparameters, cv=5, scoring='roc_auc')
        randomized_search_cv_svm_clf.fit(self.X_train, self.y_train)
        self.best_model = randomized_search_cv_svm_clf.best_estimator_

    def gbc_classifier_model(self):
        gbc_clf = GradientBoostingClassifier()
        randomized_search_cv_gbc_clf = RandomizedSearchCV(gbc_clf, self.hyperparameters, cv=5, scoring='roc_auc')
        randomized_search_cv_gbc_clf.fit(self.X_train, self.y_train)
        self.best_model = randomized_search_cv_gbc_clf.best_estimator_

    def mlp_classifier_model(self):
        mlp_clf = MLPClassifier()
        randomized_search_cv_mlp_clf = RandomizedSearchCV(mlp_clf, self.hyperparameters, cv=5, scoring='roc_auc')
        randomized_search_cv_mlp_clf.fit(self.X_train, self.y_train)
        self.best_model = randomized_search_cv_mlp_clf.best_estimator_

    def evaluate_model(self):
        self.y_test_predicted = self.best_model.predict(self.X_test)
        self.y_train_predicted = self.best_model.predict(self.X_train)
        self.accuracy_train = accuracy_score(self.y_train, self.y_train_predicted)
        self.accuracy_test = accuracy_score(self.y_test, self.y_test_predicted)
        self.precision = precision_score(self.y_test, self.y_test_predicted)
        self.recall = recall_score(self.y_test, self.y_test_predicted)
        self.f1score = f1_score(self.y_test, self.y_test_predicted)
        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_test, self.y_test_predicted)
        self.rocaucscore = auc(self.fpr, self.tpr)
        
        print(
            'Accuracy_test: {0} \nAccuracy_train: {1} \nPrecision score: {2} \nRecall score: {3} \nF1-score: {4} \nROC-AUC-score: {5}' \
                .format(self.accuracy_test, self.accuracy_train, self.precision, self.recall, self.f1score, self.rocaucscore))
        ConfusionMatrixDisplay.from_predictions(self.y_test, self.y_test_predicted)
        plt.grid(False)
        plt.show()
        RocCurveDisplay.from_estimator(self.best_model, self.X_test, self.y_test)
        plt.show()

    def feature_importances(self):
        importances = list(self.best_model.feature_importances_)
        feature_importances = [(feature, importance) for feature, importance in
                               zip(list(self.X_test.columns), importances)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True) 
        x_values = list(range(len(importances)))
        plt.bar(x_values, importances, orientation='vertical')
        plt.xticks(x_values, list(self.X_test.columns), rotation='vertical')
        plt.ylabel('Importance')
        plt.xlabel('Variable')
        plt.title('Variable Importances')
        plt.show()

    def threshold_making(self):
        self.y_train = np.where(self.y_train_regr >= self.threshold, 1, 0)
        self.y_test = np.where(self.y_test_regr >= self.threshold, 1, 0)


    def model_building(self):
        self.threshold_making()
        if self.algorithm == "random_forest":
            self.random_forest_model()
            self.evaluate_model()
            self.feature_importances()
        elif self.algorithm == "logistic_regression":
            self.logistic_regression_model()
            self.evaluate_model()
        elif self.algorithm == "ridge_classifier":
            self.ridge_classifier_model()
            self.evaluate_model()
        elif self.algorithm == "knn_classifier":
            self.X_train = np.array(self.X_train)
            self.y_train = np.array(self.y_train)
            self.X_test = np.array(self.X_test)
            self.y_test = np.array(self.y_test)
            self.knn_classifier_model()
            self.evaluate_model()
        elif self.algorithm == "svm_classifier":
            self.svm_classifier_model()
            self.evaluate_model()
        elif self.algorithm == "gbc_classifier":
            self.gbc_classifier_model()
            self.evaluate_model()
        elif self.algorithm == "mlp_classifier":
            self.mlp_classifier_model()
            self.evaluate_model()
            
    def predict(self, X):
        return self.best_model.predict(X)

    def model_parameters(self):
        return self.best_model.get_params()

    def metrics(self):
        metrics_dict = {'accuracy_train' : self.accuracy_train,
                        'accuracy_test' : self.accuracy_test,
                        'precision' : self.precision,
                        'recall' : self.recall,
                        'f1_score' : self.f1score,
                        'roc_auc_score' : self.rocaucscore}
        return metrics_dict










        
       
        
    
    
    
        
                 