from src.data_preprocessor import DataPreprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class ModelBuilder():
    
    def random_forest(self, X_train, X_test, y_train, y_test):
        rfc = RandomForestClassifier(random_state=0)
        rfc.fit(X_train, y_train)
        y_predicted= rfc.predict(X_test)
        print('Model accuracy score with 10 decision trees : {0:.2f}'.
              format(accuracy_score(y_test, y_predicted)*100))