#importing the most relevant libraries
import pandas as pd #for data manipulation and further data analysis
from sklearn.svm import SVC #for support vector classifictaion
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class HeartDiseasePredictor:# creating a class for Heart Disease Predicting as a way to train data in predicting whether or not someone has heart disease.

    def __init__(self, dataset): #initailising the dataset so it can be used in the class.
        #Beginning the process of data cleaning so that the data will be clean and easy to use.
        self.dataset = dataset.dropna() #the 'dropna' function is used here for dropping any missing values.
        self.dataset = dataset.rename(columns={'target': 'Target'})


        if 'restecg' in self.dataset.columns:
            self.dataset = self.dataset.drop(columns=['restecg']) #dropping the 'rest ECG' column as its unnecessary when predicting heart disease. This is mainly due to low correlation in studies.


        self.algo = {
            # introducing the algorithms I will use, namely KNN,SVM and Logisitic Regression
            'knn': KNeighborsClassifier(), #used KNN because it is good for handling health datasets and numerical values. However it typically requires a really good set of data, which I believe I have obtained.
            'svm': SVC(), #Similar to KNN, SVC is also great for health datsets. Mainly because it has the ability to create wider margins between different health conditions.
            'logistic_regression': LogisticRegression(max_iter = 1000) #I also used Logisitc Regression model because it is easier to intepret.
        }
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def data_prep(self):#Included a data preparation function and included the datashape so it would be easier to picture the amount of data we are using. This can be beneficial when trying to figure out the validity and the representativeness of data.
        print('Dataset shape:',self.dataset.shape)

        X = self.dataset.drop('Target',axis =1)
        y = self.dataset['Target']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2,random_state=42, shuffle=False)

    def knn_train(self, X_train, y_train):#Incorporating the KNN model into the class
        knn = self.algo['knn']
        knn.fit(self.X_train, self.y_train)#The data that I pass here will train the model
        return knn

    def svm_train(self, X_train, y_train):#Incorporating the SVC model into the class
        svm = self.algo['svm']
        svm.fit(X_train, y_train)#Data passed will train the model
        return svm

    def log_regr_train(self, X_train, y_train):#Incorporating the Logistic Regression model into the class
        logistic_regression = self.algo['logistic_regression']
        logistic_regression.fit(X_train, y_train)#Data passed will train the model
        return logistic_regression

    def algo_outcome(self, algo, X_test, y_test):
        y_est = algo.predict(self.X_test)  #Est stands for estimate
        accuracy = accuracy_score(self.y_test, y_est) #Using accuracy score from sklearn metrics library to train the model and get our accurate answer
        mse = mean_squared_error(self.y_test, y_est) #Using the mean square error from sklearn metrics library to find out the mean square error
        return accuracy, mse #Asking the code to return MSE and the Accuracy once its has been calculated

dataset = pd.read_csv('/Users/naledi/Desktop/heart.csv').dropna() #Loading the data
dataset = dataset.drop(columns=['restecg'],errors='ignore')
# This data was removed because I deemed it unnecessary to try to figure out whether there is a relationship between the features and someone having heart disease and ignores any errors python might bring up in the case of the column already being removed.
print(dataset)#The print function allows for us to print the data without the 'restecg; column

predictor = HeartDiseasePredictor(dataset) #The initialisation of the instance so python can use the clean version of the data.
predictor.data_prep() #Allows the data to be prepared through the data_prep function we called in the class. Initailly it allows us to split the data so we can begin using it.

#Training the models with the clean data
knn_mod = predictor.knn_train(predictor.X_test,predictor.y_test)
knn_accuracy, knn_mse = predictor.algo_outcome(knn_mod, predictor.X_test,predictor.y_test) #lgo_outcome calculates the accuracy of the KNN model
#Training the SVC model with the clean data
svm_mod = predictor.svm_train(predictor.X_test,predictor.y_test)
svm_accuracy, svm_mse = predictor.algo_outcome(svm_mod, predictor.X_test, predictor.y_test)#algo_outcome calculates the accuracy of the KNN model
#Training the Logistic Regression model with the clean data
logistic_mod = predictor.log_regr_train(predictor.X_test,predictor.y_test)
logistic_accuracy, logistic_mse = predictor.algo_outcome(logistic_mod, predictor.X_test, predictor.y_test)#algo_outcome calculates the accuracy of the KNN model

print(f" Accuracy for KNN model: {knn_accuracy:.2f}, Mean Sqaure Root Error: {knn_mse:.2f}") #Allowing the code to transcribe its findings for the KNN model
print(f" Accuracy for SVM model:{svm_accuracy:.2f}, Mean Sqaure Root Error: {svm_mse:.2f}")#Allowing the code to transcribe its findings for the SVC model
print(f" Accuracy for Logistic Regression: {logistic_accuracy:.2f}, Mean Square Root Error: {logistic_mse:.2f}")#Allowing the code to transcribe its findings for the logistic regression model


