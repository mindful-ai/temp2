import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from datetime import datetime
import pickle
import os, os.path

class housing(object):

    def __init__(self):
        self.dataset = r'C:/Users/mindf/Desktop/sapient-us/linear-regression/USA_Housing.csv'
        self.modelrepo = r"C:\Users\mindf\Desktop\step\WEEK-05\model-repository"
        self.data    = []
        self.X       = []
        self.y       = []
        self.X_train = []
        self.y_train = []
        self.X_test  = []
        self.y_test  = []
        self.model   = []
        self.metrics = {}

    def log(self, message):
        t = datetime.now()
        f = "%A, %d %B %Y, %I:%M %p"
        print(t.strftime(f), ' >> ', message)

 
    def acquire_data(self):
        self.log("Acquiring Data")
        try:
            self.data = pd.read_csv(self.dataset)
            return (True, )
        except Exception as error_message:
            return (False, error_message)
        
    def transform_data(self):
        self.log("Transforming Data")
        try:
            # handle null values: dropping | imputation
            # feature engineering
            # separate the target and features
            self.y = self.data['Price']
            self.X = self.data.drop(['Price', 'Address'], axis = 1)
            return (True, )
        except Exception as error_message:
            return (False, error_message)
        
    def split_data(self):
        self.log("Splitting Data")
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=100)
            return (True, )
        except Exception as error_message:
            return (False, error_message)

    def train_model(self):
        self.log("Training the Model")
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        return 

    def evaluate_model(self):
        self.log("Evaluating the model")
        predictions = self.model.predict(self.X_test)
        self.metrics['RMSE'] = np.sqrt(metrics.mean_squared_error(self.y_test, predictions))
        self.metrics['MAE'] = metrics.mean_absolute_error(self.y_test, predictions)
        return (True, 'metrics', self.metrics)

    def pipeline(self):
        self.log("Running training pipeline...\n")
        self.acquire_data()
        self.transform_data()
        self.split_data()
        self.train_model()
        self.evaluate_model()

    def getmodel(self):
        return (True, 'model', self.model)

    def savemodel(self):
        with open(os.path.join(self.modelrepo, 'housingmodel.pkl'), 'wb') as file:
            pickle.dump(self.model, file)

    def loadmodel(self):
        with open(os.path.join(self.modelrepo, 'housingmodel.pkl'), 'rb') as file:
            self.model = pickle.load(file)
            return self.model
        


if __name__ == "__main__":
    #housingObj = housing()
    #housingObj.pipeline()

    #model = housingObj.getmodel()[2]
    #print(model.predict([[100000, 15, 5, 3, 50000]]))

    #housingObj.savemodel()

    housingObj = housing()
    model = housingObj.loadmodel()
    
    print('After loading model: ', model.predict([[100000, 15, 5, 3, 50000]]))

    

