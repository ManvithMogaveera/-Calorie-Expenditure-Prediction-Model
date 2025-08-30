import os 
import joblib
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import (accuracy_score,classification_report,root_mean_squared_error,roc_curve,auc)
from sklearn.model_selection import RandomizedSearchCV,cross_val_score,StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


MODEL_FILE = 'model.pkl'
PIPELINE_FILE = 'pipeline.pkl'

def builded_pipeline(num_attrib,cat_attrib):
    num_pipeline = Pipeline([
    ("impute",SimpleImputer(strategy="median")),
    ("scaler" ,StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("encode",OneHotEncoder())
    ])
    full_pipeline = ColumnTransformer([
        ("num",num_pipeline,num_attrib),
        ("cat",cat_pipeline,cat_attrib)
    ])
    
    return full_pipeline
if not os.path.exists(MODEL_FILE):
    original_data = pd.read_csv("train.csv")


    # original_data['weight_cat'] = pd.cut(original_data["Weight"],bins=[35,63,74,87,100,np.inf],labels=[1,2,3,4,5])
    
    # splitter = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    # for train_set,test_set in splitter.split(original_data,original_data["weight_cat"]):
    #     strata_train = original_data.loc[train_set].drop("weight_cat",axis=1)
    #     strata_test = original_data.loc[test_set].drop("weight_cat",axis=1)
    
    X_train = original_data.drop("Calories",axis=1)
    Y_train = original_data["Calories"]
    

    num_attrib = X_train.drop("Sex",axis=1).columns.tolist()
    cat_attrib = ["Sex"]
    
    pipeline_new = builded_pipeline(num_attrib,cat_attrib)
    transformed_data = pipeline_new.fit_transform(X_train)
    model  = LGBMRegressor(max_depth=5, n_estimators=150, num_leaves=50,learning_rat = 0.1,random_state = 42)
    model.fit(transformed_data,Y_train)
    prediction =model.predict(transformed_data)
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline_new,PIPELINE_FILE)    
    print("MODEL HAS BEEN TRAINED !")
else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    input_data = pd.read_csv("test.csv")
    transformed_input = pipeline.transform(input_data)
    prediction = model.predict(transformed_input)
    prediction = np.maximum(0, prediction)
    input_data["Calories"] = prediction
    input_data.to_csv("test.csv",index=False)
    print("DONE and saved in test.csv ,Kindly check")