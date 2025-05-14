from enum import Enum

class Model(Enum):
    DT = "DecisionTree"
    RF = "RandomForest" 
    SVM = "SVM"
    LR = "LogisticRegression"
    XGB = "XGBoost" 