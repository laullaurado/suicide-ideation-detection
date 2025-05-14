import pandas as pd
from prepro import prepro
from decoder import create_tfidf_features
from models import train_and_evaluate_model,Model


df = pd.read_csv('./data_train(in).csv', encoding='latin-1')

df = prepro(df)

X_tfidf = create_tfidf_features(df)

train_and_evaluate_model(X_tfidf, df['is_suicide'], Model.LR)
