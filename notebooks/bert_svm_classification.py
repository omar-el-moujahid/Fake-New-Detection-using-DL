import pandas as pd 
import numpy as np
from tqdm.auto import tqdm
import torch 
import torch.nn as nn
import torch.optim as optim 
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer , DistilBertModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from transformers import BertModel, BertTokenizer


data_train=pd.read_csv("Constraint_English_Train - Sheet1.csv")
data_test=pd.read_csv("Constraint_English_Test - Sheet1.csv")
data_val=pd.read_csv("Constraint_English_Val - Sheet1.csv")


from sklearn.preprocessing import LabelEncoder
l_enc = LabelEncoder()
data_train["label_1"]=l_enc.fit_transform(data_train["label"])
# data_test["label_1"]=l_enc.fit_transform(data_test["label"])
data_val["label_1"]=l_enc.fit_transform(data_val["label"])

