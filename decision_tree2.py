# Decision Tree
# week1 assignment for Machine learning for Data Analysis

"""
Created on Mon Jul 25 11:47:24 2016

@author: taehee jeong
"""

#%% import packages
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics
from sklearn import tree
from sklearn.externals.six import StringIO  
import pydotplus

#%% Load data

data = pd.read_csv("C:/Bigdata/Data Analysis and Interpretation/Dataset/Adolescent Health/tree_addhealth.csv")

data_clean = data.dropna()

data_clean.dtypes
data_clean.describe()

# features and target


all_features =['BIO_SEX','HISPANIC','WHITE','BLACK','NAMERICAN','ASIAN',
'age','ALCEVR1','ALCPROBS1','marever1','cocever1','inhever1','cigavail','DEP1',
'ESTEEM1','VIOL1','PASSIST','DEVIANT1','SCHCONN1','GPA1','EXPEL1','FAMCONCT','PARACTV',
'PARPRES']

my_features=['marever1','ALCEVR1']
predictors = data_clean[my_features]

targets = data_clean.TREG1

#%%Split into training and testing sets

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.4)

#pred_train.shape
#pred_test.shape
#tar_train.shape
#tar_test.shape

#%% Build a decision tree model


#Build model on training data
classifier=DecisionTreeClassifier()
classifier=classifier.fit(pred_train,tar_train)

# apply decision tree model to get prediction for test data
predictions=classifier.predict(pred_test)


# Confusion Matrix
#              +---------------------------------------------+
#              |                Predicted label              |
#              +----------------------+----------------------+
#              |          (+1)        |         (-1)         |
#+-------+-----+----------------------+----------------------+
#| True  |(+1) | # of true positives  | # of false negatives |
#| label +-----+----------------------+----------------------+
#|       |(-1) | # of false positives | # of true negatives  |
#+-------+-----+----------------------+----------------------+
sklearn.metrics.confusion_matrix(tar_test,predictions)

# Accuracy
accuracy = sklearn.metrics.accuracy_score(tar_test, predictions)
print "Test Accuracy: %s" % accuracy

#%%Displaying the decision tree

# after install graphviz
#pip uninstall pyparsing
#pip install -Iv https://pypi.python.org/packages/source/p/pyparsing/pyparsing-1.5.7.tar.gz#md5=9be0fcdcc595199c646ab317c1d9a709
#pip install pydot


out_data = StringIO()
tree.export_graphviz(classifier, out_file=out_data)


graph=pydotplus.graph_from_dot_data(out_data.getvalue())
with open('C:/Bigdata/Data Analysis and Interpretation/Machine learning for Data Analysis/wk1/tree_smok_mar_alchol.png', 'wb') as f:
    f.write(graph.create_png())

#from IPython.display import Image
#Image(graph.create_png())
#graph.write_pdf("C:/Bigdata\Data Analysis and Interpretation/Machine learning for Data Analysis/wk1/tree_sex.pdf") 
