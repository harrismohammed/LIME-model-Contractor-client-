#Data Preprocessing

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13 ].values

Y = dataset.iloc[:,13].values 

# Drop unwanted columns
Xa = dataset.drop(['RowNumber','CustomerId','Surname','Exited',],axis=1)
# Missing Data#PART 1

# Extract features
float_columns=[]
cat_columns=[]
int_columns=[]
    
for i in Xa.columns:
    if Xa[i].dtype == 'float' : 
        float_columns.append(i)
    elif Xa[i].dtype == 'int64':
        int_columns.append(i)
    elif Xa[i].dtype == 'object':
        cat_columns.append(i)
        
train_cat_features = Xa[cat_columns]
train_float_features = Xa[float_columns]
train_int_features = Xa[int_columns]


## Transformation of categorical columns
# Label Encoding:
from sklearn.preprocessing import LabelEncoder
train_cat_features_ver2 = train_cat_features.apply(LabelEncoder().fit_transform)



## Transformation of float columns
# Rescale data (between 0 and 1)   
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler(feature_range=(0, 1))
    
for i in train_float_features.columns:
    X_temp = train_float_features[i].reshape(-1,1)
    train_float_features[i] = scaler.fit_transform(X_temp)

#### Finalize 
temp_1 = np.concatenate((train_cat_features_ver2,train_float_features),axis=1)
train_transformed_features = np.concatenate((temp_1,train_int_features),axis=1)
train_transformed_features = pd.DataFrame(data=train_transformed_features)
    
array = train_transformed_features.values
number_of_features = len(array[0])

X = array[:,0:number_of_features]


#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = 0.2, random_state = 0)



#PART 2 
#Building the ANN

#Importing Keras libraries and required packages
import keras
from keras.models import Sequential #To Initialise the ANN
from keras.layers import Dense #Create layers in ANN
from keras.layers import Dropout #Import dropout
#Dropout can be added to single layers or many layers
#Dropout is used to take care of overfitting

#Initialsing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 10))

##Adding the input layer and the first hidden layer with dropout : improves result 
classifier.add(Dropout(rate =0.1))
#rate can be 0.1,0.2,0.3,0.4


#Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

#Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
 
 #Compile the whole ANN (back propagation)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics= ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)



#PART 3 - MAKING THE PREDICTIONS AND EVALUATING THE MODEL

#Predicting the test set results
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)
#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


#Part 4 - Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 10))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics= ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size =10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X= X_train, y= Y_train, cv = 10, n_jobs = 1)

mean = accuracies.mean()
variance = accuracies.std()

# LIME SECTION
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
from __future__ import print_function





predict_fn_classifier = lambda x: classifier.predict_proba(x).astype(float)


# Line-up the feature names

feature_names_cat = list(train_cat_features_ver2)
feature_names_float = list(train_float_features)
feature_names_int = list(train_int_features)



feature_names = sum([feature_names_cat, feature_names_float, feature_names_int], [])
print(feature_names)


feature_names = sum([feature_names_float, feature_names_int], [])
print(feature_names)


# Create the LIME Explainer


explainer = lime.lime_tabular.LimeTabularExplainer(X_train ,feature_names = feature_names,class_names=['WILL QUIT','WILL NOT QUIT',],
                                                    kernel_width=3)

# Pick the observation in the validation set for which explanation is required
observation_1 = 2

# Get the explanation for Keras Classifier
exp = explainer.explain_instance(X_test[observation_1], predict_fn_classifier, num_features=8)
exp.show_in_notebook(show_all=False)

