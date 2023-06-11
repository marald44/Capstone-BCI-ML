# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 01:06:56 2023

@author: maral.demirsecen

3 LDAs

"""

import numpy as np
from scipy.io import loadmat
#ml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
#accuracies
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
#preprocessing
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
# to read real time data
import scipy.io as sio 

# Load the data
data_left_forward = loadmat(r'C:\Users\Dell\Downloads\eeg-data-w-combinations\LeftVSFoot_8Features(M2).mat')
data_right_forward = loadmat(r'C:\Users\Dell\Downloads\eeg-data-w-combinations\RightVSFoot_8Features(M1).mat')
data_right_left = loadmat(r'C:\Users\Dell\Downloads\eeg-data-w-combinations\LeftVSRight_8Features(M3).mat')

# Extract data
data_lf = data_left_forward['M2']
data_rf = data_right_forward['M1']
data_rl = data_right_left['M3']

# Extract features and labels for each class combination
X_lf = data_lf[:, :-1]
y_lf = data_lf[:, -1]

X_rf = data_rf[:, :-1]
y_rf = data_rf[:, -1]

X_rl = data_rl[:, :-1]
y_rl = data_rl[:, -1]


        #preprocessing
        
    # IQR - removing outliers
    
        #removing outliers for left-forward
Q1 = np.percentile(X_lf, 25, axis=0, interpolation='midpoint')
Q3 = np.percentile(X_lf, 75, axis=0, interpolation='midpoint')
IQR = Q3 - Q1

    # Above Upper bound
upper = Q3 + 1.5 * IQR
upper_array = np.array(X_lf >= upper)

    # Below Lower bound
lower = Q1 - 1.5 * IQR
lower_array = np.array(X_lf <= lower)

    # True values (outliers) in the array
upper_indices = np.argwhere(upper_array)
lower_indices = np.argwhere(lower_array)

    # dropping the outliers
if upper_indices.size > 0:
    X_lf = np.delete(X_lf, upper_indices[:, 0], axis=0)
    y_lf = np.delete(y_lf, upper_indices[:, 0], axis=0)

if lower_indices.size > 0:
    X_lf = np.delete(X_lf, lower_indices[:, 0], axis=0)
    y_lf = np.delete(y_lf, lower_indices[:, 0], axis=0)

        #removing outliers for right-forward
Q1_rf = np.percentile(X_rf, 25, axis=0, interpolation='midpoint')
Q3_rf = np.percentile(X_rf, 75, axis=0, interpolation='midpoint')
IQR_rf = Q3_rf - Q1_rf

    # Above Upper bound
upper_rf = Q3_rf + 1.5 * IQR_rf
upper_array_rf = np.array(X_rf >= upper_rf)

    # Below Lower bound
lower_rf = Q1_rf - 1.5 * IQR_rf
lower_array_rf = np.array(X_rf <= lower_rf)

    # True values (outliers) in the array
upper_indices_rf = np.argwhere(upper_array_rf)
lower_indices_rf = np.argwhere(lower_array_rf)

    # dropping the outliers
if upper_indices_rf.size > 0:
    X_rf = np.delete(X_rf, upper_indices_rf[:, 0], axis=0)
    y_rf = np.delete(y_rf, upper_indices_rf[:, 0], axis=0)

if lower_indices_rf.size > 0:
    X_rf = np.delete(X_rf, lower_indices_rf[:, 0], axis=0)
    y_rf = np.delete(y_rf, lower_indices_rf[:, 0], axis=0)



        #removing outliers for right-forward
Q1_rf = np.percentile(X_rf, 25, axis=0, interpolation='midpoint')
Q3_rf = np.percentile(X_rf, 75, axis=0, interpolation='midpoint')
IQR_rf = Q3_rf - Q1_rf

    # Above Upper bound
upper_rf = Q3_rf + 1.5 * IQR_rf
upper_array_rf = np.array(X_rf >= upper_rf)

    # Below Lower bound
lower_rf = Q1_rf - 1.5 * IQR_rf
lower_array_rf = np.array(X_rf <= lower_rf)

    # True values (outliers) in the array
upper_indices_rf = np.argwhere(upper_array_rf)
lower_indices_rf = np.argwhere(lower_array_rf)

    # dropping the outliers
if upper_indices_rf.size > 0:
    X_rf = np.delete(X_rf, upper_indices_rf[:, 0], axis=0)
    y_rf = np.delete(y_rf, upper_indices_rf[:, 0], axis=0)

if lower_indices_rf.size > 0:
    X_rf = np.delete(X_rf, lower_indices_rf[:, 0], axis=0)
    y_rf = np.delete(y_rf, lower_indices_rf[:, 0], axis=0)

        # removing outliers for right-left
Q1_rl = np.percentile(X_rl, 25, axis=0, interpolation='midpoint')
Q3_rl = np.percentile(X_rl, 75, axis=0, interpolation='midpoint')
IQR_rl = Q3_rl - Q1_rl

    # Above Upper bound
upper_rl = Q3_rl + 1.5 * IQR_rl
upper_array_rl = np.array(X_rl >= upper_rl)

    # Below Lower bound
lower_rl = Q1_rl - 1.5 * IQR_rl
lower_array_rl = np.array(X_rl <= lower_rl)

    # True values (outliers) in the array
upper_indices_rl = np.argwhere(upper_array_rl)
lower_indices_rl = np.argwhere(lower_array_rl)

    # dropping the outliers
if upper_indices_rl.size > 0:
    X_rl = np.delete(X_rl, upper_indices_rl[:, 0], axis=0)
    y_rl = np.delete(y_rl, upper_indices_rl[:, 0], axis=0)

if lower_indices_rl.size > 0:
    X_rl = np.delete(X_rl, lower_indices_rl[:, 0], axis=0)
    y_rl = np.delete(y_rl, lower_indices_rl[:, 0], axis=0)



    # Scale the features using MinMaxScaler 
scaler = MinMaxScaler()
X_lf = scaler.fit_transform(X_lf)
X_rf = scaler.fit_transform(X_rf)
X_rl = scaler.fit_transform(X_rl)



# Apply SMOTE to balance class distribution
smote = SMOTE()
X_lf, y_lf = smote.fit_resample(X_lf, y_lf)
X_rf, y_rf = smote.fit_resample(X_rf, y_rf)
X_rl, y_rl = smote.fit_resample(X_rl, y_rl)



    # Define the class combinations
class_combinations = [(1, 2), (1, 3), (2, 3)]
#left:1 right:2 forward:3  


    # Train LDA models for each class combination
ldas = []

for combination in class_combinations:
    class1, class2 = combination

    # Prepare the data for the current combination
    X = np.concatenate((X_lf[y_lf == class1], X_lf[y_lf == class2]))
    y = np.concatenate((np.zeros(np.sum(y_lf == class1)), np.ones(np.sum(y_lf == class2)))) #0-1 for differenting between 2 classes

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the LDA model
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    # Predict the class labels for the test set
    y_pred = lda.predict(X_test)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall =  recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    mse=mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2= r2_score(y_test, y_pred)
    print(f"Accuracy for classes {class1}-{class2}: {accuracy}")
    print(f"Precision for classes {class1}-{class2}: {precision}")
    print(f"Recall for classes {class1}-{class2}: {recall}")
    print(f"F1-score for classes {class1}-{class2}: {f1}")
    print(f"Mean Squared Error for classes {class1}-{class2}: {mse}")
    print(f"Mean Absolute Percentage Error for classes {class1}-{class2}: {mape}")
    print(f"R^2 Score for classes {class1}-{class2}: {r2}")

    # Add the trained LDA model to the list
    ldas.append(lda)


  
    # Make predictions on a single sample
#example sample points
sample = np.array([-1.43589, -3.31068, -3.14798, -4.06314, -5.37112, -4.65965, -4.94756, -5.71435]) #3
sample= np.array([-3.09556,	-3.64001,	-3.27239,	-4.28112,	-5.19645,	-4.50978,	-4.30835,	-5.82496]) #3
sample= np.array([-3.49385,	-4.22158,	-3.72729,	-4.74264,	-2.90173,	-4.41674,	-2.75319,	-3.0339]) #2
sample =  np.array([-3.6087,	-4.35362,	-4.06211,	-4.84057,	-3.82706,	-3.89551,	-3.31436,	-3.33825]) #1 


# Function to predict the class of a sample
def predict_class(sample):
    predictions = []
    
    for lda in ldas:
        prediction = lda.predict([sample])
        predictions.append(prediction[0])
    
    # Map the predictions to the corresponding classes
    result = []
    
    if predictions[0] == 0 and predictions[1] == 0 and predictions[2] == 0:
        result.append('left')
    elif predictions[0] == 1 and predictions[1] == 0 and predictions[2] == 0:
        result.append('not classified')
    elif predictions[0] == 0 and predictions[1] == 1 and predictions[2] == 0:
        result.append('forward')
    elif predictions[0] == 1 and predictions[1] == 1 and predictions[2] == 0:
        result.append('forward')
    elif predictions[0] == 0 and predictions[1] == 0 and predictions[2] == 1:
        result.append('left')
    elif predictions[0] == 1 and predictions[1] == 0 and predictions[2] == 1:
        result.append('right')
    elif predictions[0] == 0 and predictions[1] == 1 and predictions[2] == 1:
        result.append('not classified')
    elif predictions[0] == 1 and predictions[1] == 1 and predictions[2] == 1:
        result.append('right')
    
    return result

print("Predicted class:", predict_class(sample))
#accuracy is really goof but without overfitting - tried on validation sample

"""
if the result is - then class is: 
    null-0-0 : left ;
    1-0-0 not classified;
    0-1-0 forward; 
    1-1-0 forward;
    0-0-1 left;
    1-0-1 right;
    0-1-1 not classified;
    1-1-1 right
    #not classified = stop
"""


        #Real Time  Prediction  
    
while True:
    # Get input from user
    user_input = input("Enter 8 values (separated by spaces): ")

    # Split the input string into a list of values
    values = user_input.split() #need to seperated by space or change split()'s inside part

    # Convert the values to float and create a numpy array
    sample = np.array([float(val) for val in values])

    # Make prediction using the predict_class function
    predicted_class = predict_class(sample)

    # Print the predicted class
    print("Predicted class:", predicted_class)
    
#-1.43589 -3.31068 -3.14798 -4.06314 -5.37112 -4.65965 -4.94756 -5.71435

