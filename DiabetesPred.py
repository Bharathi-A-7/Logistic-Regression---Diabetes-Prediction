
# Range defaults to starting from 0 and doesnt include the last value 


from math import exp
from csv import reader
from random import randrange

def load_file(file):
    data = list()
    with open(file,'r') as file:
        csv_read = reader(file)
        for row in csv_read:
            if not row:
                continue
            data.append(row)
    return data


# Converts string data types to float
def str_to_float(data,column):
    for row in data:
        row[column] = float(row[column].strip())

# Returns the maximum and minimum values in each column
def maxmin(data):
    maxmin = list()
    for i in range(len(data[0])):
        column_values = [row[i] for row in data]
        minVal = min(column_values)
        maxVal = max(column_values)
        maxmin.append([minVal,maxVal])
    return maxmin

#Normalizing the data so that they lie between 0 and 1
def normalize(data,maxmin):
    for row in data:
        for i in range(len(row)):
            row[i] = (row[i] - maxmin[i][0]) / (maxmin[i][1] - maxmin[i][0])
 

#Splitting the data into random groups           
def split_into_folds(data,n_folds):
    data_split = list()
    data_copy = list(data)
    fold_size = int(len(data)/n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            indices = randrange(len(data_copy))
            fold.append(data_copy.pop(indices))
        data_split.append(fold)
    return data_split


#Accuracy Calculation
def accuracy_calc(original,predicted):
    correct_pred = 0;
    for i in range(len(original)):
        if(original[i] == predicted[i]):
            correct_pred += 1
    return correct_pred /float(len(original)) *100.0


#Testing the logistic regression algorithm
def eval_algorithm(data,logistic,n_folds,*args):
    folds = split_into_folds(data,n_folds)
    accuracies = list()
    for fold in folds :
        train_data = list(folds)
        train_data.remove(fold)
        train_data = sum(train_data,[])
        test_data = list()
        for row in fold:
            row_copy = list(row)
            test_data.append(row_copy)
            row_copy[-1] = None
        predicted =  logistic(train_data,test_data,*args)
        original = [row[-1] for row in fold]
        accuracy = accuracy_calc(original,predicted)
        accuracies.append(accuracy)
    return accuracies


def predict(data,coefficients):
    y_pred = coefficients[0]
    for i in range(len(data)-1):
        y_pred = y_pred + coefficients[i+1]*data[i]
    logit = 1.0/(1.0 + exp(-y_pred))
    return logit 


def learn_coefficients(train_data,alpha,n_loop):
    coefficients = [0.0 for i in range(len(train_data[0]))]
    for n in range(n_loop):
        for row in train_data:
            y_pred = predict(row,coefficients)
            error = row[-1] - y_pred
            coefficients[0] =  coefficients[0] + alpha * error * y_pred * (1 - y_pred)
            for i in range(len(row)-1):
                coefficients[i+1] = coefficients[i+1] + alpha * error * y_pred * (1-y_pred)*row[i]
                
    return coefficients

def logistic_regression(train_data,test_data,alpha,n_loop):
    prediction = list()
    coefficients = learn_coefficients(train_data,alpha,n_loop)
    for row in test_data:
        y_pred = predict(row,coefficients)
        y_pred = round(y_pred)
        print("Predicted :",y_pred)
        prediction.append(y_pred)
    return prediction 


# Loading the data     
file = 'D:/IV Semester/Predictive Analytics/Logistic Regression/diabetes.csv'
data = load_file(file)
#Converting column types from string to float
for i in range(len(data[0])):
    str_to_float(data,i)
#Normalizing the values
maxmin = maxmin(data)
normalize(data,maxmin)

n_folds = 5
alpha = 0.1
n_loop = 100

accuracies = eval_algorithm(data,logistic_regression,n_folds,alpha,n_loop)
print(accuracies)

print(" Prediction = 0 : No Diabetes")
print(" Prediction = 1 : Diabetes")

    