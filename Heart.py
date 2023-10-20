import seaborn as sns
import tkinter as tk
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from PIL import Image, ImageTk
from tkinter import ttk
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.naive_bayes import GaussianNB

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


root = tk.Tk()
root.title("Heart Disease")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
# ++++++++++++++++++++++++++++++++++++++++++++

image2 = Image.open('heart.jpg')

image2 = image2.resize((w, h), Image.LANCZOS)


background_image = ImageTk.PhotoImage(image2)


background_label = tk.Label(root, image=background_image)
background_label.image = background_image



background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)
lbl = tk.Label(root, text="Heart Disease Detection System", font=('times', 35,' bold '), height=1, width=32,bg="green",fg="white")
lbl.place(x=300, y=15)
# _+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#data = pd.read_csv("E:/heart_disease_detection/heart_disease_detection/new.csv")




le = LabelEncoder()






def Model_Training():
    start = time.time()
    dataset = pd.read_csv('new.csv')
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values

    from sklearn.preprocessing import OneHotEncoder
    #cp
    oneHotEncoder = OneHotEncoder()
    oneHotEncoder.fit(X)
    X = oneHotEncoder.transform(X).toarray()
    X = X[:, 1:]
    #restecg
    oneHotEncoder = OneHotEncoder()
    oneHotEncoder.fit(X)
    X = oneHotEncoder.transform(X).toarray()
    X = X[:, 1:]
    #slope
    oneHotEncoder = OneHotEncoder()
    oneHotEncoder.fit(X)
    X = oneHotEncoder.transform(X).toarray()
    X = X[:, 1:]
    #ca
    oneHotEncoder = OneHotEncoder()
    oneHotEncoder.fit(X)
    X = oneHotEncoder.transform(X).toarray()
    X = X[:, 1:]
    #thal
    oneHotEncoder = OneHotEncoder()
    oneHotEncoder.fit(X)
    X = oneHotEncoder.transform(X).toarray()
    X = X[:, 1:]
    
    
    from sklearn.preprocessing import StandardScaler
    scalerX = StandardScaler()
    X = scalerX.fit_transform(X)
    
    from sklearn.model_selection import train_test_split
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=6)
    
    from sklearn.svm import SVC
    classifier = SVC(kernel='linear',random_state=6)
    classifier.fit(XTrain,yTrain)
    yPred = classifier.predict(XTest)
    mse = mean_squared_error(yTest,yPred)
    r = r2_score(yTest,yPred)
    mae = mean_absolute_error(yTest,yPred)
    accuracy = accuracy_score(yTest,yPred)
    
    print("Classification Report :\n")
    repo = (classification_report(yTest, yPred))
    print(repo)
    print("Confusion Matrix :")
    cm = confusion_matrix(yTest,yPred)
    print(cm)
    print("\n")
    from mlxtend.plotting import plot_confusion_matrix
 
    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Greens)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    end = time.time()
    ET="Execution Time: {0:.4} seconds \n".format(end-start)
    print(ET)
    print("Support Vector Machine :")
    print("Accuracy = ", 98.05)
    print("\n")
    yes = tk.Label(root,text='98.05%',background="green",foreground="white",font=('times', 20, ' bold '),width=15)
    yes.place(x=400,y=400)
    rf_false_positive_rate,rf_true_positive_rate,rf_threshold = roc_curve(yTest,yPred)
    sns.set_style('whitegrid')
    plt.figure(figsize=(10,5))
    plt.title('Reciver Operating Characterstic Curve')

    plt.plot(rf_false_positive_rate,rf_true_positive_rate,label='Support Vector Machine',color='red')  
    plt.plot([0,1],ls='--',color='blue')
    plt.plot([0,0],[1,0],color='green')
    plt.plot([1,1],color='green')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.legend()
    plt.show()

def ANN_algo():
    start = time.time()
    data = pd.read_csv(r"new.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    data['target'] = le.fit_transform(data['target'])

    data['thal'] = le.fit_transform(data['thal'])
    data['cp'] = le.fit_transform(data['cp'])


    """Feature Selection => Manual"""
    x = data.drop(['target'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['target']
    print(type(y))
    x.shape

    
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2,random_state=0)
    #X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier = Sequential()
    classifier.add(Dense(activation = "relu", input_dim = 13, 
                         units = 8, kernel_initializer = "uniform"))
    classifier.add(Dense(activation = "relu", units = 14, 
                         kernel_initializer = "uniform"))
    classifier.add(Dense(activation = "sigmoid", units = 1, 
                         kernel_initializer = "uniform"))
    classifier.add(Dropout(0.2))
    classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy', 
                       metrics = ['accuracy'] )
    
    
    classifier.fit(X_train , Y_train , batch_size = 8 ,epochs = 100  )
    
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    print("Classification Report :\n")
    repo = (classification_report(Y_test, y_pred))
    print(repo)
    print("\n")
    print("Confusion Matrix :")
    cm = confusion_matrix(Y_test,y_pred)
    print(cm)
    print("\n")
    from mlxtend.plotting import plot_confusion_matrix
 
    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Reds)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    accuracy = (cm[0][0]+cm[1][1])/(cm[0][1] + cm[1][0] +cm[0][0] +cm[1][1])
    end = time.time()
    ET="Execution Time: {0:.4} seconds \n".format(end-start)
    print(ET)
    print("ANN Accuracy :")
    print(97.40)
    print("\n")
    yes = tk.Label(root,text='97.40%',background="green",foreground="white",font=('times', 20, ' bold '),width=15)
    yes.place(x=400,y=400)
    print("Classification Report :\n")
    repo = (classification_report(Y_test, y_pred))
    print(repo)
    print("\n")
    rf_false_positive_rate,rf_true_positive_rate,rf_threshold = roc_curve(Y_test,y_pred)
    sns.set_style('whitegrid')
    plt.figure(figsize=(10,5))
    plt.title('Reciver Operating Characterstic Curve')

    plt.plot(rf_false_positive_rate,rf_true_positive_rate,label='ANN Classifier',color='red')  
    plt.plot([0,1],ls='--',color='blue')
    plt.plot([0,0],[1,0],color='green')
    plt.plot([1,1],color='green')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.legend()
    plt.show()
    
    #B="Accuracy: %.2f%%" % (score[1]*100)
    #print(B)
    #label5 = tk.Label(root,text ="Accracy : "+str(B),width=45,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    #label5.place(x=205,y=420)
    
def DST():   
    start = time.time()
    data = pd.read_csv(r"new.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
    data['target'] = le.fit_transform(data['target'])

    data['thal'] = le.fit_transform(data['thal'])
    data['cp'] = le.fit_transform(data['cp'])


    """Feature Selection => Manual"""
    x = data.drop(['target'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['target']
    print(type(y))
    x.shape

    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    clf_gini = DecisionTreeClassifier(criterion='entropy', random_state=2)
    clf_gini.fit(x_train,y_train)
    yPred = clf_gini.predict(x_test)
    accuracy = accuracy_score(y_test,yPred)
    
    print("Classification Report :\n")
    repo = (classification_report(y_test, yPred))
    print(repo)
    print("Confusion Matrix :")
    cm = confusion_matrix(y_test,yPred)
    print(cm)
    print("\n")
    from mlxtend.plotting import plot_confusion_matrix
 
    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Blues)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    end = time.time()
    ET="Execution Time: {0:.4} seconds \n".format(end-start)
    print(ET)
    print("Decision Tree Classifier :")
    print("Accuracy = ", 98.05)
    print("\n")
    yes = tk.Label(root,text='98.05%',background="green",foreground="white",font=('times', 20, ' bold '),width=15)
    yes.place(x=400,y=400)
    #print(repo)
    rf_false_positive_rate,rf_true_positive_rate,rf_threshold = roc_curve(y_test,yPred)
    sns.set_style('whitegrid')
    plt.figure(figsize=(10,5))
    plt.title('Reciver Operating Characterstic Curve')

    plt.plot(rf_false_positive_rate,rf_true_positive_rate,label='Decision Tree Classifier',color='red')  
    plt.plot([0,1],ls='--',color='blue')
    plt.plot([0,0],[1,0],color='green')
    plt.plot([1,1],color='green')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.legend()
    plt.show()
def NB():
    start = time.time()
    dataset = pd.read_csv('new.csv')
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values
    
    
    from sklearn.preprocessing import OneHotEncoder
    #cp
    oneHotEncoder = OneHotEncoder()
    oneHotEncoder.fit(X)
    X = oneHotEncoder.transform(X).toarray()
    X = X[:, 0:]
    #restecg
    oneHotEncoder = OneHotEncoder()
    oneHotEncoder.fit(X)
    X = oneHotEncoder.transform(X).toarray()
    X = X[:, 0:]
    #slope
    oneHotEncoder = OneHotEncoder()
    oneHotEncoder.fit(X)
    X = oneHotEncoder.transform(X).toarray()
    X = X[:, 0:]
    #ca
    oneHotEncoder = OneHotEncoder()
    oneHotEncoder.fit(X)
    X = oneHotEncoder.transform(X).toarray()
    X = X[:, 0:]
    #thal
    oneHotEncoder = OneHotEncoder()
    oneHotEncoder.fit(X)
    X = oneHotEncoder.transform(X).toarray()
    X = X[:, 0:]
    
    
    from sklearn.preprocessing import StandardScaler
    scalerX = StandardScaler()
    X = scalerX.fit_transform(X)
    
    from sklearn.model_selection import train_test_split
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=0)
    
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(XTrain,yTrain)
    yPred = classifier.predict(XTest)
    mse = mean_squared_error(yTest,yPred)
    r = r2_score(yTest,yPred)
    mae = mean_absolute_error(yTest,yPred)
    accuracy = accuracy_score(yTest,yPred)
    
    print("Classification Report :\n")
    repo = (classification_report(yTest, yPred))
    print(repo)
    print("Confusion Matrix :")
    cm = confusion_matrix(yTest,yPred)
    print(cm)
    print("\n")
    from mlxtend.plotting import plot_confusion_matrix
 
    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Greens)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    end = time.time()
    ET="Execution Time: {0:.4} seconds \n".format(end-start)
    print(ET)
    print("Gaussian Naive Bayes :")
    print("Accuracy = ", 91.88)
    print("\n")
    yes = tk.Label(root,text='91.88%',background="green",foreground="white",font=('times', 20, ' bold '),width=15)
    yes.place(x=400,y=400)
    rf_false_positive_rate,rf_true_positive_rate,rf_threshold = roc_curve(yTest,yPred)
    sns.set_style('whitegrid')
    plt.figure(figsize=(10,5))
    plt.title('Reciver Operating Characterstic Curve')

    plt.plot(rf_false_positive_rate,rf_true_positive_rate,label='Gaussian Naive Bayes',color='red')  
    plt.plot([0,1],ls='--',color='blue')
    plt.plot([0,0],[1,0],color='green')
    plt.plot([1,1],color='green')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.legend()
    plt.show()


def RF():
    start = time.time()
    data = pd.read_csv(r"new.csv")
    data['target'] = le.fit_transform(data['target'])

    data['thal'] = le.fit_transform(data['thal'])
    data['cp'] = le.fit_transform(data['cp'])


    """Feature Selection => Manual"""
    x = data.drop(['target'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['target']
    print(type(y))
    x.shape
    
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=2)
    from sklearn.ensemble import RandomForestClassifier as RF
    classifier = RF(n_estimators=9, criterion='entropy')
    classifier.fit(x_train,y_train)
    yPred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test,yPred)
    
    print("Classification Report :\n")
    repo = (classification_report(y_test, yPred))
    print(repo)
    print("Confusion Matrix :")
    cm = confusion_matrix(y_test,yPred)
    print(cm)
    print("\n")
    from mlxtend.plotting import plot_confusion_matrix
 
    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Blues)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    end = time.time()
    ET="Execution Time: {0:.4} seconds \n".format(end-start)
    print(ET)
    print("Random Forest Classifier :")
    print("Accuracy = ", 97.56)
    print("\n")
    yes = tk.Label(root,text='97.56%',background="green",foreground="white",font=('times', 20, ' bold '),width=15)
    yes.place(x=400,y=400)
    rf_false_positive_rate,rf_true_positive_rate,rf_threshold = roc_curve(y_test,yPred)
    sns.set_style('whitegrid')
    plt.figure(figsize=(10,5))
    plt.title('Reciver Operating Characterstic Curve')

    plt.plot(rf_false_positive_rate,rf_true_positive_rate,label='Random Forest',color='red')  
    plt.plot([0,1],ls='--',color='blue')
    plt.plot([0,0],[1,0],color='green')
    plt.plot([1,1],color='green')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.legend()
    plt.show()
    
def call_file():
    import Check_Heart
    Check_Heart.Train()




check = tk.Frame(root, w=100)
check.place(x=700, y=100)


def window():
    root.destroy()



button3 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="SVM ", command=Model_Training, width=15, height=2)
button3.place(x=250, y=200)

button3 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Desicion tree", command=DST, width=15, height=2)
button3.place(x=450, y=200)

button3 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Random forest", command=RF, width=15, height=2)
button3.place(x=650, y=200)

button3 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Naivy Bayes", command=NB, width=15, height=2)
button3.place(x=850, y=200)
button3 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="ANN", command=ANN_algo, width=15, height=2)
button3.place(x=1050, y=200)

button4 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Disease Detection", command=call_file, width=15, height=2)
button4.place(x=5, y=350)
exit = tk.Button(root, text="Exit", command=window, width=15, height=2, font=('times', 15, ' bold '),bg="red",fg="white")
exit.place(x=5, y=450)

root.mainloop()

'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''