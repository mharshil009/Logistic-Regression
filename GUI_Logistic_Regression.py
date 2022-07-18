import pandas as pd
import tkinter
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.linear_model import LogisticRegression
from tkinter import *

dataset = pd.read_csv('C:\\Users\\datasets\\Buy_Book1.csv')

X = dataset.iloc[0,0].values 
Y = dataset.iloc[0:, -1].values


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=0)



X_train1 = np.reshape(X_train, (-1,1))
Y_train1 = np.reshape(Y_train, (-1,1))

X_test1 = np.reshape(X_test, (-1,1))
Y_test1 = np.reshape(Y_test, (-1,1))

regressor = LogisticRegression()
regressor.fit(X_train1,Y_train1) 


def model_pred():

    age1-entry.get()
    age = int(age1)

    tran_age = np.array([[age]])
    buy_book = regressor.predict(tran_age)
    buy_book = str(buy_book)

    label1 = Label (window, text=buy_book, fg='red', font=("Courier",25))
    label1.pack()

    entry.delete(0, END)



window = Tk()
window.geometry("600x700")
window.title("Template Window")
label = Label (window, text="Enter the Age of the Person", fg='red', font=("Courier", 15))
label.pack()
area = StringVar()
area.set("")

entry = Entry(window, textvariable=area, fg='green', width=10, font=("Courier",15))
entry.pack()

pred_button = Button(window, text="Predict", fg='red', command=model_pred, height=2, width=15)
pred_button.pack()

mainloop()

