import numpy as np
import string

locate_x = float(input("Enter the x point: "))
print("the locate x value is: %f",locate_x)
BW_parameter =float(input("Enter the bandwidth parameter:"))
print("the bandwidth parameter is：%f",BW_parameter)

X = np.loadtxt('./data/x.dat' , dtype = float)
y = np.loadtxt('./data/y.dat' , dtype = float)

locate_X = np.ones(X.shape)
w = np.exp( - np.sum((np.square(X - locate_X)),axis=1)/(2*BW_parameter*BW_parameter) )

