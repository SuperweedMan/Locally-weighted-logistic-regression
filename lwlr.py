import numpy as np
import time
import string
def lwlr( point_x , train_X, train_y ,BW ,mode = 0):
    #获取参数
    if (mode == 0):
        lambd = 1e-4 #lambda参数
        locate_x_str = input("Enter the x point: \r\neg. [1,2]")
        locate_x = np.array(eval(locate_x_str))
        print("the locate x value is: %f",locate_x)
        BW_parameter =float(input("Enter the bandwidth parameter:"))
        print("the bandwidth parameter is：%f",BW_parameter)
        #从本地读取样本值
        X = np.loadtxt('./data/x.dat' , dtype = float)
        y = np.loadtxt('./data/y.dat' , dtype = float)
    else :
        locate_x = point_x
        BW_parameter = BW
        X = train_X
        y = train_y
    #x与theta矩阵化
    locate_X = np.ones(X.shape)
    locate_X = locate_x * locate_X
    theta = np.zeros((X.shape[1],1))
    #计算加权系数 w矩阵
    w = np.exp( - np.sum((np.square(X - locate_X)),axis=1)/(2*BW_parameter*BW_parameter) )

    h = 1+np.exp(np.matmul(X,-theta))
    J = np.ones((X.shape[1],1))
    while (np.linalg.norm(J) > 1e-6 ):
        J = np.matmul(X.T,(w * (y - h) )) - lambd*theta
        H = np.matmul(X.T,(-np.diag(w*h*(1-h))))  - lambd*np.eye(X.shape[1],dtype=float)
        theta = theta - np.matmul(np.linalg.inv(H),J)

