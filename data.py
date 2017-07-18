import numpy as np
import matplotlib.pyplot as plt
def load_data(data_path):
    f= open(data_path)
    n_feat=len(f.readline().split('\t')) -1
    input_data=[];label_data=[]
    for line in f.readlines():
        elements=line.split('\t')
        input_data.append(elements[:-1])
        label_data.append(elements[-1])
    input_data=np.asarray(input_data).astype(float)
    label_data=np.asarray(label_data).astype(float)
    if __debug__==True:
        print 'input shape', np.shape(input_data)
        print 'the number of featrue:' , n_feat
        print 'The number of data',len(input_data)
        print 'sample of input data',input_data[0:2]

        print 'labels shape',np.shape(label_data)
    return input_data, label_data

def reg(input , label):
    x=np.mat(input)
    y=np.mat(label)
    x=x.astype(float)
    y=y.astype(float)
    xTx= np.matmul(x.T  ,x)


    if np.linalg.det(xTx)==0:
        print "this matrix cannot Inverve"
    ws=np.linalg.solve( xTx , np.matmul(x.T ,y.T))
    if __debug__==True:
        print 'x.T shape is : ',np.shape(x.T)
        print 'x shape',np.shape(x)
        print 'x sample:',x[0]
        print 'xTx shape:',np.shape(xTx)
        print 'y shape is :',np.shape(y)
        print 'ws shape is :',np.shape(ws)
    return ws

def plot(x, y):
    x=np.mat(x)
    y=np.mat(y)
    fig=plt.figure()
    ax = fig.add_subplot(111)
    x_= x.copy()
    y_ = x_* ws
    ax.plot(x_[:,1],y_)

    print 'x[:,1].flatten().A[0]) shape ',np.shape(x[:,1].flatten().A[0])
    print 'y[:].flatten().A[0]) shape ',np.shape(y[:].flatten().A[0])
    ax.scatter(x[:,1].flatten().A[0] , y[:].flatten().A[0])
    plt.show()
if __name__ == '__main__':
    #data_path='./data/horseColicTraining2.txt'
    data_path='./data/ex0.txt'
    data_path='./data/abalone.txt'
    input_data , label_data=load_data(data_path)
    ws=reg(input_data , label_data)
    plot(input_data,label_data)
    a=np.matmul(input_data , ws)
    b=np.mat(label_data)
    coef = np.corrcoef(a.T,b)
    print coef





