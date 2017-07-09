import numpy as np
import data
import matplotlib.pyplot as plt
import sys , os
def show_progress(i,max):
    msg='\r progress {0}/{1}'.format(i,max)
    sys.stdout.write(msg)
    sys.stdout.flush()

def lwlr(test_point ,  x , y  , k=1.0):
    debug_flag=False
    x=np.mat(x); y=np.mat(y)
    m,n=x.shape
    weights=np.mat(np.eye(m)) #
    for j in range(m):
        diff_mat=test_point - x[j,:]
        weights[j,j]= np.exp((diff_mat * diff_mat.T) / (-2.0 * (k ** 2)))
    xTx = x.T*(weights*x)
    if np.linalg.det(xTx) == 0.0:
        print "this matrix is singular , cannot do inverse"
        return
    ws=xTx.I*(x.T*(weights*y.T)) # weights*y.T
    error=get_error(y[0,0],test_point*ws)
    if __debug__==debug_flag:
        print '###debug####'
        print 'x shape is :',np.shape(x)
        print 'y shape is :',np.shape(y)
        print 'xTx shape is:',np.shape(xTx)
        print 'xTx.I shape is',np.shape(xTx.I)
        print 'test_points shape is',np.shape(test_point)
        print 'weight shape is ',np.shape(weights)
        #print 'weight', weights
        print 'weights*y.T shape is ',np.shape(weights*y.T)
        print 'weights*ys sample',(weights*y.T)[:10].T
        print 'x.T*(weights*y.T) shape',np.shape((x.T*(weights*y.T)))
        print 'xTx.I*(x.T*(weights*y.T)) shape',np.shape( xTx.I*(x.T*(weights*y.T)))
        print 'test_point*ws shape is ',np.shape(test_point*ws)
        print 'y',y.shape
        print 'y',y[0,0]
        print error

    return test_point*ws

def lwlr_test(test_arr , x ,y ,k=1.):

    m = test_arr.shape[0]
    y_hat= np.zeros([m])
    for i in range(m):
        show_progress(i,m)
        y_hat[i]=lwlr( test_arr[i] , x, y , k)
    print 'y_hat shape:',np.shape(y_hat)
    print 'error:',get_error(y,y_hat)
    np.save('./data/abalone.npy',y_hat)
    return y_hat
def get_error(y , y_hat):
    err = ((y_hat-y)**2).sum()
    return err
def plot_(x , y_hat, y):
    fig= plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(x , y)
    ax.scatter(x,y_hat,s=2,c='red')
    plt.show()

if __debug__ == True:
    input_data , label_data=data.load_data('./data/abalone.txt')
    print lwlr(test_point=input_data[0] , x=input_data , y=label_data ,k=1)
    y_hat=lwlr_test(test_arr=input_data , x=input_data , y=label_data , k=0.01)

    """
    srt_ind=input_data[:,1].argsort(axis=0)
    srt_input_data=input_data[srt_ind]
    srt_label_data=label_data[srt_ind]
    srt_y_hat=y_hat[srt_ind]
    plot_(srt_input_data[:,1] , y_hat[srt_ind] , srt_label_data)
    get_error(srt_label_data,srt_y_hat )
    """


    #print 'label data[0]', label_data[0]


