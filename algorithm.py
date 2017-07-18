import numpy as np
import data
import matplotlib.pyplot as plt
import sys , os
def show_progress(i,max):
    msg='\r progress {0}/{1}'.format(i,max)
    sys.stdout.write(msg)
    sys.stdout.flush()

def lwlr(test_point ,  x , y  , k=1.0):
    debug_flag=True
    x=np.mat(x); y=np.mat(y)
    m,n=x.shape
    weights=np.mat(np.eye(m)) #
    for j in range(m):
        diff_mat=test_point - x[j,:] #test point has same # feature with # x[j,m]
        weights[j,j]= np.exp((diff_mat * diff_mat.T) / (-2.0 * (k ** 2)))
    xTx = x.T*(weights*x)
    if np.linalg.det(xTx) == 0.0:
        #print "this matrix is singular , cannot do inverse"
        return
    ws=xTx.I*(x.T*(weights*y.T)) # weights*y.T
    #error=get_error(y[0,0],test_point*ws)
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
        print 'weights*y.T shape is :',np.shape(weights*y.T)
        print 'y',y.shape
        print 'y',y[0,0]
    

    return test_point*ws

def lwlr_test(test_arr , x ,y ,k=1.):

    m = test_arr.shape[0]
    y_hat= np.zeros([m])
    err_count=0
    for i in range(m):
        show_progress(i,m)
        y_hat[i]=lwlr( test_arr[i] , x, y , k)
        #print 'y_hat',y_hat[i]
        if str(y_hat[i])=='nan':
            err_count+=1
            print '\terror count:',err_count

    print 'y_hat shape:',np.shape(y_hat)
    print 'error:',get_error(y,y_hat)
    np.save('./data/abalone.npy',y_hat)
    return y_hat
def get_error(y , y_hat):
    print 'shape label:',np.shape(y)
    print 'shape y_hat',np.shape(y_hat)
    count=0
    sum_=0
    for i in range(len(y_hat)):
        if str(y_hat[i]) == 'nan':
            pass
        else:
#            print y_hat[i] , y[i]
            sum_ += ((y_hat[i]-y[i])**2)
            count+=1
    err=sum_ / float(count)
    print 'sum:',sum_
    print 'count:',count
    print 'error',err
    return err
def plot_(x , y_hat, y):
    fig= plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(x , y)
    ax.scatter(x,y_hat,s=2,c='red')
    plt.show()

def ridge_reg(x,y,lam=0.2):
    xTx = x.T*x
    denom = xTx+np.eye(np.shape(x)[1])*lam
    if np.linalg.det(denom)==0:
        print "this matrix is singular , c" \
              "annot do inverse"
        return
    ws = denom.I*(x.T*y)
    return ws

def ridge_test(x,y ,num_test=30):
    x = np.mat(x) ; y = np.mat(y).T
    y_mean=np.mean(y,0)
    y = y-y_mean
    x_mean = np.mean(x,0)
    x_var = np.var(x,0)
    x= x-x_mean/x_var
    w = np.zeros((num_test, np.shape(x)[1]))
    for i in range(num_test):
        ws=ridge_reg(x,y,np.exp(i-10))
        w[i,:]=ws.T
    print np.shape(w)
    return w



def rigde_plot(ridge_weights):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridge_weights)
    plt.show()

if __debug__ == True:
    """abalone data : lwlr"""

    input_data , label_data=data.load_data('./data/abalone.txt')
    lwlr(test_point=input_data[0] , x=input_data , y=label_data ,k=1)
    y_hat=lwlr_test(test_arr=input_data , x=input_data , y=label_data , k=0.01)
    y_hat=np.load('./data/abalone.npy')
    print '#####',np.shape(y_hat)
    print get_error(label_data,y_hat)

    """
    srt_ind=input_data[:,1].argsort(axis=0)
    srt_input_data=input_data[srt_ind]
    srt_label_data=label_data[srt_ind]
    srt_y_hat=y_hat[srt_ind]
    plot_(srt_input_data[:,1] , y_hat[srt_ind] , srt_label_data)
    get_error(srt_label_data,srt_y_hat )
    """

    """abalone data : ridge regression"""
    """
    input_data, label_data = data.load_data('./data/abalone.txt')
    ridge_weights=ridge_test(input_data,label_data )
    rigde_plot(ridge_weights)
    """


    #print 'label data[0]', label_data[0]


