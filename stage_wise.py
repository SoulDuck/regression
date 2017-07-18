
import numpy as np
import data
import os,sys
def rssError(y, yHat):
    return ((y - yHat) ** 2).sum()
def show_progress(i,max):
    msg='\r progress {0}/{1}'.format(i,max)
    sys.stdout.write(msg)
    sys.stdout.flush()
def regularize(x):
    input = x.copy()
    input_mean=np.mean(input,0)
    input_var =np.var(input, 0)
    input = (input - input_mean)/input_var
    return input

def stagewise(x,y,eps=0.01,n_iter=100):
    debug_flag=True
    x=np.mat(x);y=np.mat(y)
    y_mean= np.mean(y,0)
    y_var = np.mean(y,0)
    y=(y-y_mean)/y_var

    h,w=np.shape(x)
    ws = np.zeros([w,1]);
    ws_test=ws.copy()
    ws_max=ws.copy()
    return_mat=np.zeros([n_iter , w])
    for i in range(n_iter):

        print '\ti:',i,' ',ws.T
        lowest_error= np.inf
        for j in range(w):
            show_progress(j,w)
            for sign in [-1,1]:
                ws_test=ws.copy()
                ws_test[j] += eps*sign
                y_test = x * ws_test
                rssE = rssError( np.asarray(y),np.asarray(y_test))
                if rssE < lowest_error:
                    lowest_error = rssE
                    ws_max=ws_test
                if __debug__ ==debug_flag:
                    """ """
                    #print 'lowest_error:',rssE
                    #print 'ws:',ws_max

                    #print 'ws_test',ws_test.T
                    #print 'y_test shape',np.shape(y_test)
        ws=ws_max.copy()
        return_mat[i,:] = ws.T
    return return_mat



if __name__=='__main__':
    input_data , label_data=data.load_data('./data/abalone.txt')
    best_ws=stagewise(input_data , label_data )
    print 'best ws is :',best_ws




