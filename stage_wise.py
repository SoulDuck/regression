
import numpy as np
import data
import os,sys
def rssError(y, yHat):
    return ((y - yHat )**2).sum()
def show_progress(i,max):
    msg='\r progress {0}/{1}'.format(i,max)
    sys.stdout.write(msg)
    sys.stdout.flush()
def regularize(x):
    input = x.copy()
    input_mean=np.mean(input,0)
    input_var =np.var(input, 0)
    input = (input - input_mean)/input_var


    if __debug__ ==True:
        print '########################### rebularize debug ###############################'
        print 'input_mean shape',np.shape(input_mean)
        print 'input_mean',input_mean
        print 'input_mean var', np.shape(input_var)
        print 'input_var',input_var
        print 'input',input
        print '################################################################'

    return input

def stagewise(x,y,eps=0.01,n_iter=100):

    debug_flag=True
    x=np.mat(x);y=np.mat(y)
    y_mean= np.mean(y,1)
    y_var = np.mean(y,1)
    y=(y-y_mean)/y_var
    x=regularize(x)
    if __debug__ == debug_flag:

        print '########################### step wise debug ###############################'
        print 'y_mean',y_mean
        print 'y_var',y_var
        print 'y',y
        print '###########################################################################'
    h,w=np.shape(x)
    ws = np.zeros([w,1]);
    ws_test=ws.copy()
    ws_max=ws.copy()
    return_mat=np.zeros([n_iter , w])
    for i in range(n_iter):

        print '\ti:',i,' ',ws.T
        lowest_error= np.inf
        for j in range(w):
            #show_progress(j,w)
            for sign in [-1,1]:
                ws_test=ws.copy()
                ws_test[j] += eps*sign
                y_test = x * ws_test
                rssE = rssError( np.asarray(y),np.asarray(y_test))

                print 'j:',j,'\t',ws_test.T,'\t','rssE:',rssE
                if rssE ==0:
                    print y_test.T[:100]
                    print y[:100]
                if rssE < lowest_error:
                    lowest_error = rssE
                    ws_max=ws_test
                if __debug__ ==debug_flag:
                    """ """
                    #print 'lowest_error:',rssE
                    #print 'ws:',ws_max

                    #print 'ws_test',ws_test.T
                    #print 'y_test shape',np.shape(y_test)
        print '-----best ws-----'
        ws=ws_max.copy()
        return_mat[i,:] = ws.T
        #print 'ws',ws.T
        #print 'return_mat',return_mat
        print '-----------------'
    return return_mat



if __name__=='__main__':
    input_data , label_data=data.load_data('./data/abalone.txt')
    print label_data[:100]
    best_ws=stagewise(input_data , label_data )
    print 'best ws is :',best_ws




