import numpy as np
import data
def rssError(y, yHat):
    return ((y - yHat) ** 2).sum()
def regularize(x):
    input = x.copy()
    input_mean=np.mean(input,0)
    input_var =np.var(input, 0)
    input = (input - input_mean)/input_var
    return input

def stagewise(x,y,eps=0.01,n_iter=100):
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
        print ws.T
        lowest_error= np.inf
        for j in range(h):
            print j
            for sign in [-1,1]:
                ws_test=ws.copy()
                ws_test += eps*sign
                y_test = x * ws_test
                rssE = rssError( np.asarray(y),np.asarray(y_test))
                if rssE < lowest_error:
                    lowest_error = rssE
                    ws_max=ws_test
        ws=ws_max.copy()
        return_mat[i,:] = ws.T
    return return_mat



if __name__=='__main__':
    input_data , label_data=data.load_data('./data/abalone.txt')
    stagewise(input_data , label_data )



