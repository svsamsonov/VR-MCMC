import numpy as np
import numpy.polynomial as P

def approx_q(X_train,Y_train,N_traj_train,lag,max_deg):
    """
    Function to regress q functions on a polynomial basis;
    Args:
        X_train - train tralectory;
        Y_train - function values;
        N_traj_train - number of teraining trajectories;
        lag - truncation point for coefficients, those for |p-l| > lag are set to 0;
        max_deg - maximum degree of polynomial in regression
    """
    coefs_poly = np.zeros((lag,max_deg+1),dtype = float)
    for i in range(lag):
        x_all = np.array([])
        y_all = np.array([])
        for j in range(N_traj_train):
            y = Y_train[j,i:,0]
            if i == 0:
                x = X_train[j,:]
            else:
                x = X_train[j,:-i]
            x_all = np.concatenate([x_all,x])
            y_all = np.concatenate([y_all,y])
    res = P.polynomial.polyfit(x,y,max_deg)
    coefs_poly[i,:] = res
    return coefs_poly