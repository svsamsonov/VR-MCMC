import numpy as np
import numpy.polynomial as P
from sklearn.preprocessing import PolynomialFeatures
from samplers import ULA
from potentials import GaussPotential,GaussMixture,GausMixtureIdent,GausMixtureSame
import copy

def H(k, x):
    if k==0:
        return 1.0
    if k ==1:
        return x
    if k==2:
        return (x**2 - 1)/np.sqrt(2)
    h = hermitenorm(k)(x) /  np.sqrt(math.factorial(k))
    return h

def split_index(k,d,max_deg):
    """
    transforms single index k into d-dimensional multi-index d with max_deg degree each coordinate at most;
    """
    k_vec = np.zeros(d,dtype = int)
    for i in range(d):
        k_vec[-(i+1)] = k % (max_deg + 1)
        k = k // (max_deg + 1)
    return k_vec

def Hermite_val(k_vec,x_vec):
    P = 1.0
    d = x_vec.shape[0]
    for i in range(d):
        P = P * H(k_vec[i],x_vec[i])
    return P

def Eval_Hermite(k,x_vec,max_deg):
    """
    Evaluates Hermite polynomials at component x_vec by multi-index obtained from single integer k;
    Args:
        max_deg - integer, maximal degree of a polynomial at fixed dimension component;
        k - integer, number of given basis function; 1 <= k <= (max_deg+1)**d
        x_vec - np.array of shape(d,N), where d - dimension, N - Train (or Test) sample size
    """
    k_vec = split_index(k,len(x_vec),max_deg)
    #now we initialised k_vec
    return Hermite_val(k_vec,x_vec)

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
    dim = X_train[0,:].shape[0]
    print("dimension = ",dim)
    coefs_poly = np.array([])
    for i in range(lag):
        x_all = np.array([])
        y_all = np.array([])
        for j in range(N_traj_train):
            y = Y_train[j,i:,0]
            if i == 0:
                x = X_train[j,:]
            else:
                x = X_train[j,:-i]
            #concatenate results
            if x_all.size == 0:
                x_all = x
            else:
                x_all = np.concatenate((x_all,x),axis = 0)
            y_all = np.concatenate([y_all,y])
        #should use polyfeatures here
        poly = PolynomialFeatures(max_deg)
        X_features = poly.fit_transform(x_all)
        lstsq_results = np.linalg.lstsq(X_features,y_all,rcond = None)
        coefs = copy.deepcopy(lstsq_results[0])
        coefs.resize((1,X_features.shape[1]))           
        if coefs_poly.size == 0:
            coefs_poly = copy.deepcopy(coefs)
        else:
            coefs_poly = np.concatenate((coefs_poly,coefs),axis=0)
    return coefs_poly

def test_traj(r_seed,max_deg):
    """
    """
    X_test,Noise = ULA(r_seed,Potential,step, N, n, d, return_noise = True)
    #compute number of basis polynomials
    num_basis_funcs = (max_deg+1)**d - 1
    print("number of basis functions = ",num_basis_funcs)
    #compute polynomials of noise variables Z_l
    poly_vals = np.zeros((num_basis_funcs,N_test),dtype = float)
    for k in range(num_basis_funcs):
        poly_vals[k,:] = Eval_hermite(k,Noise,max_deg)
    #initialize function
    f_vals_vanilla = set_func(X_test)
    cvfs = np.zeros_like(f_vals_vanilla)
    for i in range(1,len(cvfs)):
        #start computing a_{p-l} coefficients
        num_lags = min(lag,i)
        a_vals = np.zeros((num_lags,num_basis_funcs),dtype = float)#cobtrol values
        
        for nfunc in range(num_lags):#for a fixed lag Q function
            #compute \hat{a} with fixed lag
            x = X_test[i-1-nfunc]#should be -2 here?
            a_cur = np.zeros(...,dtype=float)
            for m in range(len(coefs_poly[0])):
                poly_vspom = np.zeros((max_deg_regr+1)*d,dtype=float)
                poly_indices = split_index(m,d,max_deg_regr)
                print(poly_indices)
                for u in range(m+1):
                    deg = u % d
                    poly_vspom[u] = ((x-gamma*Cur_pot.gradpotential(x))**(m-deg))*((np.sqrt(2*gamma))**deg)*sp.special.binom(m,deg)
                #print(poly_vspom)
                a_cur = P.polynomial.polyadd(a_cur,coefs_poly[npol,m]*poly_vspom)
            #print("a_cur")
            #print(a_cur)
            for k in range(1,max_deg+1):
                c = np.zeros(max_deg+1)
                c[k] = 1
                herm_coef = P.hermite_e.herme2poly(c)
                #normalize now
                herm_coef = herm_coef / np.sqrt(sp.special.factorial(k))
                integr_coefs = P.polynomial.polymul(herm_coef,a_cur)
                #Note that a_vals are stored in reversed order
                a_vals[-(npol+1),k] = np.dot(integr_coefs,moments_stand_norm[:len(integr_coefs)])
            #OK, now I have coefficients of the polynomial, and I need to integrate it w.r.t. Gaussian measure
        #print(a_vals.shape)
        #print(a_vals)
        cvfs[i] = np.sum(a_vals*(poly_vals[:,i-num_poly+1:i+1].T))
        #cvfs[i] = np.sum(np.mean(a_vals,axis = 0))
        if (i%100 == 0):
            print("100 observations proceeded")
        #save results
        test_stat_vanilla[ind,i] = np.mean(f_vals_vanilla[1:(i+1)])
        test_stat_vr[ind,i] = test_stat_vanilla[ind,i] - np.sum(cvfs[:i])/i