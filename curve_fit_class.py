import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import display
from IPython.display import Math
import time
from scipy.optimize import curve_fit
from scipy.optimize import LinearConstraint
from scipy.misc import factorial
import lmfit




class Optimizer(object):
    ## fit setting functions
    # multi parameter fitting 
    # https://stackoverflow.com/questions/34136737/using-scipy-curve-fit-for-a-variable-number-of-parameters




    def coherentPn(self,n,nbar):
        return nbar**n*np.exp(-nbar)/factorial(n)

    def thermalPn(n,nbar):
        return (nbar/(1.0+nbar))**n/(nbar+1)

    def FockDist(self, t, t0, Omega0, gamar0, Pn, N):
        n = np.arange(N).reshape(N,1)
        omega = Omega0 * np.sqrt(n+1)
        gamar_n = gamar0 * (n+1)**0.7
        Pn = Pn.reshape(N,1)
        #yn = Pn*np.sin(omega*(t-t0))**2*np.exp(-gamar_n*(t-t0))
        yn = Pn*np.cos(2*omega*(t-t0))*np.exp(-gamar_n*(t-t0))
        return np.sum(yn,axis=0)

    def CoherentDist(t, t0, Omega0, gamar0, nbar, N):
        n = np.arange(N).reshape(N,1)
        omega = Omega0 * np.sqrt(n+1)
        gamar_n = gamar0 * (n+1)**0.7
        Pn = coherentPn(n,nbar)
        yn = Pn*np.sin(omega*(t-t0))**2*np.exp(-gamar_n*(t-t0))
        return np.sum(yn,axis=0)

    # Pn distribution
    def wrapper_fit_func(self, t, N, *args):
        t0     = args[0][0]
        Omega0 = args[0][1]
        gamar0 = args[0][2]
        Pn = np.array(args[0][3:N+3])
        return self.FockDist(t, t0, Omega0, gamar0, Pn, N)

    #coherent distribution
    def wrapper_fit_coherent(t, N, *args):
        t0     = args[0][0]
        Omega0 = args[0][1]
        gamar0 = args[0][2]
        nbar = args[0][3]
        return CoherentDist(t, t0, Omega0, gamar0, nbar, N)



    def curve_fitter(self, dim, curve_set, curve_size, t_scale):
        t=np.linspace(0,t_scale,curve_size)
        y=curve_set
        N=dim
        nbar = 1.5
        #params_0 = [0.0, 2*np.pi/50, 0.0001]  # t0, omega0, gamar0
        params_0 += [self.coherentPn(i, nbar) for i in range(N)]  # using coherentstate distribution for initialization 
        plt.plot(t, y, 'o')
        #plt.show()
        popt, pcov = curve_fit(lambda t, *p_0: self.wrapper_fit_func(t, N, p_0), t, y, p0=params_0)  #*make use of parameter's dimension instead of num of parameters
        
        
        #plt.plot(t, wrapper_fit_func(t,N,params_0))
        plt.plot(t, self.wrapper_fit_func(t,N,popt),'r')
        plt.show()
        print(popt[3:])
        return(popt[3:])





class Optimizer_fix_omega(object):
    ## fit setting functions
    # multi parameter fitting 
    # https://stackoverflow.com/questions/34136737/using-scipy-curve-fit-for-a-variable-number-of-parameters

    def coherentPn(self,n,nbar):
        return nbar**n*np.exp(-nbar)/factorial(n)

    def thermalPn(n,nbar):
        return (nbar/(1.0+nbar))**n/(nbar+1)
    '''
    #fit gamma
    def FockDist(self, t, t0, gamar0, Pn, N, Omega0):
        n = np.arange(N).reshape(N,1)
        omega = Omega0 * np.sqrt(n+1)
        gamar_n = gamar0 * (n+1)**0.7
        Pn = Pn.reshape(N,1)
        Pn=Pn/np.sum(Pn)
        #yn = Pn*np.sin(omega*(t-t0))**2*np.exp(-gamar_n*(t-t0))
        yn = Pn*np.cos(omega*(t-t0))*np.exp(-gamar_n*(t-t0))
        return 0.5*(1-np.sum(yn,axis=0))

    # Pn distribution
    def wrapper_fit_func(self, t, N, Omega0, *args):
        t0     = args[0][0]
        gamar0 = args[0][1]
        Pn = np.array(args[0][2:N+2])
        return self.FockDist(t, t0, gamar0, Pn, N, Omega0)


    def curve_fitter(self, om0, dim, curve_set, exp_time):
        t=exp_time
        y=curve_set
        N=dim
        Omega0=om0
        nbar = 1.5
        params_0 = [0.0, 0.00001]  # t0, gamar0
        params_0 += [self.coherentPn(i, nbar) for i in range(N)]  # using coherentstate distribution for initialization 
        #origin curve points
        plt.plot(t, y, 'o')
        #plt.show()
        popt, pcov = curve_fit(lambda t, *p_0: self.wrapper_fit_func(t, N, Omega0, p_0), t, y, p0=params_0)  #*make use of parameter's dimension instead of num of parameters
        
        
        # plt.plot(t, wrapper_fit_func(t,N,params_0))
        #fitted curve
        plt.plot(t, self.wrapper_fit_func(t,N,Omega0,popt),'r')
        plt.show()
        print(popt[1],popt[2:])
        return(popt[2:])
    '''
    
    '''
    #give gamma
    def FockDist(self, t, t0, gamar0, Pn, N, Omega0):
        n = np.arange(N).reshape(N,1)
        omega = Omega0 * np.sqrt(n+1)
        gamar_n = gamar0 * (n+1)**0.7
        Pn = Pn.reshape(N,1)
        Pn=Pn/np.sum(Pn)
        #yn = Pn*np.sin(omega*(t-t0))**2*np.exp(-gamar_n*(t-t0))
        yn = Pn*np.cos(omega*(t-t0))*np.exp(-gamar_n*(t-t0))
        return 0.5*(1-np.sum(yn,axis=0))

    # Pn distribution
    def wrapper_fit_func(self, t, N, Omega0, gamar0, *args):
        t0     = args[0][0]
        
        Pn = np.array(args[0][1:N+1])
        return self.FockDist(t, t0, gamar0, Pn, N, Omega0)


    def curve_fitter(self, om0, gamma0, dim, curve_set, exp_time):
        t=exp_time
        y=curve_set
        N=dim
        Omega0=om0
        gamar0=gamma0
        nbar = 1.5
        params_0 = [0.0]  # t0
        params_0 += [self.coherentPn(i, nbar) for i in range(N)]  # using coherentstate distribution for initialization 
        #origin curve points
        plt.plot(t, y, 'o')
        #plt.show()
        popt, pcov = curve_fit(lambda t, *p_0: self.wrapper_fit_func(t, N, Omega0, gamar0, p_0), t, y, p0=params_0)  #*make use of parameter's dimension instead of num of parameters
        
        
        # plt.plot(t, wrapper_fit_func(t,N,params_0))
        #fitted curve
        plt.plot(t, self.wrapper_fit_func(t,N,Omega0,gamar0,popt),'r')
        plt.show()
        print(popt[0],popt[1:])  #P series
        return(popt[1:])

    '''
    '''
    #no t0, fit gamma
    def FockDist(self, t, gamar0, Pn, N, Omega0):
        n = np.arange(N).reshape(N,1)
        omega = Omega0 * np.sqrt(n+1)
        gamar_n = gamar0 * (n+1)**0.7
        Pn = Pn.reshape(N,1)
        Pn=Pn/np.sum(Pn)
        #yn = Pn*np.sin(omega*(t-t0))**2*np.exp(-gamar_n*(t-t0))
        yn = Pn*np.cos(omega*(t))*np.exp(-gamar_n*(t))
        return 0.5*(1-np.sum(yn,axis=0))

    # Pn distribution
    def wrapper_fit_func(self, t, N, Omega0, *args):
        #t0     = args[0][0]
        gamar0 = args[0][0]
        Pn = np.array(args[0][1:N+1])
        return self.FockDist(t, gamar0, Pn, N, Omega0)


    def curve_fitter(self, om0, dim, curve_set, exp_time):
        t=exp_time
        y=curve_set
        N=dim
        Omega0=om0
        nbar = 1.5
        params_0 = [0.00001]  # t0, gamar0
        params_0 += [self.coherentPn(i, nbar) for i in range(N)]  # using coherentstate distribution for initialization 
        #origin curve points
        plt.plot(t, y, 'o')
        #plt.show()
        popt, pcov = curve_fit(lambda t, *p_0: self.wrapper_fit_func(t, N, Omega0, p_0), t, y, p0=params_0, bounds=(0,1))  #*make use of parameter's dimension instead of num of parameters
        
        
        # plt.plot(t, wrapper_fit_func(t,N,params_0))
        #fitted curve
        plt.plot(t, self.wrapper_fit_func(t,N,Omega0,popt),'r')
        plt.show()
        print(popt[0],popt[1:])
        return(popt[1:])

    '''

    #no t0, given gamma
    def FockDist(self, t, gamar0, Pn, N, Omega0):
        n = np.arange(N).reshape(N,1)
        omega = Omega0 * np.sqrt(n+1)
        gamar_n = gamar0 * (n+1)**0.7
        Pn = Pn.reshape(N,1)
        Pn=Pn/np.sum(Pn)  #not always right, if gammr is not proper, sum may not be 1
        #yn = Pn*np.sin(omega*(t-t0))**2*np.exp(-gamar_n*(t-t0))
        yn = Pn*np.cos(omega*(t))*np.exp(-gamar_n*(t))
        P_1 = 0.5*(1-np.sum(yn,axis=0))
        #Pn=abs(Pn) #no use
        #Pn=Pn/np.sum(Pn)  #wrong, sum! =1
        return P_1

    # Pn distribution
    def wrapper_fit_func(self, t, N, Omega0, gamar0, *args):
        
        Pn = np.array(args[0][0:N])
        return self.FockDist(t, gamar0, Pn, N, Omega0)


    def curve_fitter(self, om0, gamma0, dim, curve_set, exp_time):
        t=exp_time
        y=curve_set
        N=dim
        Omega0=om0
        gamar0=gamma0
        nbar = 1.5
        params_0=[]
        params_0 += [self.coherentPn(i, nbar) for i in range(N)]  # using coherentstate distribution for initialization 
        #origin curve points
        plt.plot(t, y, 'o')
        #plt.show()
        #linear_constraint = LinearConstraint(N*[1], [1], [1])
        #curve_fit(f, xdata, ydata)
        popt, pcov = curve_fit(lambda t, *p_0: self.wrapper_fit_func(t, N, Omega0, gamar0, p_0), t, y, p0=params_0, bounds=(0,1) )  #*make use of parameter's dimension instead of num of parameters
        
        print(popt[0:])
        # plt.plot(t, wrapper_fit_func(t,N,params_0))
        #fitted curve
        plt.plot(t, self.wrapper_fit_func(t,N,Omega0,gamar0,popt),'r')
        plt.savefig("examples.jpg")
        plt.show()
        #print(popt[0:])  #P series
        return(popt[0:])
    


## lmfit package with direct contraints on fitting  sum(pi)=1
class Optimizer_lmfit(object):
    def __init__(self, dim):
        self.dim=dim
    def coherentPn(self,n,nbar):
        return nbar**n*np.exp(-nbar)/factorial(n)

    def curve_fitter(self, om0, gamma0, dim, curve_set, exp_time):
        t=exp_time
        y=curve_set
        N=dim
        Omega0=om0
        gamar0=gamma0
        nbar = 1.5
        def model_func(n_params, x, **kwargs):
            # an example function that takes a variable number of parameters
            n = np.arange(N).reshape(N,1)
            omega = om0 * np.sqrt(n+1)
            gamar_n = gamma0 * (n+1)**0.7
            yn_set=[]
            for i in range(n_params):
                omega = om0 * np.sqrt(i+1)
                gamar_n = gamma0 * (i+1)**0.7
                pi = kwargs['p%d'%i]
                yn = pi*np.cos(omega*(x))*np.exp(-gamar_n*(x))
                yn_set.append(yn)
            P_1 = 0.5*(1-np.sum(yn_set,axis=0))
            return P_1

        def fn(*args, **kwds):
            return model_func(self.dim, *args, **kwds)  

        # specify args/kwargs (lmfit uses these attributes internally)
        fn.argnames = ['x']
        fn.kwargs = [('p%d'%i,0) for i in range(N)]  #unknown num of pi parameters, so pack it into one list

        # create the model
        model = lmfit.Model(fn, independent_vars=['x'])
        params = lmfit.Parameters()
        str=''
        nbar=1.5
        for i in range(N-1):
            params.add('p'+'%s'%i, value=float(self.coherentPn(i, nbar)), vary=True, min=0, max=1) #float has no arcsin
            #params.add('p'+'%s'%i, value=b, vary=True, min=0, max=1) #float has no arcsin
            str+='-'+'p'+'%s'%i
        params.add('p'+'%s'%(N-1), expr='1'+str)  #pn=1-p1-p2-...pn-1, as constraints

        result = model.fit(y, x=t, params=params)
        plt.plot(t,y,'o','b')
        #plt.plot(t, result.init_fit, 'k--')
        plt.plot(t, result.best_fit, 'r-')
        #plt.show()

        p_re=[]
        for key in result.params:
            print(key, "=", result.params[key].value, "+/-", result.params[key].stderr)
            p_re.append(result.params[key].value)
        return p_re