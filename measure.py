from sympy import *
from qutip import *
import numpy as np
import qinfer as qi
import matplotlib.pyplot as plt
import math
import random

class Measure(object):
    def __init__(self, dim, num_dis):
        self.dim = dim
        self.bas = []
        self.alpha0=3
        self.alpha=[]
        self.dis=[]
        self.dis_bas=[]  #displaced basis, [num_basis][dim]
        self.P=[]
        self.dm=[]
        self.num_dis=num_dis
    def generate_basis(self):
        for i in range (self.dim):    #generate measurement basis
            self.bas.append(basis(self.dim,i))
    def example_matrix(self):
        self.generate_basis()
        '''
        state=0.2*self.bas[0]+0.4*self.bas[1]+0.6*self.bas[2]
        '''
        state=self.bas[0]
        for i in range (1,self.dim):
            state+=np.random.rand()*self.bas[i]
        
        state_norm=state/state.norm()  #arbiturary pure state
        dm=ket2dm(state_norm)
        return dm
    #self.dm=example_matrix()
#m=Measure(dim=4).example_matrix()
#print(m)
    def measure(self, dis_alpha):
        for i in range(self.num_dis):
            #self.alpha0 = 3
            #possible value on upper half of complex plane, physically realizable
            #self.alpha0 = random.uniform(2,3) #strength of laser
            #delta_t = random.uniform(1,10) #lasting time and delta of laser
            #alpha_i = self.alpha0*(1-np.exp(delta_t*1j))
            '''
            all value on complex plane, not physically realizable
            self.alpha0 = random.uniform(2,3)
            alpha_i = self.alpha0*np.exp(1j*2*3.1415926/self.num_dis*i)
            '''
            self.alpha.append(dis_alpha[i])
            dis_i = displace(self.dim,dis_alpha[i])
            dis_bas_i=[]
            P_i=[]
            for j in range (self.dim):
                dis_bas_ij=dis_i*self.bas[j]
                dis_bas_i.append(dis_bas_ij)
                #print(dis_bas_ij)
                #P_ij=(dis_bas_ij.dag()*self.example_matrix()*dis_bas_ij).tr()
                #P_i.append(P_ij)
            self.dis.append(dis_i)
            #self.P.append(P_i)
            self.dis_bas.append(dis_bas_i)
        return self.dis, self.dis_bas
    def generate_curves(self, om0, gamma0, num_points, t_scale):
        #generate curves
        t=np.linspace(0,t_scale,num_points)   #time range, num of time points
        
        curve_set=[]
        sum_y=[]
        for j in range (self.num_dis):
            sum_y=0
            for i in range (self.dim):
                y=[]
                Om=om0*math.sqrt(i+1)
                Gamma=gamma0*(i+1)**0.7
                y=self.P[j][i]*np.cos(Om*t)*np.exp(-Gamma*t)
                sum_y+=y
            curve_set.append(0.5*(1-sum_y))
        return curve_set
        
    def tomograph(self):
        shap_P=np.array(self.P).reshape(self.num_dis*self.dim,1).flatten()  #num_dis*dim=num of basis(displaced)[[][]...]->[...]
        shap_dis_bas=np.array(self.dis_bas).reshape(self.num_dis*self.dim,1).flatten() #shaped: [[Qobj(vector)],[],..]
        #regenerate measurement times N, event times n
        N = np.ones(len(shap_P))*10000  #[10000,10000,...]
        n = np.real(np.array(shap_P))*N   #[P1*10000,P2*10000,...]

        def ML_EstimateState(n,N,MeasurBasis):
            ''' Give result and measurement Basis to estimate the density matrix of the state
            
            n: sucess measurement shots
            N: total measurement shots
            MeasurBasis: measurement basis used in the experiment
            '''
            # judge how many qubits
            dims = MeasurBasis[0].dims[0]  #dim of basis vector, =[self.dim]
            dim = np.product(dims)  #value of dims, =self.dim

            
            def Gen_GM_Basis (dims): 
                ibasis = qi.tomography.gell_mann_basis(dims[0])
                ibasis = ibasis[range(len(ibasis))]
                if (len(dims) > 1):
                    return [ tensor(e, e1) for e in ibasis for e1 in  Gen_GM_Basis(dims[1:])]
                else:
                    return ibasis
                
            B_all = Gen_GM_Basis(dims)
            B0 = B_all[0]   #  1/d matrix, d is the dimension = 2**N
            B  = B_all[1:]  # pauli matrix list, length d-1
            # generate tranform matrix X
            X  = [[(ket2dm(Pi)*Bj).tr() for Bj in B] for Pi in MeasurBasis]
            X  = np.array(X)
            
            f = (n+0.5)/(N+1)
            Y = f- 1.0/dim
            
            a = np.sqrt(N/(f*(1-f)))
            Y1 = a*Y
            X1 = np.array([np.array(Xk)*ak for Xk,ak in zip(X,a)])
            
            # calculate initial value by linear transform
            x00 = np.array((np.linalg.inv((X.T.dot(X))).dot(X.T)).dot(Y)) #linear algebra: inverse
            
            from scipy.optimize import minimize
            # estimate function
            def estimate_fun(theta):
                return (np.linalg.norm(Y1-X1.dot(theta)))**2
            # to estimate values
            res = minimize(estimate_fun,x00)
            
            rho = np.sum([x*Bi for x, Bi in zip(res.x,B)],axis=0)+B0*B0  #res.x: solution of minimization, i.e. theta, theta*B=rho
        
            return (rho,res.x)

        return ML_EstimateState(n,N,shap_dis_bas)[0]


    def predict_tomograph(self, predict_P):
        shap_P=np.array(predict_P).reshape(self.num_dis*self.dim,1).flatten()
        shap_dis_bas=np.array(self.dis_bas).reshape(self.num_dis*self.dim,1).flatten()
        N = np.ones(len(shap_P))*10000
        n = np.real(np.array(shap_P))*N

        def ML_EstimateState(n,N,MeasurBasis):
            ''' Give result and measurement Basis to estimate the density matrix of the state
            
            n: sucess measurement shots
            N: total measurement shots
            MeasurBasis: measurement basis used in the experiment
            '''
            # judge how many qubits
            dims = MeasurBasis[0].dims[0]
            dim = np.product(dims)
            
            def Gen_GM_Basis (dims): 
                ibasis = qi.tomography.gell_mann_basis(dims[0])
                ibasis = ibasis[range(len(ibasis))]
                if (len(dims) > 1):
                    return [ tensor(e, e1) for e in ibasis for e1 in  Gen_GM_Basis(dims[1:])]
                else:
                    return ibasis
                
            B_all = Gen_GM_Basis(dims)
            B0 = B_all[0]   #  1/d matrix, d is the dimension = 2**N
            B  = B_all[1:]  # pauli matrix list, length d-1
            # generate tranform matrix X
            X  = [[(ket2dm(Pi)*Bj).tr() for Bj in B] for Pi in MeasurBasis]
            X  = np.array(X)
            
            f = (n+0.5)/(N+1)
            Y = f- 1.0/dim
            
            a = np.sqrt(N/(f*(1-f)))
            Y1 = a*Y
            X1 = np.array([np.array(Xk)*ak for Xk,ak in zip(X,a)])
            
            # calculate initial value by linear transform
            x00 = np.array((np.linalg.inv((X.T.dot(X))).dot(X.T)).dot(Y))
            
            from scipy.optimize import minimize
            # estimate function
            def estimate_fun(theta):
                return (np.linalg.norm(Y1-X1.dot(theta)))**2
            # to estimate values
            ## bounds on 15 paramters(res), no use, B is minus sometimes
            #bnd=((0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1))
            #res = minimize(estimate_fun,x00,bounds=bnd)
            res = minimize(estimate_fun,x00)
            #print('!!!!!!!!!!!')
            #print(res,B)
            rho = np.sum([x*Bi for x, Bi in zip(res.x,B)],axis=0)+B0*B0

            #positive matrix, method discribed in PRL
            val, vec = rho.eigenstates(sort='high')
            a = 0.0
            for i in range(len(val),0,-1):
                if (val[i-1]+a/i)<0:
                    a += val[i-1]
                else:
                    break
            eigv = val + a/i
            if (len(val)-i)>0:
                eigv[-(len(val)-i):] = 0.0

            rho = np.sum([v*ket2dm(w) for v, w in zip(eigv,vec)],axis=0)
            return (rho,res.x)
            
        return ML_EstimateState(n,N,shap_dis_bas)[0]