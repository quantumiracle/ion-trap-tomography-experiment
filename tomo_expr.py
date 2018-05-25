# -*- coding: UTF-8 -*-  
from sympy import *
from qutip import *
import numpy as np
import qinfer as qi
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
from measure import Measure
#from NNpredictor import Predictor
from curve_fit_class import Optimizer
from curve_fit_class import Optimizer_fix_omega
from curve_fit_class import Optimizer_lmfit
import pickle




dim = 4
num_dis = 16  #number of displacements
#Rabi_time_blue = 58.0e-6
#om0=2*3.14159/Rabi_time_blue   #om: 10k-100k(0-100k)
gamma0=2.0e-1  #gamma: 0-10k
#num_points=100

#############################
#
#load experiment data
#
#############################
fn = open('data.pkl','rb')
exp_data=pickle.load(fn)
num_pts=120
ts = np.linspace(0,2*num_pts,num_pts)  #num_pts
dis_alpha=[]
bsb=np.array(exp_data)[:,0,:num_pts]
exp_P=np.array(exp_data)[:,1,:num_pts]

for theta in [1.0, 2.0]:
    for phi in np.linspace(0, 2 * 3.14159, 9)[:-1]:
        dis_alpha.append(1.0j*theta/2*np.exp(-1.0j*phi))
        #dis_alpha.append(theta * np.exp(1.0j * phi))



#############################
#
#blue sideband Rabi time t0 fitting 
#
#############################


def funSin(x, x0, t0, A, B, tau):
    Y = B - A * np.exp(-(x - x0) / tau) * np.cos(2 * 3.14159 * (x - x0) / t0) / 2.0 
    return Y

def fitRabi(t, y):
    '''

    '''
    data_fft = abs(np.fft.fft(y))
    pos = max(enumerate(data_fft[1:len(t) / 2]), key=lambda x: x[1])[0]
    xmin = t[0]
    xmax = t[-1]
    xscale = xmax - xmin
    t0 = xscale / (pos + 1)
    A = max(y)
    B = 0.5
    x0 = 0.0
    tau = 0.5e-3  # 500ms ms
    op = [x0, t0, A, B, tau]

    popt, pcov = curve_fit(funSin, t, y, op)
    # perr = np.sqrt(np.diag(pcov)
    #print (1.0 / popt[1], popt[1])

    return (1.0 / popt[1], popt, pcov)
bsb_time=[]  #Rabi time
om0=[]    #Rabi frequency
for i in range(len(bsb)):
    data=bsb[i]
    W0, popt, pcov = fitRabi(ts*1.0e-6, data)   #fit
    #plt.plot(ts,data, 'o')
    #plt.plot(ts,funSin(ts*1e-6,*popt))

    #print('x0, t0, A, B, tau')
    #print(popt)
    bsb_time.append(popt[1])
    om0_i=2*3.14159/popt[1]
    om0.append(om0_i)
    #plt.show()
#print(bsb_time,om0)


#############################
#
#tomography
#
#############################

measure = Measure(dim = dim, num_dis = num_dis)
measure.generate_basis()
#print(example_matrix)
dis, dis_bas = measure.measure(dis_alpha)

'''choosing optimizer'''
#density matrix tomograph with predicted P
###### 1. Neural Network predictor
#predictor = Predictor()
#predict_P = predictor.NNpredictor(dim, curve_set)
###### 2. scipy curve fit(omega also need fitting)
#optimizer = Optimizer()
###### 3. scipy curve fit(given omega, manual setting om0)
#optimizer = Optimizer_fix_omega()
###### 4. lmfit with constraints on fitting : sum(pi)=1
optimizer = Optimizer_lmfit(dim=dim)
curve_fit_P_set=[]
for i in range(num_dis):
    curve_fit_P=[]
    #curve_fit_P = optimizer.curve_fitter(dim, curve_set[i], num_points, t_scale)
    #fit gamma
    #curve_fit_P = optimizer.curve_fitter(om0[i], dim, exp_P[i], ts*1e-6)
    #give gamma
    curve_fit_P = optimizer.curve_fitter(om0[i], gamma0, dim, exp_P[i], ts*1e-6)
    curve_fit_P_set.append(curve_fit_P)
print("!!!!!!!!!")
print(curve_fit_P_set)

predict_tomo_matrix = measure.predict_tomograph(curve_fit_P_set)

'''theoritical coherent state: 1. qutip  2. D(alpha)*state0 ; 1、2 is same as long as dim=N(no boundary effect)'''
#qutip coherent state
N=15
theta=2
phi=0
rho_coherent = coherent_dm(N, -1.0j*theta/2*np.exp(-1.0j*phi)) #dim, alpha
plot_wigner_fock_distribution(rho_coherent)


theta=2
phi=0
alpha=-1.0j*theta/2*np.exp(-1.0j*phi)  #when alpha=theta*np.exp(-1.0j*phi), same as qutip coherent
D=displace(dim,alpha)
coh_state=D*basis(dim,0)
coh_matrix=ket2dm(coh_state)
state0=ket2dm(basis(dim,0))
print(predict_tomo_matrix)
print(fidelity(predict_tomo_matrix, coh_matrix))
plot_wigner_fock_distribution(predict_tomo_matrix)
plot_wigner_fock_distribution(coh_matrix)

''''''
lbls_list = [[str(d) for d in range(dim)], ["u", "d"]]
xlabels = []
for inds in tomography._index_permutations([len(lbls) for lbls in lbls_list]):
    xlabels.append("".join([lbls_list[k][inds[k]]for k in range(len(lbls_list))]))
fig, ax = matrix_histogram(coh_matrix, xlabels, xlabels, limits=[0,1])
ax.view_init(azim=-55, elev=45)


lbls_list = [[str(d) for d in range(dim)], ["u", "d"]]
xlabels = []
for inds in tomography._index_permutations([len(lbls) for lbls in lbls_list]):
    xlabels.append("".join([lbls_list[k][inds[k]]for k in range(len(lbls_list))]))
fig, ax = matrix_histogram(predict_tomo_matrix, xlabels, xlabels, limits=[0,1])
ax.view_init(azim=-55, elev=45)

plt.show()