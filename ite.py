from __future__ import division
from sympy import *
from qutip import *
import numpy as np
import qinfer as qi
import matplotlib.pyplot as plt
import math



num_dis=16
dim_r=4
dim=dim_r
dis_alpha=[]
bas=[]
bas_r=[]
dis_bas=[]
theta=2
phi=0
N=dim


rho_coherent = coherent_dm(N, -1.0j*theta/2*np.exp(-1.0j*phi)) #dim, alpha
# bas_r=rho_coherent.eigenstates()[1]

sta=(rho_coherent+thermal_dm(dim,8))/2
bas_r=sta.eigenstates()[1]


# for i in range (dim_r):    #generate measurement basis
#     bas_r.append(basis(dim_r,i))

    
for i in range (dim):    #generate measurement basis
    bas.append(basis(dim,i))
    
    
for theta in [1.0, 2.0]:
    for phi in np.linspace(0, 2 * 3.14159, 9)[:-1]:
        dis_alpha.append(1.0j*theta/2*np.exp(-1.0j*phi))
for i in range(num_dis):
    dis_i = displace(dim,dis_alpha[i])
    dis_bas_i=[]
    for j in range (dim):
        dis_bas_ij=dis_i*bas[j]
        #dis_bas_i.append(dis_bas_ij)
        #dis_bas.append(dis_bas_i)
        dis_bas.append(dis_bas_ij)
#print(dis_bas[0])
#print(dis_bas[1].norm())


#import experiment fitting data:f
import pickle
ff = open('data_p.pkl','rb')
f=pickle.load(ff)
f=np.array(f).reshape(num_dis*dim)
#print(len(f))
f=f/np.sum(f)
#print(np.sum(f))

#initializing
#r=np.array(rho_coherent.eigenstates()[0]).reshape(1,dim_r)
# r=np.array(dim_r*[1/dim_r]).reshape(1,dim_r)
r=np.array(sta.eigenstates()[0]).reshape(1,dim_r)
print(r)
rho_0=ket2dm(Qobj([dim_r*[0]]))
for i in range(dim_r):
    rho_0+=r[0][i]*ket2dm(bas_r[i])
#print(rho_0)

#rho_0=maximally_mixed_dm(dim)


#h matrix
h=[]
for i in range(dim_r):
    h_i=[]
    for j in range(len(f)):
        #print(bas[i].trans(),dis_bas[j])
        h_ij0=bas_r[i].trans()*dis_bas[j]
        #print(h_ij0)
        h_ij=h_ij0*h_ij0.conj()
        #print(h_ij)
        h_i.append(h_ij)
    h.append(h_i)
#print(np.array(h).reshape(dim,num_dis*dim))
#print(h[1])
print(h)

p=(r.dot(h)).reshape(len(f))
p_=[]
for i in range(len(f)):
    p_.append(p[i].full().reshape(1))
    if p[i].full()==0:
        print(i,h[7][i])  #if =0, -> p[i]=0 -> r_=nan (error)
p_=np.array(p_).reshape(len(f))
#print(p_)  #array
#print(p.data)




''''''
#normoalizing the measurement basis
sum_dis_bas=ket2dm(Qobj([dim*[0]]))
for i in range(len(dis_bas)):
    dis_bas[i]=dis_bas[i]/(np.sqrt(num_dis)*dis_bas[i].norm()) #dis_bas[i].norm()=1, state/sqr(n)<=>dm/n  #normalizing -> complete
    sum_dis_bas+=ket2dm(dis_bas[i])
#print(sum_dis_bas.norm())
print(sum_dis_bas)


ite=30
#print(len(f/p_))
for i in range (ite):
    print(i)

    b=np.array(h).dot((f/p_))
    print(p_)
    print('b', b)
    r=r.reshape(dim_r)
    r_=[]
    for j in range (dim_r):
        #print(r[j],b[j])
        rj=r[j]*b[j]
        r_.append(rj.full())
    r_=np.array(r_).reshape(dim_r)
    #np.nan_to_num(r_)
    print('r:', r)
    print('r_', r_)

    #R
    R=ket2dm(Qobj([dim*[0]]))
    for j in range(len(dis_bas)):
        R+=(f/p_)[j]*ket2dm(dis_bas[j])
    #print(R)

    #rho
    rho=ket2dm(Qobj([dim_r*[0]]))
    for i in range(dim_r):
        rho+=r[i]*ket2dm(bas_r[i])
    #G
    #print(rho)
    G=1.0j*commutator(rho, R)
    #print(G)

    #U
    epsil=0.5
    U=1+1.0j*epsil*G
    U=U/U.tr()*dim  #->1
    print(U.trans()*U)
    print('!!!!!!!')
    print(U)

    #new bas
    bas_=U*bas_r
    #print(bas_)

    bas_r=bas_
    r=r_
    
    h=[]
    for i in range(dim_r):
        h_i=[]
        for j in range(len(f)):
            #print(bas[i].trans(),dis_bas[j])
            h_ij0=bas_r[i].trans()*dis_bas[j]
            #print(h_ij0)
            h_ij=h_ij0*h_ij0.conj()
            #print(h_ij)
            h_i.append(h_ij)
        h.append(h_i)
    
    p=(r.dot(h)).reshape(len(f))
    p_=[]
    for i in range(len(f)):
        p_.append(p[i].full().reshape(1)) 
    p_=np.array(p_).reshape(len(f))
    
    
    
    rho_=ket2dm(Qobj([dim_r*[0]]))
    for i in range(dim_r):
        rho_+=r[i]*ket2dm(bas_r[i])
    print(rho_)


print('bas:',bas) 
print('r:',r)
lbls_list = [[str(d) for d in range(dim)], ["u", "d"]]
xlabels = []
for inds in tomography._index_permutations([len(lbls) for lbls in lbls_list]):
    xlabels.append("".join([lbls_list[k][inds[k]]for k in range(len(lbls_list))]))
fig, ax = matrix_histogram(rho_, xlabels, xlabels, limits=[0,1])
ax.view_init(azim=-55, elev=45)    

print(rho_.tr())
plot_wigner_fock_distribution(rho_)
plt.show()



plot_wigner_fock_distribution(rho_coherent)
plt.show()
print(fidelity(rho_, rho_coherent))