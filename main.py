import numpy as np
import random
import math
import matplotlib.pyplot as plt

ns=12
na=8
b=10
p=5
M=40
B=10
Tc=5
T=15
MDP_step=(M+Tc*B)*T
gamma=0.95

#table of states
i=1
s=[]
while i<ns+1:
    s.append(i)
    i=i+1


#table of actions
i=1
a=[]
while i<na+1:
    a.append(i)
    i=i+1


#reward
i=1
r=[]
while i<ns+1:
  r.append(random.random())
  i=i+1



#generate MDP

MDP_S=[]
MDP_A=[]
MDP_R=[]

for i in range(10):
  MDP_s=[]
  MDP_a=[]
  MDP_r=[]
  s_start=random.randint(1,ns)
  MDP_s.append(s_start)
  j=1
  while j<MDP_step+1:
    MDP_a.append(random.randint(1,na))
    MDP_s.append(random.randint(1,ns))
    j=j+1
  MDP_S.append(MDP_s)
  MDP_A.append(MDP_a)
  for k in MDP_s:
    MDP_r.append(r[k-1])
  MDP_R.append(MDP_r)

print(len(MDP_S[2]))


#feature matrix
i=0
fea=[]
while i<na*ns:
  k=0
  ks=[]
  while k<p:
    ks.append(random.random())
    k=k+1
  fea.append(ks)
  i=i+1


for i in fea:
  i=i/(np.dot(i,i)**0.5)

phi={}
for i in s:
  for j in a:
    phi[i,j]=fea[(i-1)*na+j-1]



import math
theta=np.array([0]*p)
def softmax(theta,state,action):
  return 1/na


def grad_softmax(theta,state,action):
  return np.array([0]*p)

def Vbar(theta,state):
  sum=0
  for i in a:
    sum=sum+softmax(theta,state,i)*np.dot(theta,np.array(phi[state,i]))
  return sum

def delta(theta,state,action,newstate):
  return r[state-1]+gamma*Vbar(theta,newstate)-np.dot(theta,np.array(phi[state,action]))

def hatphi(theta,state):
  sum1=np.array([0]*p)
  for i in a:
    sum1=sum1+softmax(theta,state,i)*np.array(phi[state,i])+np.dot(theta,phi[state,i])*grad_softmax(theta,state,i)
  return sum1



omega=np.array([0]*p)

def G(theta,omega,state,action,newstate):
  return delta(theta,state,action,newstate)*np.array(phi[state,action])-gamma*(np.dot(omega,np.array(phi[state,action])))*hatphi(theta,newstate)



def H(theta,omega,state,action,newstate):
  return delta(theta,state,action,newstate)*np.array(phi[state,action])-np.dot(np.array(phi[state,action]),omega)*np.array(phi[state,action])


def gradJ(theta):
  sum1=0*(np.outer(grad_softmax(theta,1,1),phi[1,1]))
  sum2=np.array([0]*p)
  for i in s:
    for j in a:
      for k in s:
        for t in a:
          sum1=sum1+softmax(theta,k,t)*np.outer(phi[k,t],phi[i,j])/(ns*na*b)
        sum2=sum2+np.dot(delta(theta,i,j,k),phi[i,j])
  sum2=sum2/(ns*na*b)
  return np.dot(sum1,sum2)

GQ=[]
u=0
for u in range(5):
    theta=np.array([0]*p)
    omega=np.array([0]*p)
    gq=[]
    k=0
    while k<MDP_step:
        theta0 = theta
        omega0 = omega
        gq.append(np.dot(gradJ(theta), gradJ(theta))*(1))
        theta=theta+(0.01/1)*G(theta0,omega0,MDP_S[u][k*1],MDP_A[u][k*1],MDP_S[u][k*1+1])
        omega=omega+(0.05/1)*H(theta0,omega0,MDP_S[u][k*1],MDP_A[u][k*1],MDP_S[u][k*1+1])
        print(k)
        k=k+1
    GQ.append(gq)

print(GQ)



GQu=[]
GQl=[]
for i in range(len(gq)):
    GQu.append(max(GQ[0][i],GQ[1][i],GQ[2][i],GQ[3][i],GQ[4][i]))
    GQl.append(min(GQ[0][i], GQ[1][i], GQ[2][i],GQ[3][i],GQ[4][i]))


M1=[]
u=0
for u in range(5):
    theta=np.array([0]*p)
    omega=np.array([0]*p)
    mini0=[]
    k=0
    while k<MDP_step//10:
        j=0
        theta0 = theta
        omega0 = omega
        while j<10:
            theta0 = theta
            mini0.append(np.dot(gradJ(theta), gradJ(theta))*(1))
            theta=theta+(0.1/10)*G(theta0,omega0,MDP_S[u][k*10+j],MDP_A[u][k*10+j],MDP_S[u][k*10+j+1])
            omega=omega+(0.5/10)*H(theta0,omega0,MDP_S[u][k*10+j],MDP_A[u][k*10+j],MDP_S[u][k*10+j+1])
            j=j+1
        print(k)
        k=k+1
    M1.append(mini0)
#mini0=mini0[0:1000]


m1u=[]
m1l=[]
for i in range(len(mini0)):
    m1u.append(max(M1[0][i],M1[1][i],M1[2][i],M1[3][i],M1[4][i]))
    m1l.append(min(M1[0][i], M1[1][i], M1[2][i],M1[3][i],M1[4][i]))

M2=[]
u=0
for u in range(5):
    theta=np.array([0]*p)
    omega=np.array([0]*p)
    mini1=[]
    k=0
    while k<MDP_step//20:
        j=0
        theta0 = theta
        omega0 = omega
        while j<20:
            theta0 = theta
            mini1.append(np.dot(gradJ(theta), gradJ(theta))*(1))
            theta=theta+(0.2/20)*G(theta0,omega0,MDP_S[u][k*20+j],MDP_A[u][k*20+j],MDP_S[u][k*20+j+1])
            omega=omega+(1/20)*H(theta0,omega0,MDP_S[u][k*20+j],MDP_A[u][k*20+j],MDP_S[u][k*20+j+1])
            j=j+1
        print(k)
        k=k+1
    M2.append(mini1)
#mini0=mini0[0:1000]
m2u=[]
m2l=[]
for i in range(len(mini1)):
    m2u.append(max(M2[0][i],M2[1][i],M2[2][i],M2[3][i],M2[4][i]))
    m2l.append(min(M2[0][i], M2[1][i], M2[2][i],M2[3][i],M2[4][i]))

M3=[]
u=0
for u in range(5):
    theta=np.array([0]*p)
    omega=np.array([0]*p)
    mini2=[]
    k=0
    while k<MDP_step//15:
        j=0
        theta0 = theta
        omega0 = omega
        while j<15:
            theta0 = theta
            mini2.append(np.dot(gradJ(theta), gradJ(theta))*(1))
            theta=theta+(0.15/15)*G(theta0,omega0,MDP_S[u][k*15+j],MDP_A[u][k*15+j],MDP_S[u][k*15+j+1])
            omega=omega+(0.75/15)*H(theta0,omega0,MDP_S[u][k*15+j],MDP_A[u][k*15+j],MDP_S[u][k*15+j+1])
            j=j+1
        print(k)
        k=k+1
    M3.append(mini2)
m3u=[]
m3l=[]
for i in range(len(mini2)):
    m3u.append(max(M3[0][i],M3[1][i],M3[2][i],M3[3][i],M3[4][i]))
    m3l.append(min(M3[0][i], M3[1][i], M3[2][i],M3[3][i],M3[4][i]))


N1=[]
for u in range(5):
    theta=np.array([0]*p)
    omega=np.array([0]*p)
    n1=[]
    k=0
    while k<MDP_step//14:
        j=0
        theta0 = theta
        omega0 = omega
        while j<14:
            theta0 = theta
            n1.append(np.dot(gradJ(theta), gradJ(theta))*(1))
            theta=theta+(0.01/30)*G(theta0,omega0,MDP_S[u][k*14+j],MDP_A[u][k*14+j],MDP_S[u][k*14+j+1])
            omega=omega+(0.05/30)*H(theta0,omega0,MDP_S[u][k*14+j],MDP_A[u][k*14+j],MDP_S[u][k*14+j+1])
            j=j+1
        print(k)
        k=k+1
    N1.append(n1)
n1u=[]
n1l=[]
for i in range(len(n1)):
    n1u.append(max(N1[0][i],N1[1][i],N1[2][i],N1[3][i],N1[4][i]))
    n1l.append(min(N1[0][i], N1[1][i], N1[2][i],N1[3][i],N1[4][i]))



N2=[]
for u in range(5):
    theta=np.array([0]*p)
    omega=np.array([0]*p)
    n2=[]
    k=0
    while k<MDP_step//30:
        j=0
        theta0 = theta
        omega0 = omega
        while j<30:
            theta0 = theta
            n2.append(np.dot(gradJ(theta), gradJ(theta))*(1))
            theta=theta+(0.01/30)*G(theta0,omega0,MDP_S[u][k*30+j],MDP_A[u][k*30+j],MDP_S[u][k*30+j+1])
            omega=omega+(0.05/30)*H(theta0,omega0,MDP_S[u][k*30+j],MDP_A[u][k*30+j],MDP_S[u][k*30+j+1])
            j=j+1
        print(k)
        k=k+1
    N2.append(n2)
n2u=[]
n2l=[]
for i in range(len(n2)):
    n2u.append(max(N2[0][i],N2[1][i],N2[2][i],N2[3][i],N2[4][i]))
    n2l.append(min(N2[0][i], N2[1][i], N2[2][i],N2[3][i],N2[4][i]))


le=min(len(GQl),len(m1u),len(m2l),len(m3l),len(n1l),len(n2l))
T=[]
for i in range(le):
    T.append(i)




plt.fill_between(T,GQl[0:le],GQu[0:le],label="vanilla",alpha=0.3)
plt.fill_between(T,m1l[0:le],m1u[0:le],label='mini-batch,10',alpha=0.3)
plt.fill_between(T,m2u[0:le],m2l[0:le],label='mini-batch,20',alpha=0.3)
plt.fill_between(T,m3l[0:le],m3u[0:le],label='mini-batch,15',alpha=0.3)
plt.fill_between(T,n1u[0:le],n1l[0:le],label='nested-loop, B=10, M=20',alpha=0.3)
plt.fill_between(T,n2l[0:le],n2u[0:le],label='nested-loop, B=20, M=25',alpha=0.3)
plt.xlabel('number of samples used', fontsize=10)
plt.ylabel('norm of gradient', fontsize=10)
plt.legend( fontsize=10)
plt.show()
