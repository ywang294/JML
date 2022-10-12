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
T=5
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

NE=[]
GR=[]
MI=[]

u=1

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
#mini0=mini0[0:1000]

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
#mini0=mini0[0:1000]




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
#mini0=mini0[0:1000]



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
#mini0=mini0[0:1000]



theta=np.array([0]*p)
omega=np.array([0]*p)
ne1=[]
k=0
while k<MDP_step//12:
    j=0
    theta0 = theta
    omega0 = omega
    while j<2:
        ne1.append(np.dot(gradJ(theta), gradJ(theta))*(1))
        omega=omega+0.1*H(theta0,omega0,MDP_S[u][12*k+j],MDP_A[u][12*k+j],MDP_S[u][12*k+j+1])
        j=j+1
    theta0 = theta
    omega0 = omega
    t=0
    while t<10:
        ne1.append(np.dot(gradJ(theta), gradJ(theta)) * (1))
        theta = theta0 + 0.1 / 10 * G(theta0, omega0, MDP_S[u][12 * k + 2+t], MDP_A[u][12 * k + 2+t], MDP_S[u][12 * k + 2+t + 1])
        t=t+1
    print(k)
    k=k+1






plt.plot(gq,label="vanilla")
plt.plot(mini0,label='mini-batch,10')
plt.plot(mini1,label='mini-batch,20')
plt.plot(mini2,label='mini-batch,15')
plt.plot(ne1,label='nest')
plt.xlabel('number of samples used', fontsize=10)
plt.ylabel('norm of gradient', fontsize=10)
plt.legend( fontsize=10)
plt.show()