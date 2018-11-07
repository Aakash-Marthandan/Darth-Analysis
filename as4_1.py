# -- coding: utf-8 --
"""
Created on Fri Nov  2 18:37:48 2018

@author: Raunak
"""
import numpy as np
import pandas as pd
from scipy.integrate import quad
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import spline

def darkmatterfunction(x):
    if  x<15 or x>25:
       return 0
    else:
       return sigma*20*(25-x)

def backgroundrate(x):
    return 1000*(math.exp(-x/10))

data=1
backgroundsignalcalc=[0]*40
darkmattersignalcalc=[[0 for j in range(40)] for i in range(5)]



data = np.genfromtxt("recoilenergydata_EP219.csv", delimiter=',', skip_header = 1, usecols = (0,1))
dt=np.transpose(data)


err=0
for i in range(40):
    backgroundsignalcalc[i],err=quad(backgroundrate,data[i][0]-.5,data[i][0]+.5)

    for j in range(5):
        sigma=10**(j-2)
        darkmattersignalcalc[j][i],err=quad(darkmatterfunction,data[i][0]-.5,data[i][0]+.5)
        darkmattersignalcalc[j][i]+=backgroundsignalcalc[i]
        j+=1

plt.bar(dt[0],dt[1])
plt.title('Measured signal')
plt.xlabel('Energy in kEV')
plt.ylabel('Number of events')
plt.show()


plt.bar(dt[0],backgroundsignalcalc)
plt.title('Calculated Background signal')
plt.xlabel('Recoil energy in kEV')
plt.ylabel('Number of events')
plt.savefig('BackgroundSignal.png')
plt.show()


for j in range(5):
    sigma=10**(j-2)
    print("Total signal for sigma=", sigma)
    plt.bar(dt[0],darkmattersignalcalc[j])
    plt.title('Calculated signal')
    plt.xlabel('Energy in kEV')
    plt.ylabel('Number of events')
    plt.savefig('Signal for Sigma.png')
    plt.show()







#
#We call the array containing predicted values of N for sigma=0.01 as N_1 (which has 40 elements) and so on
#N is the array obtained from csv file for no. of events (Not really)

j=1
N=np.transpose(darkmattersignalcalc)
L=[0]*5
m=M=[[0.0]*5]*40
L[j] = M[0][j]

for j in range(0,5):
    for i in range(0,40):
        m[i][j]=(N[i][j] + abs(N[i][j]- data[i][1]))
        M[i][j]=data[i][1]*math.log(m[i][j])-math.log(data[i][1])-m[i][j]   #Calculated in log itself since product is too big
        L[j] = L[j]+M[i][j]                                       #LOG Likelihood

s=[-2,-1,0,1,2]
plt.plot(s,L)
plt.title('Maximum likelihood Estimate')
plt.xlabel('Log Sigma')
plt.ylabel('Log likelihood')
plt.savefig('LogLikelihood.png')
plt.show()

s1=np.array([-2,-1,0,1,2])
xnew=np.linspace(s1.min(),s1.max(),300) #interpolating the given data into a smooth curve to find the 1-sigma interval
L_new=spline(s,L,xnew)
q=0
j=0
for i in range(len(L_new)):
    j=abs(L_new[i]-(L[0]/math.sqrt(math.e)))  #since even after interpolation, the values of the likelihood don't exactly drop by sqrt(e), we found the only value present in the interpolated set which has an error of 54 points of L_max/sqrt(e).
    if j<54.75:
        q=xnew[i]
        print(q)#the value of the order of sigma(parameter) for the 1-sigma interval
        print("The Sigma interval Value")
    else:
        continue

plt.plot(xnew,L_new)
z=plt.axvline(q)
g=plt.axvline(-2)
plt.xlabel("Log sigma")
plt.ylabel("Value of Log Likelihood")
plt.show()
