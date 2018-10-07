import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

df_cr=pd.read_csv('crimerate.csv')
df_up=pd.read_csv('unemploymentrate.csv')

#print(df_cr.columns)
#print(df_up.columns)

var1=df_cr['Year - 2016 (Col.6)'][df_cr['Category (Col.2)'].isin(['State','UT'])].sum()
var1=var1/2
num1=np.array
num1= df_cr['Year - 2016 (Col.6)'][df_cr['Category (Col.2)'].isin(['State','UT'])]
meancr=var1/(len(num1)-2)
print(meancr)

var2=np.array
var2=df_up['2015-16']
var3=var2[:-1].copy()        #var4 has the column of unemploymentrate 2016
guru=var3.sum()
meanup=guru/(len(var3))
var4=[]
for i in range(len(var3)-1):
    if i<5:
        var4.append(var3[i])     #code to fix the postion of delhi
    else :
        var4.append(var3[i+1])

var4.insert(33,3.1)#insert the value of delhi at the requisite position
print(meanup)
var21=var4[29]
var4[29]=var4[30]       #subcode to interchange the positions of Uttar Pradesh and Uttarkhand to later relaye it with the crime array
var4[30]=var21
#standard deviation for cr
num3=np.array
num3=num1[:-1].copy()
l=len(num3)
up=[]
for i in range(l-1):
  if i<29:
     up.append(num3[i])
     print(up[i])
     print(i)
  else:
      up.append(num3[i+1])
      print(up[i])
      print(i)              #up has the crimerate columns, has 36 entities in it
j=0
#s.dev for cr
for t in range(len(up)):
    j=((meancr-up[i])**2)+j

sdevcr1=math.sqrt(j/(len(up)-1))
print(sdevcr1)
 #this sdev corresponds to the deviation of each crimerate of the individual
 #state from the estimated mean i.e. it is an estimate of sdev of the original pdf
sdevcr2=sdevcr1/(math.sqrt(len(up)))
print(sdevcr2)
#sd dev for up
varup=0#defining the variance variable
for i in range(len(var4)):
    varup=varup+((var4[i]-meanup)**2)

varup=varup/(len(var4)-1)
sdevup1=math.sqrt(varup)#this gives the estimated value of the variance of the original pdf but not the error in the estimated mean
print(sdevup1)
sdevup2=sdevup1/(math.sqrt(len(var4)))#this gives the error in the estimated mean
print(sdevup2) #the difference in sdevup2 and sdevup1 shows that when we average our precision improves and results in more accuracy

#print(var3)

#plot of Unemployment
plt.hist(var4, bins=10)
plt.title('Unemployment rate vs Number of States')
plt.xlabel('Unemployment Rate in %')
plt.ylabel('Number of States')
plt.axvline(meanup, color='k', linestyle='solid', linewidth=1)
plt.axvline(meanup-(sdevup1/2),color='r',linestyle='dashdot',linewidth=0.5)
plt.axvline(meanup+(sdevup1/2),color='r',linestyle='dashdot',linewidth=0.5)
plt.show()

#plot of cr
plt.hist(up,bins='auto')
plt.title('Crimerate in different States and UTs')
plt.xlabel('Crimerate (in crimes per 100,000)')
plt.ylabel('Number of States and UTs')
plt.axvline(meancr, color='k', linestyle='solid', linewidth=1)
plt.axvline(meancr-(sdevcr1/2),color='r',linestyle='dashdot',linewidth=0.5)
plt.axvline(meancr+(sdevcr1/2),color='r',linestyle='dashdot',linewidth=0.5)
plt.show()
plt.scatter(var4,up)
plt.show()
plt.hist2d(var4,up,bins=(10,20))
plt.show()
#correlation sqaure calculation
correlation=0
for i in range(len(var4)):
    correlation=(var4[i]-meanup)*(up[i]-meancr)+correlation

correlation=-correlation/(len(var4)-1)
correlation=math.sqrt(correlation/(len(var4)))#correlation for the final data
print(correlation/(math.sqrt(sdevup2*sdevcr2)))