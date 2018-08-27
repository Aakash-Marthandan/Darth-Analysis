import numpy as np                                          #importing numpy for arrays
import pandas as pd                                         #importing pandas to read files
import matplotlib.pyplot as plt
import math
As2_data = pd.read_excel("swachhbharat.xlsx", "Sheet 0")    #  reading the excel file as a pandas array
As2_UP = np.array                                           # creating a numpy array to work with
As2_UP = As2_data[As2_data['StateName']=='Uttar Pradesh']   # reading the data rows corressponding to Uttar Pradesh into the numpy array
UP_data = np.array                                          # creating a new numoy array
UP_data = np.vstack([As2_UP['TotalVillage'], As2_UP['TotalODFVillage']])          # isolating just the columns needed - total villages and ODF villages
UP = []                                                     # new array to store the fractional values
l = len(UP_data[0])
for i in range(l):                                           # loop to compute the fractional value and store it in UP (if the fraction is defined)
    if UP_data[1,i] !=0:
        UP.append(UP_data[1,i]/UP_data[0,i])
    else:
        UP.append(0)
UP_l = len(UP)                  
sum_UP = 0
for k in UP:                                                # loop to sum up the fractions and calculate mean 
    sum_UP  += k
mean = sum_UP/UP_l
Sum_UP = 0                                                  # loop to sum up the square of the diffrences
for t in UP:
    Sum_UP += math.pow((mean - t), 2)
Var_UP = Sum_UP/(UP_l - 1)                                  #calculating variance and hence standard deviation
stdd = math.sqrt(Var_UP)
plt.figure(1)                                               #plotting the histogram with standard deviation and mean marked
plt.hist(UP, bins = 'auto', edgecolor = 'black')            # auto function is used as last assignment
plt.vlines(x = mean, ymin = 0, ymax = 160, colors = 'red', label = 'Mean')
plt.hlines(y = 0, xmin = mean-stdd, xmax = mean + stdd, colors = 'yellow', label = 'Standard Deviation')
plt.xlabel('Fraction of villages declared as ODF')
plt.ylabel('# of Districts in Uttar Pradesh') 
plt.title('Distribution of ODF villages in UP districts')   
plt.legend()
plt.show()  
