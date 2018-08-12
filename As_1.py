import numpy as np                      #numpy to create and handle a numpy array
import matplotlib.pyplot as plt         #matplotlib to plot (more manageable than numpy histograms)
As1_data = np.genfromtxt('As1_pre_primary_school_data.csv', delimiter=',', skip_header = 1, usecols = (12, 13, 14, 15)) #getting specific input columns from data file and skipping the first row that includes strings
print(*As1_data)                        #for verification purposes
for i in range(105):                    #As there are 105 rows
    if As1_data[i, 0] > 0:              #since division is involved
      As1_data[i, 3] = (As1_data[i, 1]+ As1_data[i, 2])/As1_data[i, 0]
    else:      
      As1_data[i, 3]=0
As_R = []                               #declaring arrays
As_U = []
As_T = []
x=0
while x<35:
    As_R.append(As1_data[3*x, 3])       #assigning the sequential values from the dataset
    As_U.append(As1_data[3*x+1, 3])
    As_T.append(As1_data[3*x+2, 3])
    x+=1
plt.hist(As_R, bins='auto')     #auto function is a new feature availabe in the updated libraries
plt.title('# of States vs Rural Pre-primary teachers')  # that automatically varies bin sizes based on the data
plt.xlabel('Teachers per institution')
plt.ylabel('Number of States')
plt.show()                              #Plotting
plt.hist(As_U, bins='auto')
plt.title('# of States vs Urban Pre-primary teachers')
plt.xlabel('Teachers per institution')
plt.ylabel('Number of States')
plt.show()                              #Plotting x2
plt.hist(As_T, bins='auto')
plt.title('# of States vs Total Pre-primary teachers')
plt.xlabel('Teachers per institution')
plt.ylabel('Number of States')
plt.show()                              #Plotting x3
