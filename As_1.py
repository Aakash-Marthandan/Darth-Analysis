import numpy as np
import matplotlib.pyplot as plt
As1_data = np.genfromtxt('pre_primary_school_data.csv', delimiter=',', skip_header = 1, usecols = (12, 13, 14, 15))
print(*As1_data)
for i in range(105):
    if As1_data[i, 0] > 0:
      As1_data[i, 3] = (As1_data[i, 1]+ As1_data[i, 2])/As1_data[i, 0]
    else:
      As1_data[i, 3]=0
As_R = []
As_U = []
As_T = []
x=0
while x<35:
    As_R.append(As1_data[3*x, 3])
    As_U.append(As1_data[3*x+1, 3])
    As_T.append(As1_data[3*x+2, 3])
    x+=1
plt.hist(As_R, bins='auto')
plt.title('# of States vs Rural Pre-primary teachers')
plt.xlabel('Teachers per institution')
plt.ylabel('Number of States')
plt.show()
plt.hist(As_U, bins='auto')
plt.title('# of States vs Urban Pre-primary teachers')
plt.xlabel('Teachers per institution')
plt.ylabel('Number of States')
plt.show()
plt.hist(As_T, bins='auto')
plt.title('# of States vs Total Pre-primary teachers')
plt.xlabel('Teachers per institution')
plt.ylabel('Number of States')
plt.show()
