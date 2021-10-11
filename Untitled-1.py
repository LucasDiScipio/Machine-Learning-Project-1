import numpy as np
import matplotlib.pyplot as plt
x = np.array ( [ 1 , 2 , 3 , 4 ] )
y1 = x
y2 = x**2
plt.plot (x , y1,'r')
plt.plot (x , y2 ,'b')
plt.xlabel ('x')
plt.ylabel ('y')
plt.show()