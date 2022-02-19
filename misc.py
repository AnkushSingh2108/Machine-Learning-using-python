# import matplotlib.pyplot as plt
# x1 = [1,2,3,4,5]
# y1 = [1,2,5,6,8]
# plt.plot(x1,y1)

# x2 = [5,6,7,89,54]
# y2 = [5,6,7,89,54]
# plt.plot(x2,y2)

# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# x1 = np.linspace(0.0,6.0)
# y1 = np.sin(2*np.pi*x1)

# x2 = np.linspace(0.0,6.0)
# y2  = np.sin(2*np.pi*x2)

# plt.subplot(2,1,1)
# plt.plot(x1,y1,"--")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Sine Graph")

# plt.subplot(2,1,2)
# plt.plot(x2,y2,"--")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Cosine Graph")
# plt.grid()
# plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  #this module is used for 3D plotting
import numpy as np


X = np.arange(-5,5,0.25)
Y = np.arange(-5,5,0.25)
X,Y = np.meshgrid(X,Y)
R = np.sqrt(X**2 + Y**2)
S = np.sqrt(X + Y)
B = np.sin(R)
Z = np.exp(R)
A = np.exp(-R)
D = np.exp(S)
F = np.exp(-S)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X,Y,B, rstride = 1, cstride = 2)
plt.show()


fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X,Y,A, rstride = 1, cstride = 2)
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X,Y,Z, rstride = 1, cstride = 2)
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X,Y,D, rstride = 1, cstride = 2)
plt.show()

fig = plt.figure()
plt.show()
ax = Axes3D(fig)
ax.plot_surface(X,Y,F, rstride = 1, cstride = 2)