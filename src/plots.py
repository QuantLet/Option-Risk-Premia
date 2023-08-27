import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def simple_3d_plot(x,y,z):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)
    plt.show()