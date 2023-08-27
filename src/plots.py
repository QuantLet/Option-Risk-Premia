import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def simple_3d_plot(x,y,z, save_path):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)
    ax.set_xlabel('Tau')
    ax.set_ylabel('Moneyness')
    ax.set_zlabel('Fitted IV')
    plt.xlim(0, 1)
    plt.ylim(0, 2)
    #plt.zlim(0, 4)
    plt.savefig(save_path)