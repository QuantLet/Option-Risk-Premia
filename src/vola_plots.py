import datetime
import pandas as pd
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt

def trisurf(x, y, z, xlab, ylab, zlab, filename, blockplots):
    # This is the data we have: vola ~ tau + moneyness
    # https://stackoverflow.com/questions/9170838/surface-plots-in-matplotlib
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('Moneyness')
    ax.set_ylabel('Tau')
    ax.set_zlabel('IV')
    #x, y = np.meshgrid(x, y)

    df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    surf = ax.plot_trisurf(df.x, df.y, df.z, cmap=plt.cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)   
    plt.title('Empirical Vola Smile') 
    plt.savefig(filename)
    plt.draw()
    #plt.show(block = blockplots)

def vola_surface_interpolated(df, out_path = 'out/volasurface/', moneyness_min = 0.7, moneyness_max = 1.3):


    # Adjust date
    df['date_short'] = df['date'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d'))
    
    # Adjust mark iv
    df['mark_iv'] = df['mark_iv']/100

    print('Using Calls only for Vola Surface. Restricting Moneyness to ', moneyness_min, ' - ',moneyness_max)
    for d in df['date_short'].unique():
        
        sub = df.loc[df['date_short'] == d]

        fig = plt.figure(figsize=(12, 7)) 
        ax = fig.gca(projection='3d')   # set up canvas for 3D plotting

        sub = sub.loc[(sub['mark_iv'] >= 0) & (sub['mark_iv'] <= 3) & (sub['is_call'] == 1) & (sub['moneyness'] >= moneyness_min) & (sub['moneyness'] <= moneyness_max)]

        X = sub['moneyness'].tolist()
        Y = sub['tau'].tolist()
        Z = sub['mark_iv'].tolist()
        
        plotx,ploty, = np.meshgrid(np.linspace(np.min(X),np.max(X),50),\
                            np.linspace(np.min(Y),np.max(Y),50))
        
        plotz = interp.griddata((X,Y),Z,(plotx,ploty),method='linear')

        surf = ax.plot_surface(plotx,ploty,plotz,cstride=3,rstride=3,cmap=plt.cm.coolwarm, antialiased = True, linewidth = 0.5) 
        #surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, linewidth=0.5,antialiased=True)  # creates 3D plot
        ax.view_init(30, 30)
        ax.set_xlabel('Moneyness')  
        ax.set_ylabel('Time-to-Maturity')  
        ax.set_zlabel('IV')  
        ax.set_zlim(0, 3)
        
        fig.colorbar(surf)
        surf.set_clim(vmin=0, vmax = 3)

        fname = out_path + pd.to_datetime(d).strftime('%Y-%m-%d') + '.png'
        plt.savefig(fname,transparent=True)

    def plot_iv_surface(self, moneyness_min = 0.9, moneyness_max = 1.1, min_rows = 100):
        """

        """

        ivdat = self.load_update_collection(do_sample = True)

        # Prepare date string
        ivdat['date_short'] = ivdat['date'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d'))
        
        # Adjust mark iv
        ivdat['mark_iv'] = ivdat['mark_iv']/100

        # Filter
        filtered = ivdat[(ivdat['moneyness'] > moneyness_min) & (ivdat['moneyness'] < moneyness_max)]

        # Loop over dates
        unique_dates = filtered['date_short'].unique()
        for currdate in unique_dates:
            print(currdate)
            sub = filtered[(filtered['date_short'] == currdate)]

            if sub.shape[0] > min_rows:
                # Plot
                trisurf(sub['moneyness'], sub['tau'], sub['mark_iv'], 'moneyness', 'tau', 'vola', 'ivsurfaces/empirical_vola_smile_' + currdate, False)

        return None

def plot_volasmile(m2, sigma, sigma1, sigma2, mindate, plot_ident, blockplots):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    plt.plot(m2, sigma, label = 'Implied \nVolatility')
    plt.plot(m2, sigma1, label = '1st Derivative')
    plt.plot(m2, sigma2, label = '2nd Derivative')
    plt.xlabel('Moneyness')
    plt.ylabel('Implied Volatility')
    plt.title('Fitted Volatility Smile on 2020-' + str(mindate.month) + '-' + str(mindate.day))
    
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #  Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('pricingkernel/plots/fitted_vola_smile' + plot_ident)
    plt.draw()
    #plt.show(block = blockplots)


def plot_atm_iv_term_structure(df, currdate, bw, ymin = 0.5, ymax = 0.85):

    datestr = currdate.strftime("%Y-%m-%d")

    # Set namesmw
    fname = 'termstructure_reloaded/term_structure_atm_options_on_-' + datestr + str(bw) + '.png'
    titlename = 'IV Term Structure of ATM' + ' Options on '+ datestr

    #df.mark_iv = df.mark_iv / 100 # Rescale 
    call_sub = df[(df.mark_iv > 0) & (df.is_call == 1) & (df.moneyness <= 1.1) & (df.moneyness >= 0.9)]
    put_sub = df[(df.mark_iv > 0) & (df.is_call == 0) & (df.moneyness <= 1.1) & (df.moneyness >= 0.9)]
    
    # nimm einfach mal n tau ueber das man die iv plotten kann
    fig = plt.figure()
    ax = plt.subplot(111) #plt.axis()
    call_sub.groupby(["tau"])['mark_iv'].mean().plot()
    #put_sub.groupby(["tau"])['mark_iv'].mean().plot(label = 'Put IV')

    ax.set_ylabel('Implied Volatility')
    ax.set_xlabel('Tau')
    plt.title(titlename)
    plt.ylim(ymin, ymax)

    # Shrink current axis by 20%
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    #  Put a legend to the right of the current axis
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(fname, transparent = True)
    #plt.show(block = False)
    plt.draw()

    return None

def real_vola(df):
    o = df.groupby(['maturitydate_char']).std()['index_price'] * np.sqrt(252)
    return o.to_frame()

