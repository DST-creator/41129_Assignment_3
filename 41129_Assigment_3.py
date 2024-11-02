#%% Module imports
#General imports
import os
import scipy
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import norm
import scipy.stats as scs
import warnings

#Plotting imports
import matplotlib as mpl
import matplotlib.pyplot as plt

#%% Global plot settings

#Figure size:
mpl.rcParams['figure.figsize'] = (16, 8)  

#Lines and markers
mpl.rcParams['lines.linewidth'] = 1.2
mpl.rcParams['lines.markersize'] = 7
mpl.rcParams['scatter.marker'] = "+"
mpl.rcParams['lines.color'] = "k"
# mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k', 'k', 'k', 'k'])
#Cycle through linestyles with color black instead of different colors
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k', 'k', 'k', 'k'])\
                                + mpl.cycler('linestyle', ['-', '--', '-.', ':'])\
                                + mpl.cycler('linewidth', [1.2, 1.2, 1.3, 1.8])

#Text sizes
mpl.rcParams['font.size'] = 25
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.labelsize'] = 25
mpl.rcParams['axes.titlesize'] = 30
mpl.rcParams['legend.fontsize'] = 20

#Padding
mpl.rcParams['figure.subplot.top'] = .94    #Distance between suptitle and subplots
mpl.rcParams['xtick.major.pad'] = 5         
mpl.rcParams['ytick.major.pad'] = 5
mpl.rcParams['axes.labelpad'] = 20

#Latex font
mpl.rcParams['text.usetex'] = True          #Use standard latex font
mpl.rcParams['font.family'] = 'serif'  # LaTeX default font family
mpl.rcParams["pgf.texsystem"] = "pdflatex"  # Use pdflatex for generating PDFs
mpl.rcParams["pgf.rcfonts"] = False  # Ignore Matplotlib's default font settings
mpl.rcParams['text.latex.preamble'] = "\n".join([r'\usepackage{amsmath}',  # Optional, for math symbols
                                                 r'\usepackage{siunitx}'])
mpl.rcParams.update({"pgf.preamble": "\n".join([ # plots will use this preamble
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{amsmath}",
        r"\usepackage[detect-all,locale=DE]{siunitx}",
        ])})

#Export
mpl.rcParams['savefig.bbox'] = "tight"

#%%
exp_fld = "./00_export/"
jet_data = scipy.io.loadmat("./01_rsc/Exercise3.mat", 
                           struct_as_record=False, 
                           squeeze_me=True)["Jet"]

class JetStat:
    def __init__(self, 
                 data_path = "./01_rsc/Exercise3.mat", 
                 exp_fld = "./00_export/"):
        self.exp_fld = exp_fld
        if not os.path.isdir(self.exp_fld):
            os.mkdir(self.exp_fld)
            
        self.jet_data = scipy.io.loadmat("./01_rsc/Exercise3.mat", 
                                   struct_as_record=False, 
                                   squeeze_me=True)["Jet"]
    
    def basic_stats (self, jet_nr = 12, plt_res=False):
        """Calculate the basic statistics for a specified jet number. 
        Calculated metrics include the mean velocity, variance, skewness and 
        kurtosis
        
        Parameters:
            jet_nr (int or str):
                Number of the jet flow test (default: 12)
            plt_res (bool):
                Whether the results should be plotted (default: False)
        
        
        
        """
        try:
            jet_nr = int(jet_nr)
        except:
            raise TypeError("jet_nr must be an integer or a string "
                            "representing an integer")
        
        if jet_nr >12 or jet_nr <0:
            raise ValueError("jet_nr must be within 0<=jet_nr<=12")
        else:
            jdata = self.jet_data[jet_nr-1]
        
        #Task 1 & 2
        u = jdata.u
        t = jdata.t
        nu = jdata.nu
        
        # u_mean = np.sum(u*t)/np.sum(t)
        u_mean = np.average(u, weights=t)
        u_p = u - u_mean
        
        # u_var = np.sum(np.power(u_p,2)*t)/np.sum(t)
        u_var = np.var(u, mean=u_mean)
        
        # sigma = np.sqrt(u_var)
        sigma = np.std(u, mean=u_mean)
        
        # s_u = np.sum(np.power(u_p,3)*t)/np.sum(t)/np.power(u_var, 1.5)
        s_u = scs.skew(u)
        
        # f_u = np.sum(np.power(u_p,4)*t)/np.sum(t)/np.power(u_var, 2)
        f_u = scs.kurtosis(u)
        
        turb = u_var/u_mean
        
        if plt_res:
            #Plot u vs t
            fig, ax = plt.subplots()
            ax.plot(t,u_p,
                     zorder=2)
            
            #Formatting
            ax.set_ylabel(r'$u^\prime\:\unit{[\m/\s]}$')
            ax.set_xlabel(r'$t\:\unit{[\s]}$')
            ax.grid(zorder=1)
            
            fig.savefig(self.exp_fld+f"u_p_vs_t_jet{jet_nr}.svg")
            plt.close(fig)
        
        # Task 3    
        if plt_res:
            #Plot p.d.f.
            d_ubins = .1
            ubins=np.arange(-5,5+d_ubins,d_ubins)*u_var
            ubins_mid = (ubins+d_ubins*u_var/2)[:-1]
            counts,_ = np.histogram(a=u_p, bins=ubins)
            
            ahist = integrate.cumulative_trapezoid(counts, ubins_mid)[-1]
            counts_norm = counts/ahist
            
            #Calculate pdf
            u_p_sorted = np.sort(u_p)
            pdf = lambda u_p: 1/(sigma*np.sqrt(2*np.pi))\
                  *np.exp(-np.power(u_p,2)/(2*u_var))
            
            pdf1 = pdf(u_p_sorted)
            pdf2 = norm.pdf(u_p_sorted, np.average(u_p, weights=t), sigma)
            
            fig, ax = plt.subplots()
            ax.stairs(counts_norm, ubins)
            # ax.plot(ubins_mid, counts_norm)
            ax.plot(u_p_sorted, pdf1)
            # ax.plot(u_p_sorted, pdf2, c="r")
            
            #Formatting
            ax.set_xlabel(r'$u^\prime\:\unit{[\m/\s]}$')
            ax.set_ylabel(r'$p(u^\prime)$')
            ax.grid(zorder=1)
            
            fig.savefig(self.exp_fld+f"pdf_jet{jet_nr}.svg")
            plt.close(fig)
        
        # Task 4
        n=0
        N = len(u)
        R_e = np.array([np.average(u**2)/u_var])
        while R_e[-1]>0 and n<N:
            n+=1
            R_e = np.append(R_e, np.mean(u[1:-1-n+1]*u[n:-1])/u_var)  #Eq.4.38
        if np.min(R_e)>0:
            print("No zero crossing found")
        
        dt = t[1]-t[0]
        T_e = integrate.trapezoid(R_e, t[0:n])                      #Eq. 4.45
        tau_E = np.sqrt(2*u_var/np.mean((np.diff(u)/dt)**2))        #Eq. 4.44
        
        # Task 5
        Lambda_f = u_mean*T_e                                       #Eq. 4.55
        lambda_f = u_mean*tau_E                                     #Eq. 4.55
        #For checking the applicability of Eq. 4.55, see Eq. 4.53 & 4.54
        
        # Task 6
        epsilon = 30*nu*u_var/lambda_f**2                           #Eq. 4.83
        eta_k = np.power(nu**3/epsilon, .25)                        #Eq. 4.86
        tau_k = np.sqrt(nu/epsilon)                                 #Eq. 4.87
        
        #Task 7
        f_N = 1/(2*dt)
        df = f_N/(N/2)
        f = np.arange(0,f_N,df)
        
        u_fft = scipy.fft.fft(u)/N
        A = abs([u_fft[0], 2*u_fft[1:int(N/2)-1], u_fft[int(N/2)]])
        S = 0.5*A**2/df
        
        #Smooth Energy spectrum
        l_win = round(N/100)
        S_w, f_w = scipy.signal.welch(u, nperseg=l_win, detrend=False)
        f_w = f_w*f_N/np.pi
        S_w = S_w*np.pi/f_N
        
        #Check
        S_area = integrate.trapezoid(S, f)
        if (S_area-u_var)/u_var*100>1: #I.e. Deviation larger than 1%
            warnings.warn("Area of Energy spectrum deviates from the velocity"
                          + f" variance by {(S_area-u_var)/u_var*100:.2f} %")
        
        if plt_res:
            #Plot S vs f
            fig, ax = plt.subplots()
            ax.plot(f,S, label="Raw signal", zorder=2)
            ax.plot(f_w,S_w, label="Smoothed signal", zorder=2)
            
            #Formatting
            ax.set_xlabel(r'$f\:\unit{[\hertz]}$')
            ax.set_ylabel(r'$S\:\unit{[\m^2/\s]}$')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(zorder=1)
            
            fig.savefig(self.exp_fld+f"S_vs_f_jet{jet_nr}.svg")
            plt.close(fig)    
        
        # Task 8
        k = 2*np.pi*f/u_mean                          #Eq. from Assignement
        F = u_mean/(4*np.pi)*S                                      #Eq.4.125
        u_var_t8 = 2*integrate.trapezoid(F, f)                      #Eq.4.124
        if (u_var_t8-u_var)/u_var*100>1: #I.e. Deviation larger than 1%
            warnings.warn("Variance from wave-number spectrum doesn't match "
                          + "orginial variance. "
                          + f"Deviation: {(u_var_t8-u_var)/u_var*100*100:.2f} %")
        
        F_vk = Lambda_f*u_var/np.pi \
                * np.power(1+70.78*(k*Lambda_f/2/np.pi)**2, -5/6)   #Eq.4.1301
        
        if plt_res:
            #Plot F vs k
            fig, ax = plt.subplots()
            ax.plot(k,F, label="Original wave number spectrum", zorder=2)
            ax.plot(k,F_vk, label="Karman spectrum", zorder=2)
            
            #Formatting
            ax.set_xlabel(r'$k\:\unit{[\m^-1]}$')
            ax.set_ylabel(r'$F\:\unit{[\m^3/\s^2]}$')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(zorder=1)
            
            fig.savefig(self.exp_fld+f"S_vs_f_jet{jet_nr}.svg")
            plt.close(fig)    
        
        # Task 10
        k_macro = 1/Lambda_f
        k_micro = 1/lambda_f
        k_kolmogorov = 1/eta_k
        
        return u_mean, u_p, u_var, s_u, f_u, turb

        
        
        
if __name__ == "__main__":
    JS = JetStat()
    JS.basic_stats(plt_res = True)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        