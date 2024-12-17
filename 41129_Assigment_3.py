#%% Module imports
#General imports
import os
import scipy
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import norm
import scipy.stats as scs
from scipy.signal import welch
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
mss = {".":dict(marker = ".", s=15, c="#575757", alpha=.5),
       "+":dict(marker = "+", s=150, c="k"), 
       "d":dict(marker = "d", s=70, c="k"), 
       "1":dict(marker = "1", s=150, c="k"), 
       "v":dict(marker = "v", s=70, c="k"),
       "default": dict(marker = "+", s=100, c="k")
       }

#Text sizes
mpl.rcParams['font.size'] = 25
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.labelsize'] = 25
mpl.rcParams['axes.titlesize'] = 30
mpl.rcParams['legend.fontsize'] = 25

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

#%% Class
exp_fld = "./00_export/"
jet_data = scipy.io.loadmat("./01_rsc/Exercise3.mat", 
                           struct_as_record=False, 
                           squeeze_me=True)["Jet"]

class JetStat(object):
    def __init__(self, 
                 data_path = "./01_rsc/Exercise3.mat", 
                 exp_fld = "./00_export/"):
        self.exp_fld = exp_fld
        if not os.path.isdir(self.exp_fld):
            os.mkdir(self.exp_fld)
            
        self.jet_data = scipy.io.loadmat("./01_rsc/Exercise3.mat", 
                                   struct_as_record=False, 
                                   squeeze_me=True)["Jet"]
        
        self.res_struct = Struct()
    
    def add_jet(self, name, value):
        self.res_struct.__setattr__(name, value)
    
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
                   
        self.add_jet(jet_nr, JetStruct(jet_nr))
        
        #%%Task 1 & 2 - basic stats
        u = jdata.u
        t = jdata.t
        nu = jdata.nu
        
        u_mean = np.mean(u)
        u_p = u - u_mean
        
        u_var = np.var(u, mean=u_mean)
        
        sigma = np.std(u, mean=u_mean)
        
        s_u = scs.skew(u)
        
        f_u = scs.kurtosis(u)
        
        turb = u_var/u_mean
        
        #Save results to results struct
        self.res_struct[jet_nr].add_attrs(V=jdata.V, u=u, t=t, nu=nu, 
                                          u_mean=u_mean, u_p=u_p, u_var=u_var,
                                          sigma=sigma, s_u=s_u, f_u=f_u, 
                                          turb=turb)
        
        if plt_res:
            #Plot u vs t
            fig, ax = plt.subplots(figsize=(20,8))
            ax.plot(t, u_p, ls="-", lw=1, alpha=.8,zorder=2)
            
            #Formatting
            ax.set_ylabel(r'$u^\prime\:\unit{[\m/\s]}$', 
                          fontsize=mpl.rcParams['axes.labelsize']*1.2)
            ax.set_xlabel(r'$t\:\unit{[\s]}$', 
                          fontsize=mpl.rcParams['axes.labelsize']*1.2)
            ax.tick_params(axis='both', 
                           labelsize=mpl.rcParams['xtick.labelsize']*1.2)
            ax.set_xlim([-.05, 10.05])
            ax.grid(zorder=1)
            
            fig.savefig(self.exp_fld+f"T1_u_p_vs_t_jet{jet_nr}.svg")
            plt.close(fig)
            
            #Plot u vs t (zoomed in)
            t_window = .015         #Width of the zoomed in window
            t_1_lst = [4,6]         #Start of the zoomed in window
            for i,t_1 in enumerate(t_1_lst):
                i_low = np.argwhere(t<=t_1).flatten()[-1]
                i_high = np.argwhere(t>=t_1+t_window).flatten()[0]
                
                fig, ax = plt.subplots(figsize=(14,10))
                ax.plot(t[i_low:i_high+1]-t[i_low], 
                        u_p[i_low:i_high+1], zorder=2)
                
                
                #Formatting
                ax.set_xlim([0, t_window])
                ax.ticklabel_format(axis="x", style='scientific', 
                                       scilimits=(0, 0))
                ax.set_ylabel(r'$u^\prime\:\unit{[\m/\s]}$', 
                              fontsize=mpl.rcParams['axes.labelsize']*2)
                ax.set_xlabel(r'$\Delta t\:\unit{[\s]}$', 
                              fontsize=mpl.rcParams['axes.labelsize']*2)
                ax.tick_params(axis='both', 
                               labelsize=mpl.rcParams['xtick.labelsize']*2)
                ax.xaxis.offsetText.set_fontsize(
                        mpl.rcParams['xtick.labelsize']*2)
                ax.grid(zorder=1)
                
                fig.savefig(self.exp_fld
                            + f"T1_u_p_vs_t_zoomed_jet{jet_nr}_{i}.svg")
                plt.close(fig)

        
        #%% Task 3 - PDF 
        if plt_res:
            #Plot p.d.f.
            d_ubins = .1
            ubins=np.arange(-5,5+d_ubins,d_ubins)*u_var
            ubins_mid = (ubins+d_ubins*u_var/2)[:-1]
            counts,_ = np.histogram(a=u_p, bins=ubins, )
            
            ahist = integrate.cumulative_trapezoid(counts, ubins_mid)[-1]
            counts_norm = counts/ahist
            
            #Calculate pdf
            u_p_sorted = np.sort(u_p)
            pdf = lambda u_p: 1/(sigma*np.sqrt(2*np.pi))\
                  *np.exp(-np.power(u_p,2)/(2*u_var))               #Eq. 4.3
            
            pdf1 = pdf(u_p_sorted)
            # pdf2 = norm.pdf(u_p_sorted, np.average(u_p, weights=t), sigma)
            
            fig, ax = plt.subplots(figsize=(25,8))
            ax.stairs(counts_norm, ubins, label="Discrete distribution")
            ax.plot(u_p_sorted, pdf1, label="Normal distribution")
            
            #Formatting
            ax.set_xlabel(r'$u^\prime\:\unit{[\m/\s]}$', 
                          fontsize=mpl.rcParams['axes.labelsize']*1.3)
            ax.set_ylabel(r'$p(u^\prime)$', 
                          fontsize=mpl.rcParams['axes.labelsize']*1.3)
            ax.tick_params(axis='both', 
                           labelsize=mpl.rcParams['xtick.labelsize']*1.3)
            ax.set_xlim([-15,15])
            ax.legend()
            ax.grid(zorder=1)
            
            fig.savefig(self.exp_fld+f"T3_pdf_jet{jet_nr}.svg")
            plt.close(fig)
            
            #Save results to results struct
            self.res_struct[jet_nr].add_attrs(ubins=ubins, counts=counts)
        
        #%% Task 4 - Time correlation
        n=0
        N = len(u_p)
        R_e = np.array([np.mean(u_p**2)/u_var])
        while R_e[-1]>0 and n<N:
            n+=1
            R_e = np.append(R_e, np.mean(u_p[0:N-n]*u_p[n:N])/u_var)  #Eq.4.38
        if np.min(R_e)>0:
            print("No zero crossing found")
        
        dt = t[1]-t[0]
        T_e = integrate.trapezoid(R_e, t[0:n+1])                      #Eq. 4.45
        tau_E = np.sqrt(2*u_var/np.mean((np.diff(u_p)/dt)**2))        #Eq. 4.44
        tau_E_cont = np.sqrt(-2/np.gradient(np.gradient(R_e, t[0:n+1]), t[0:n+1]))
        
        #Save results to results struct
        self.res_struct[jet_nr].add_attrs(n=n, N=N, R_e=R_e, dt=dt, 
                                          T_e=T_e, tau_E=tau_E)
        
        if plt_res:
            #Plot R_E vs tau
            tau = dt*np.arange(0, n+1)
            fig, ax = plt.subplots(figsize=(20,8))
            ax.plot(tau,R_e, zorder=3)
            ax.axvline(tau_E, ls="--", zorder=2)
            # ax.plot(tau,1-(tau/tau_E)**2, ls="--", zorder=2)    #Micro timescale as parabola
            ax.add_patch(mpl.patches.Rectangle((0,0), T_e, 1, 
                         ls="--", ec="k", fill=False))
            ax.text(tau_E*1.5, 0.03,
                    r"$\tau_E$", 
                    color='k', va="bottom", ha="left", 
                    fontsize = mpl.rcParams['legend.fontsize'], 
                    bbox=dict(facecolor='w', alpha=0.4, ls="none"))
            ax.text(T_e*1.1, 0.03,
                    r"$T_E$", 
                    color='k', va="bottom", ha="left", 
                    fontsize = mpl.rcParams['legend.fontsize'], 
                    bbox=dict(facecolor='w', alpha=0.4, ls="none"))

            #Formatting
            ax.set_ylim([0,1.05])
            ax.set_xlabel(r'$\tau\:\unit{[\s]}$', 
                          fontsize=mpl.rcParams['axes.labelsize']*1.2)
            ax.set_ylabel(r'$R_E$', 
                          fontsize=mpl.rcParams['axes.labelsize']*1.2)
            ax.tick_params(axis='both', 
                           labelsize=mpl.rcParams['xtick.labelsize']*1.2)
            ax.grid(zorder=1)
            
            fig.savefig(self.exp_fld+f"T4_Re_vs_tau_jet{jet_nr}.svg")
            plt.close(fig)
            
            #Plot u vs t (zoomed in)
            t_1 = 1                 #Start of the zoomed in window
            t_step = (tau_E, T_e)
            t_win = 5
            lbl = (r"\tau_E", "T_E")
            for i, stp in enumerate (t_step):        #Width of the zoomed in window
                i_low = np.argwhere(t<=t_1).flatten()[-1]
                i_high = np.argwhere(t>=t_1+stp*t_win).flatten()[0]
                
                fig, ax = plt.subplots(figsize=(14,10))
                t_section = t[i_low:i_high+1]-t[i_low]
                ax.plot(t_section, 
                        u_p[i_low:i_high+1], zorder=2)
                ax.set_xlim([0, t_section[-1]])
                # ax.set_xlim([t_1, t_1+stp*t_win])
                # ax.set_ylim([np.floor(np.min(u_p[i_low:i_high+1])),
                #              np.ceil(np.max(u_p[i_low:i_high+1]))])
                ax.set_ylabel(r'$u^\prime\:\unit{[\m/\s]}$', 
                              fontsize=mpl.rcParams['axes.labelsize']*2)
                ax.set_xlabel(r'$\Delta t\:\unit{[\s]}$', 
                              fontsize=mpl.rcParams['axes.labelsize']*2)
                xax_ticks = [r"$0$", fr"${lbl[i]}$"] \
                         + [fr"${j}\cdot {lbl[i]}$" for j in range(2, t_win+1)]
                ax.set_xticks(np.linspace(t_section[0], t_section[-1], t_win+1),
                              labels = xax_ticks)
                ax.tick_params(axis='both', 
                               labelsize=mpl.rcParams['xtick.labelsize']*2)
                ax.grid(zorder=1)
                
                fig.savefig(self.exp_fld +"T4_u_p_vs_t_zoomed"
                            + "_{}".format(lbl[i].replace('\\',''))
                            + f"_jet{jet_nr}.svg")
                plt.close(fig)
        
        #%% Task 5 - Micro and Makro scales of turbulence
        Lambda_f = u_mean*T_e                                       #Eq. 4.55
        lambda_f = u_mean*tau_E                                     #Eq. 4.55
        
        #For checking the applicability of Eq. 4.55, see Eq. 4.53 & 4.54
        if sigma/u_mean>.1:       #eq. 4.53
            warnings.warn("Turbulence is too high for Eq. 4.55 "
                          + "(scales of turbulence) to be applicable "
                          + f"(sigma/u_mean={sigma/u_mean:.2f})")
        elif min(Lambda_f, lambda_f)*u_mean/nu<20:      #eq. 4.54
            warnings.warn("Reynolds number of turbulence scales is too low for"
                          + " Eq. 4.55 (scales of turbulence) to be applicable"
                          + " (min(Lambda_f, lambda_f)*u_mean/nu="
                          + f"{min(Lambda_f, lambda_f)*u_mean/nu:.d})")
        
        #Save results to results struct
        self.res_struct[jet_nr].add_attrs(Lambda_f=Lambda_f, lambda_f=lambda_f,
                                          turb_T5=sigma/u_mean,
                                          Re_lambda = min(Lambda_f, lambda_f)*u_mean/nu)
        
        #%% Task 6 - Kolmogorov
        epsilon = 30*nu*u_var/lambda_f**2                           #Eq. 4.83
        eta_k = np.power(nu**3/epsilon, .25)                        #Eq. 4.86
        tau_k = np.sqrt(nu/epsilon)                                 #Eq. 4.87
        
        #Save results to results struct
        self.res_struct[jet_nr].add_attrs(epsilon=epsilon, eta_k=eta_k, 
                                          tau_k=tau_k)
        
        #%% Task 7 - Energy density spectrum
        f_N = 1/(2*dt)
        df = f_N/(N/2)
        f = np.arange(0,f_N+df,df)
        
        u_fft = scipy.fft.fft(u_p)/N
        A = np.abs(np.concatenate((u_fft[0].reshape(-1), 
                                   2*u_fft[1:int(N/2)], 
                                   u_fft[int(N/2)].reshape(-1))))
        S = 0.5*A**2/df #Power density spectrum
        
        #Smooth Energy spectrum
        l_win = round(N/200)
        f_w, S_w = welch(u_p, nperseg=l_win, fs = 1/dt, detrend=False)
        
        #Check
        S_area = integrate.trapezoid(S, f)
        if (S_area-u_var)/u_var*100>1: #I.e. Deviation larger than 1%
            warnings.warn("Area of Energy spectrum deviates from the velocity"
                          + f" variance by {(S_area-u_var)/u_var*100:.2f} %")
        
        #Save results to results struct
        self.res_struct[jet_nr].add_attrs(f_N=f_N, df=df, f=f, u_fft=u_fft, 
                                          A=A, S=S, l_win=l_win, 
                                          f_w=f_w, S_w=S_w, S_area=S_area)
        
        if plt_res:
            #Plot S vs f
            fig, ax = plt.subplots()
            ax.scatter(f,S, label="Raw signal", **mss["."], zorder=2)
            ax.plot(f_w,S_w, label="Approximation with Welch's method", zorder=3)
            # ax.axline(xy1=(1e2, 1), slope=-5/3, ls="-.")
            # ax.text(2e3, 1e-2,
            #          "$-5/3$ slope", 
            #         color='k', va="bottom", ha="left", 
            #         rotation=-20, rotation_mode='anchor',
            #         fontsize = mpl.rcParams['legend.fontsize'], 
            #         bbox=dict(facecolor='w', alpha=0.4, ls="none"))
            
            #Formatting
            ax.set_xlabel(r'$f\:\unit{[\s^{-1}]}$')
            ax.set_ylabel(r'$S\:\left[\unit{\m^2/\s}\right]$')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim([1e-13, 1e1])
            ax.legend()
            ax.grid(zorder=1)
            
            fig.savefig(self.exp_fld+f"T7_S_vs_f_jet{jet_nr}.svg")
            # fig.savefig(self.exp_fld+f"T7_S_vs_f_jet{jet_nr}.pdf")
            fig.savefig(self.exp_fld+f"T7_S_vs_f_jet{jet_nr}.png")
            plt.close(fig)    
        
        #%% Task 8 - Wave number spectrum
        k = 2*np.pi*f/u_mean                                        #Eq. 4.55
        F = u_mean/(4*np.pi)*S                                      #Eq.4.125
        u_var_t8 = 2*integrate.trapezoid(F, k)                      #Eq.4.124
        if (u_var_t8-u_var)/u_var*100>1: #I.e. Deviation larger than 1%
            warnings.warn("Variance from wave-number spectrum doesn't match "
                          + "orginial variance. "
                          + f"Deviation: {(u_var_t8-u_var)/u_var*100*100:.2f} %")
        
        F_vk = Lambda_f*u_var/np.pi \
                * np.power(1+70.78*(k*Lambda_f/2/np.pi)**2, -5/6)   #Eq.4.131
       
        #Smoothed spectrum
        k_w = 2*np.pi*f_w/u_mean                          #Eq. from Assignement
        F_w = u_mean/(4*np.pi)*S_w                        #Eq.4.125
       
        # E_vk = 8*F_vk[0]*np.power(k*Lambda_f,4)/np.power(1+(k*Lambda_f)**2,3) #Eq. 4.130
        
        
        #Save results to results struct
        self.res_struct[jet_nr].add_attrs(k=k, F=F, F_vk=F_vk, F_w=F_w, k_w=k_w)
        
        if plt_res:
            
            #Plot F vs k
            fig, ax = plt.subplots()
            ax.scatter(k, F, label="Original wave number spectrum", 
                       **mss["."], zorder=2)
            ax.plot(k_w, F_w, label="Approximation with Welch's method", 
                    lw=1, zorder=3)
            ax.plot(k, F_vk, label="von Karman spectrum", lw=1.8, zorder=3)
            ax.axline(xy1=(1e2, 1e-1), slope=-5/3, ls="-.")
            ax.text(2e2, 5e-2,
                     "$-5/3$ slope", 
                    color='k', va="bottom", ha="left", 
                    rotation=-20, rotation_mode='anchor',
                    fontsize = mpl.rcParams['legend.fontsize'], 
                    bbox=dict(facecolor='w', alpha=0.4, ls="none"))
            
            
            #Region boundaries
            ax.axvline(4e1, ls=":", lw="1.5", c="k")
            ax.axvline(2.5e3, ls=":", lw="1.5", c="k")
            ax.axvline(max(k), ls=":", lw="1.5", c="k")
            
            #Annotations for regions
            y_text = 3e1
            arrowstyle = dict(arrowstyle="<->", 
                              connectionstyle="angle,angleA=90,angleB=0")
            # Drawing arrows for the regions
            ax.annotate("", (0,1.02), (480, 0), 
                         xycoords='axes fraction', textcoords='offset points', 
                         va='top', arrowprops = arrowstyle)
            ax.text(1e0, y_text,
                     "Energy containing range", 
                    color='k', va="bottom", ha="center", 
                    fontsize = mpl.rcParams['legend.fontsize'], 
                    bbox=dict(facecolor='w', alpha=0.4, ls="none"))

            ax.annotate("", (.535,1.02), (273, 0), 
                         xycoords='axes fraction', textcoords='offset points', 
                         va='top', arrowprops = arrowstyle)
            ax.text(2.5e2, y_text,
                     "Intertial subrange", 
                    color='k', va="bottom", ha="center", 
                    fontsize = mpl.rcParams['legend.fontsize'], 
                    bbox=dict(facecolor='w', alpha=0.4, ls="none"))
            
            ax.annotate("", (.838,1.02), (105, 0), 
                         xycoords='axes fraction', textcoords='offset points', 
                         va='top', arrowprops = arrowstyle)
            ax.text(6e3, y_text,
                     "Dissipation\n subrange", 
                    color='k', va="bottom", ha="center", 
                    fontsize = mpl.rcParams['legend.fontsize'], 
                    bbox=dict(facecolor='w', alpha=0.4, ls="none"))
            
            #Formatting
            ax.set_xlabel(r'$k\:\left[\unit{\m^{-1}}\right]$')
            ax.set_ylabel(r'$F\:\left[\unit{\m^3/\s^2}\right]$')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim([1e-13, 1e1])
            ax.legend()
            ax.grid(zorder=1)
            
            # fig.savefig(self.exp_fld+f"T8_F_vs_k_jet{jet_nr}.svg")
            # fig.savefig(self.exp_fld+f"T8_F_vs_k_jet{jet_nr}.pdf")
            fig.savefig(self.exp_fld+f"T8_F_vs_k_jet{jet_nr}.png")
            plt.close(fig) 
        
        #%% Task 10 - Characteristic wave number scales
        k_macro = 1/Lambda_f
        k_micro = 1/lambda_f
        k_kolmogorov = 1/eta_k
        
        #Save results to results struct
        self.res_struct[jet_nr].add_attrs(k_macro = k_macro, k_micro=k_micro, 
                                          k_kolmogorov = k_kolmogorov)
        
        
        return self.res_struct[jet_nr]

        
class Struct(object):
    def __init__(self, **vals): # constructor
        if vals:
            for attribute, value in vals.items():
                setattr(self, attribute, value)
        
    def __setattr__(self, name, value):
        super(Struct, self).__setattr__(str(name), value)
        
    def __getitem__(self, key):
        if hasattr(self, str(key)):
            return getattr(self, str(key))
        else:
            raise TypeError("Invalid Argument Type")

class JetStruct (Struct):
    def __init__(self, jet_nr, **vals):
        super().__init__(**vals)
        self.__dict__["nr"] = jet_nr    #Read-only property 
    
    @property
    def nr(self):
        return self.__dict__["nr"]
    
    @nr.setter
    def nr(self, value):
        raise ValueError("The attribute name '_nr' is reserved for the"
                         + " jet number")
        
    def add_attrs (self, **kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key, value)
    
#%% Main    
if __name__ == "__main__":
    JS = JetStat()
    D = .03                     #[m] - exit diameter
    stats = []
    Re = np.zeros(12)
    eta_k = np.zeros(12)
    Lambda_f = np.zeros(12)
    lambda_f = np.zeros(12)
    
    js12_stats = JS.basic_stats(jet_nr = 12,plt_res = True)
    js12_stats.add_attrs(Re=js12_stats.V*D/js12_stats.nu)
    
    
    Re[11] = js12_stats.Re
    eta_k[11] = js12_stats.eta_k
    Lambda_f[11] = js12_stats.Lambda_f
    lambda_f[11] = js12_stats.lambda_f
    
    #%%Task 11 - Comparison over all exit velocities
    Task_11 = False
    if Task_11:
        for i in range(0,11):
            stats.append(JS.basic_stats(jet_nr = i+1,plt_res = False))
            stats[i].add_attrs(Re=stats[i].V*D/stats[i].nu)
            
            Re[i] = stats[i].Re
            eta_k[i] = stats[i].eta_k
            Lambda_f[i] = stats[i].Lambda_f
            lambda_f[i] = stats[i].lambda_f
        stats.append(js12_stats)
        
        df_jets = pd.DataFrame(columns=["lambda_f", "Lambda_f", "eta_k", "Re"],
                               data=np.vstack((lambda_f, Lambda_f, eta_k, Re)).T,
                               index=np.arange(1,13))
        
        #Plot scale ratios        
        fig,ax = plt.subplots() 
        ax.scatter(Re, Lambda_f/eta_k, 
                   label=r"$\frac{\Lambda_f}{\eta_k}$", **mss["+"])
        ax.scatter(Re, Lambda_f/lambda_f, 
                   label=r"$\frac{\Lambda_f}{\lambda_f}$", **mss["d"])
        
        ax.axline(xy1=(min(Re), min(Lambda_f/eta_k)), 
                  slope=3/4, ls="--")
        ax.axline(xy1=(3e4, 6), 
                  slope=1/2, ls="--")
        
        ax.text(4e4, 8,  
                 r"$\sim Re^{1/2}$", 
                color='k', va="bottom", ha="right", 
                fontsize = mpl.rcParams['legend.fontsize'], 
                bbox=dict(facecolor='w', alpha=0.4, ls="none"))
        ax.text(4e4, 3.5e2,  
                 r"$\sim Re^{3/4}$", 
                color='k', va="bottom", ha="right", 
                fontsize = mpl.rcParams['legend.fontsize'], 
                bbox=dict(facecolor='w', alpha=0.4, ls="none"))
        
        ax.legend(fontsize = mpl.rcParams['legend.fontsize']*1.3)
        ax.set_xlabel(r"Re")
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        ax.set_xlim([np.floor(min(Re)*1e-4)*1e4, np.ceil(max(Re)*1e-4)*1e4])
        plt.tick_params(axis='x', which='minor')
        ax.xaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter("%.1e"))
        
        ax.grid(which="both", axis="x")
        ax.grid(which="major", axis="y")
        
        fig.savefig("./00_export/T11_lengthscales_vs_Re.svg")
        plt.close(fig)
        
        
        
        
#%% Energy spectra comparison        
        #Scales vs. Reynolds number
        from matplotlib.colors import LinearSegmentedColormap

        # Create a custom colormap from blue to red
        colors = ["blue", "red"]
        n_lines = 12
        custom_cmap = LinearSegmentedColormap.from_list("blue_red", colors, 
                                                        N=n_lines)
        
        fig, ax = plt.subplots()
        epsilon_mean = np.mean([stats[i].epsilon for i in range(0,11)])
        for i in range(0,11):
            if i==1:
                lbl = fr"$V={stats[i].V}" + r"\:\unit{\m/\s}\:|\:Re=" \
                    + rf"{stats[i].Re/1e4 :.1f}" + r"\times 10^4$"
            else:
                lbl = fr"$V={stats[i].V}" + r"\:\unit{\m/\s}\quad\:\!|\:Re=" \
                    + rf"{stats[i].Re/1e4 :.1f}" + r"\times 10^4$"
# =============================================================================
#             Energy Spectrum 
#             E = stats[i].k**2*np.gradient(np.gradient(stats[i].F_vk,stats[i].k),stats[i].k)- stats[i].k*np.gradient(stats[i].F_vk,stats[i].k)
#             ax.plot(stats[i].k*stats[i].eta_k,
#                     E/np.power(np.power(stats[i].nu, 5)
#                                        *stats[i].epsilon 
#                                        ,.25)
#                     )
#             
# =============================================================================

# =============================================================================
#             #von Karman power density spectrum (Wave number domain)
#             ax.plot(stats[i].k*stats[i].eta_k,
#                     np.divide(2*stats[i].F_vk, 
#                               np.power(np.power(stats[i].nu, 5)
#                                        *stats[i].epsilon,.25)),
#                     color=custom_cmap(i / (n_lines - 1)),
#                     ls="-", label=lbl)
# =============================================================================
            
            #Smoothed Power density spectrum (Wave number domain)
            ax.plot(stats[i].k_w*stats[i].eta_k,
                    np.divide(2*stats[i].F_w, 
                              np.power(np.power(stats[i].nu, 5)
                                       *epsilon_mean,.25)),
                    color=custom_cmap(i / (n_lines - 1)),
                    ls="-", label=lbl)

# =============================================================================
#             #Smoothed Power density spectrum (frequency domain)
#             ax.plot(stats[i].f_w,
#                     stats[i].S_w,
#                     color=custom_cmap(i / (n_lines - 1)),
#                     ls="-", label=lbl)
# =============================================================================
            
        #Formatting
        # ax.set_xlabel(r'$k\eta_K$')
        # ax.set_ylabel(r'$\frac{2F(k)}{\sqrt[4]{\nu^5\cdot\varepsilon}}$', 
        #               fontsize=mpl.rcParams['axes.labelsize']*1.3)
        ax.set_xlabel(r'$f\:\left[\unit{\s^{-1}}\right]$')
        ax.set_ylabel(r'$S\:\left[\unit{\m^2/\s}\right]$')
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1.05))
        ax.grid(zorder=1)
        
        fig.savefig("./00_export/T11_S_vs_f.svg")
        fig.savefig("./00_export/T11_S_vs_f.pdf")
        plt.close(fig)
        
        
    #%% Test 
# =============================================================================
#     k = js12_stats.k 
#     F_vk = js12_stats.F_vk
#     E = (k**2)*np.gradient(np.gradient(F_vk,k),k) - k*np.gradient(F_vk,k)  #Eq. 4.119
#     # E = 8*F_vk[0]*np.power(k*Lambda_f[11],4)/np.power(1+(k*Lambda_f[11])**2,3) #Eq. 4.130
#     diss = (k**2)*E
#     
#     #Plot E vs k
#     fig, ax1 = plt.subplots()
#     ax1.plot(k,E, zorder=3)
#     ax2=ax1.twinx()
#     ax2.plot(k,diss)
#     # ax.plot(k,E_vk, zorder=3)
#     # ax.plot(k,diss, zorder=3)
#     # ax.plot(k,F, zorder=3)
# 
#     #Formatting
#     ax1.set_xlabel(r'$k\:\left[\unit{\m^{-1}}\right]$')
#     ax1.set_ylabel(r'$E\:\unit{[\m^3/\s^2]}$')
#     ax2.set_ylabel(r'$k^2\cdot E\:\left[\unit{\m/\s^2}\right]$')
#     ax1.set_xscale("log")
#     ax1.grid(zorder=1)
#     
#     fig.savefig("./00_export/T8_E_vs_k_jet12.svg")
#     fig.savefig("./00_export/T8_E_vs_k_jet12.pdf")
#     plt.close(fig)
# 
#             
#             
#     for i in range(0,11):
#         print(max(stats[i].F_vk))
#         
#         
# =============================================================================
        
        
        
        
        
        
        
        
        
        