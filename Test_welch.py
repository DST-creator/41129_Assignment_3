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

mss = {".":dict(marker = ".", s=15, c="#575757", alpha=.5),
       ".2":dict(marker = ".", s=15, c="#D61921", alpha=.3),
       "+":dict(marker = "+", s=150, c="k"), 
       "d":dict(marker = "d", s=70, c="k"), 
       "1":dict(marker = "1", s=150, c="k"), 
       "v":dict(marker = "v", s=70, c="k"),
       "default": dict(marker = "+", s=100, c="k")
       }

exp_fld = "./02_test/"
data = scipy.io.loadmat("./01_rsc/matlab_data.mat", 
                           struct_as_record=False, 
                           squeeze_me=True)["exp_struct"]

#Prep
dt = 0.001
u_var = 0.004
N = 1e6
u_p = data.u_p
t = data.u_p

f_N_mat = data.f_N
df_mat = data.df
f_mat = data.f
l_win_mat = data.l_win
u_fft_mat = data.u_fft
A_mat = data.A
S_mat = data.S
S_w_mat = data.S_w
S_w_raw_mat = data.S_w_raw
f_w_mat = data.f_w
f_w_raw_mat = data.f_w_raw


#Calc
f_N = 1/(2*dt)
df = f_N/(N/2)
f = np.arange(0,f_N+df,df)

u_fft = scipy.fft.fft(u_p)/N
A = np.abs(np.concatenate((u_fft[0].reshape(-1), 
                           2*u_fft[1:int(N/2)], 
                           u_fft[int(N/2)].reshape(-1))))
S = 0.5*A**2/df

#Smooth Energy spectrum
l_win = round(N/100)

# f_w, S_w = welch(u_p, window = win, nperseg=l_win, detrend=False)
f_w_raw, S_w_raw = welch(u_p, nperseg=l_win, fs = 1/dt, detrend=False)
f_w = f_w_raw*f_N/np.pi
S_w = S_w_raw*np.pi/f_N


#Plot
fig, ax = plt.subplots()
ax.plot(f_w_raw, S_w_raw, label="Python")
ax.plot(f_w_raw_mat, S_w_raw_mat, label="Matlab", ls="--")
ax.plot(f_w_mat, S_w_mat, label="Matlab", ls="-.")

#Formatting
ax.set_xlabel(r't')
ax.set_ylabel(r'S')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.grid(zorder=1)

fig.savefig(exp_fld+f"S_w_raw_vs_f_test.pdf")
plt.close(fig)



# fig, ax = plt.subplots()
# # ax.scatter(f,S, label="Raw signal", **mss["."], zorder=2)
# # ax.scatter(data.f,data.S, label="Raw signal - Matlab", **mss[".2"], zorder=3)
# ax.plot(f_w,S_w, label="Smoothed signal", zorder=4)
# ax.plot(data.f_w, data.S_w, label="Smoothed signal - Matlab", ls="--", zorder=4)

# #Formatting
# ax.set_xlabel(r'f')
# ax.set_ylabel(r'S')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.legend()
# ax.grid(zorder=1)

# fig.savefig(exp_fld+f"S_vs_f_test.pdf")
# plt.close(fig)


#Slope -5/3
k_1 = -100.2218487
s_slope = lambda f: np.exp(-5/3*np.exp(f+k_1))
f_range_slope = np.linspace(1e2, 1e4, 50)

#Plot S vs f
fig, ax = plt.subplots()
ax.plot(f_w_mat, S_w_mat, label="Matlab", ls="-.")
plt.axline(xy1=(1, 1e-2), slope=-5/3)

#Formatting
ax.set_xlabel(r't')
ax.set_ylabel(r'S')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.grid(zorder=1)

fig.savefig(exp_fld+f"S_vs_f_slope.pdf")
plt.close(fig)
