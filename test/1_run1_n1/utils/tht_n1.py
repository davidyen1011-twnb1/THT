import numpy as np
import sys
import math
import time

# ----------------------------------------------------------
# Triangular Hilbert Transform (THT)
#　 Test 1 --- 
# 
#   To check list
#   1. Interation range
#   [-inf, inf] x [-inf, inf] -->　[-100, 100] x [-100, 100] 
#   2. nx convergence test
#
#   3. nx needs to be odd.
#
#   4. Now sacrifice space for efficiency.
#
#   5. b > 0
# -----------------------------------------------------------

# -------------------------------------------
def tht_const(oned_array,epsilon,f,g,h,f_l3,g_l3,h_l3):
    
    # f : 2d numpy array [n_xyz, n_xyz]
    # g : 2d numpy array [n_xyz, n_xyz]
    # h : 2d numpy array [n_xyz, n_xyz]

    n_xyz = int(oned_array.shape[0])
    dxyz = float(oned_array[1] - oned_array[0])
    tht_ = 0.0 + 0.0 * 1j

    for i_x in range(n_xyz):
        for i_y in range(n_xyz):
            for i_z in range(n_xyz):
                xpypz = float(oned_array[i_x]+oned_array[i_y]+oned_array[i_z])
                if xpypz <= epsilon:
                    continue
                tht_ += f[i_x,i_y] * g[i_y,i_z] * h[i_z,i_x] *\
                        (1.0/xpypz) * dxyz * dxyz * dxyz 
    
    tht_ = tht_ / f_l3 / g_l3 / h_l3 

    return tht_
# -------------------------------------------
def L3_norm(fgh, oned_array):
    
    n_xyz = int(oned_array.shape[0])
    dxyz = float(oned_array[1] - oned_array[0])
    l3_norm = 0.0

    for i_1 in range(n_xyz):
        for i_2 in range(n_xyz):
            #l3_norm +=(np.abs(fgh[i_1, i_2])**3) * dxyz * dxyz
            l3_norm +=(1.0**3) * dxyz * dxyz

    l3_norm = l3_norm**(1.0/3.0)
    
    return l3_norm 
# -------------------------------------------
def fgh(a,b,c,x):
    
    # a : R2    2d numpy array (vector)
    # b : R     1d float number
    # c : R2    2d numpy array (vector)
    # x : R2    2d numpy array (vector)

    fgh = np.exp(2*np.pi*1j*((a*x).sum()))
    if (x[0] <= c[0] + b) and (x[0] >= c[0]-b)\
            and (x[1] <= c[1] + b) and (x[1] >= c[1] - b):
                return fgh
    else:
        return 0
# -------------------------------------------
def main():
    
    print("--------------------------------------")
    print(" THT constant calculation (one shot)  ")
    print("--------------------------------------")
    
    start_time = time.time()
    # Basic Parameters
    left_pt = -10.0
    right_pt = 10.0
    nxyz  = 201
    oned_array = np.linspace(left_pt, right_pt, nxyz)

    # Initialize h,g,f function
    f_par = {
        'a' : np.array([1.0, 3.0]),
        'b' : 5.0,
        'c' : np.array([2.0,5.0])
    }
    g_par = {
        'a' : np.array([1.5, 3.8]),
        'b' : 9.0,
        'c' : np.array([18.9,18.9])
    }
    h_par = {
        'a' : np.array([5.0, 30.8]),
        'b' : 5.0,
        'c' : np.array([1.0,-6.0])
    }
    f = np.zeros((nxyz, nxyz), dtype=np.complex_)
    g = np.zeros((nxyz, nxyz), dtype=np.complex_)
    h = np.zeros((nxyz, nxyz), dtype=np.complex_)
    
    par_time = time.time()
    print("--------------------------------------")
    print('     Set Parameter : %.4f (s)'%(par_time-start_time))
    print("--------------------------------------")
    
    for i1 in range(nxyz):
        for i2 in range(nxyz):
            f[i1,i2] =\
            fgh(f_par['a'],f_par['b'],f_par['c'],np.array([oned_array[i1],oned_array[i2]]))
            
            g[i1,i2] =\
            fgh(g_par['a'],g_par['b'],g_par['c'],np.array([oned_array[i1],oned_array[i2]]))
            
            h[i1,i2] =\
            fgh(h_par['a'],h_par['b'],h_par['c'],np.array([oned_array[i1],oned_array[i2]]))
    
    ini_time = time.time()
    print("--------------------------------------")
    print(' Initialize fgh : %.4f (s)'%(ini_time-par_time))
    print("--------------------------------------")
    
    # Calculate L3-norm
    f_l3 = L3_norm(f, oned_array)
    g_l3 = L3_norm(g, oned_array)
    h_l3 = L3_norm(h, oned_array)

    l3_time = time.time()
    print("--------------------------------------")
    print(' L3 norm : %.4f (s)'%(l3_time-ini_time))
    print("--------------------------------------")
    print("--------------------------------------")
    print(" f L3 norm = ", f_l3)
    print(" g L3 norm = ", g_l3)
    print(" h L3 norm = ", h_l3)
    print("--------------------------------------")
    
    # Calculate THT constant
    epsilon = 0.0001
    tht_= tht_const(oned_array, epsilon, f, g, h, f_l3, g_l3, h_l3)
    
    tht_time = time.time()
    print("--------------------------------------")
    print(' THT constant : %.4f (s)'%(tht_time-l3_time))
    print("--------------------------------------")
    print ("        ")
    print("--------------------------------------")
    print(" THT constant = ", tht_)
    print("--------------------------------------")

# -------------------------------------------
if __name__ == '__main__':
    main()


