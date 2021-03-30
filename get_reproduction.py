"""Python code for calculating the optimal reproducion points, distortion,
bound for minimum distortion for different Rs
"""

import numpy as np
import scipy.stats as st
from pdb import set_trace
import matplotlib.pyplot as plt
from sympy import Symbol, oo
from sympy.stats import Normal, E, cdf, density,P
import pandas as pd


def get_v_bounds(points):
    """Function for calculating the bounds of the Voronoi regions from the xhat
    values
    """
    v_bounds = []
    for j in range(len(points)-1):
        v_bounds.append((points[j+1]+points[j])/2)
    return v_bounds

def get_new_xhat(rvs, rvs2, v_bounds, get_distortion = False):
    """Function for calculating the updated xhat values and optionally the
    distortions for the regions.
    """
    exps = []
    exps_symb = []
    distortions = []

    for j in range(len(v_bounds)+1):
        if j == 0:
            lb = None
            lb2 = -oo
        else:
            lb = v_bounds[j-1]
            lb2 = v_bounds[j-1]
        if j == len(v_bounds):
            ub = None
            ub2 = oo
            tmp_exp = rvs.expect(lb = lb, ub = ub)/(1-rvs.cdf(lb))
        elif j == 0:
            ub = v_bounds[j]
            ub2 = v_bounds[j]
            tmp_exp = rvs.expect(lb = lb, ub = ub)/rvs.cdf(ub)
        else:
            ub = v_bounds[j]
            ub2 = v_bounds[j]
            tmp_exp = rvs.expect(lb = lb, ub = ub)/(rvs.cdf(ub)-rvs.cdf(lb))

        if i == K-1:
            tmp_exp2 =  E(rvs2,( rvs2<ub2)&(rvs2>lb2))
            exps_symb.append(tmp_exp2)
        exps.append(tmp_exp)
        if get_distortion == True:
            distortions.append(E((rvs2-float(tmp_exp2))**2,( rvs2<ub2)&(rvs2>lb2))*P((rvs2<ub2)&(rvs2>lb2)))
    if get_distortion==False:
        return exps, exps_symb
    else:
        return exps, exps_symb, distortions



K = 200 #number of iteration of the Iloyd's algorithm
sigmas = [1] #define which sigma values to run algorithm on
Rs = np.arange(2,100,2) #define which R values to run the algorithm on
all_res = pd.DataFrame() #initalize dataframe for storing results
all_xhat=[] #store xhat values for different settings and the distortions for
# each region

# the algorithm
for jj in range(len(Rs)):
    points = np.sort(np.random.rand(Rs[jj]))
    for ii in range(len(sigmas)):
        rvs = st.norm(scale = sigmas[ii])
        rvs2 = Normal('rvs2', 0, sigmas[ii])

        for i in range(K):
            v_bounds = get_v_bounds(points)
            exps, exps_symb = get_new_xhat(rvs,rvs2, v_bounds)

            points = exps

        final_v_bounds = get_v_bounds(points)
        exps, exps_symb, distortions = get_new_xhat(rvs,rvs2, final_v_bounds, get_distortion = True )
        all_xhat.append((exps,exps_symb,distortions))

        distortion = np.sum(distortions)
        D = sigmas[ii]**2*2.**(-2*Rs[jj])
        tmp_df = pd.DataFrame([[Rs[jj],sigmas[ii],float(distortion),D]], columns = ['R','sigma','distortion','D'])
        all_res = pd.concat([all_res,tmp_df])
        print(tmp_df)
#todo plots
