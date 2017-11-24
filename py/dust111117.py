import numpy as np
import sys
sys.path.append('/Users/pela/Dust/')
import Pei
import Cardelli
import Calzetti

RV_SMC, RV_LMC, RV_MW = 2.93, 3.16, 3.08        #RV values

def calcit(fesc,ext):
  """
  Usage:
  ------
  To calculate AV given a Lya escape fraction of 30%, assuming an SMC
  extinction, use
  >>> calcit(.3,'SMC')
  ---> 0.225
  """
  tau  = -np.log(fesc)
  ALya = 2.5/np.log(10) * tau
  lam0 = .121567
  lamV = .5510
  
  if ext == 'SMC':
    RV   = 2.93
    kLya = Pei.k(lam0,'SMC') 
  elif ext == 'LMC':
    RV   = 3.16
    kLya = Pei.k(lam0,'LMC') 
  elif ext == 'MW':
    RV   = 3.08
    kLya = Pei.k(lam0,'MW') 
  elif ext == 'Cardelli':
    RV   = 3.1
    kLya = Cardelli.k(lam0)
  elif ext == 'Calzetti':
    RV   = 4.05
    kLya = Calzetti.k(lam0)
  
  AV = (ALya/kLya) * RV
  return AV
