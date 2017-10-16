'''
Created on 16 October 2017

Steady State Optimization for CSTR + Distillation Column A 

@author: suwartad
'''

from casadi import *
import numpy as np

def buildModelEq(x,dist,price):
    NT    = dist["NT"]
    NF    = 21
    alpha = 1.5
    Muw   = 0.5
    taul  = 0.063
    F0    = 1
    qF0   = 1
    L0    = 2.70629
    L0b   = L0 + qF0*F0
    
    # Inputs and disturbances    
    LT  = x[2*NT+2]     # Reflux
    VB  = x[2*NT+3]     # Boilup
    F   = x[2*NT+4]     # Feedrate
    D   = x[2*NT+5]     # Distillate
    B   = x[2*NT+6]     # Bottoms
    
    F_0 = dist["F_0"]                     
    zF  = dist["zF"]    # Feed composition         
    qF  = dist["qF"]    # Feed liquid fraction
    
    # THE MODEL
    # objective function  
    J = price["pf"]*F_0 + price["pV"]*VB - price["pB"]*B - price["pD"]*D
    
    # Vapor-liquid equilibria
    y = []
    V = []
    L = []
    dMdt  = []
    dMxdt = []
    for i in range(NT-1):
        yk = MX.sym('y_' + str(i))
        y += [yk]
        
        Vk = MX.sym('V_' + str(i))
        V += [Vk]
        
        Lk = MX.sym('L_' + str(i))
        L += [Lk]

        dMdtk = MX.sym('dMdt_' + str(i))
        dMdt += [dMdtk]
        
        dMxdtk = MX.sym('dMxdt_' + str(i))
        dMxdt += [dMxdtk]
    
    L_NT          = MX.sym('L_' + str(NT-1))
    L            += [L_NT]
    dMdt_NT       = MX.sym('dMdt_' + str(NT-1))
    dMdt         += [dMdt_NT]
    dMxdt_NT      = MX.sym('dMxdt_' + str(NT-1))
    dMxdt        += [dMxdt_NT]
    dMdt_NT1      = MX.sym('dMdt_' + str(NT))
    dMdt         += [dMdt_NT1]
    dMxdt_NT1     = MX.sym('dMxdt_' + str(NT))
    dMxdt        += [dMxdt_NT1]
    
    y = vertcat(*y)
    for i in range(NT-1):
        y[i] = alpha*x[i] / (1 + (alpha-1)*x[i])
        
    # Vapor Flows assuming constant molar flows
    V = vertcat(*V)
    for i in range(NT-1):
        if i >= NT:
            V[i] = VB + (1-qF)*F
        else:
            V[i] = VB
            
    L     = vertcat(*L)
    L     = vertcat(L,LT)
    for i in range(1,NT-1):
        if i <= NF:
            L[i] = L0b + (x[NT+1+i]-Muw)/taul
        else:
            L[i] = L0  + (x[NT+1+i]-Muw)/taul
            
    # Time derivatives from  material balances for 
    # 1) total holdup and 2) component holdup
    
    # Column
    dMdt  = vertcat(*dMdt)
    dMxdt = vertcat(*dMxdt)
    
    for i in range(1,NT-1):
        dMdt[i] = L[i+1]         - L[i]       + V[i-1]         - V[i]
        dMxdt[i]= L[i+1]*x[i+1]  - L[i]*x[i]  + V[i-1]*y[i-1]  - V[i]*y[i]
        
    # Correction for feed at the feed stage
    # The feed is assumed to be mixed into the feed stage
    dMdt[NF-1] = dMdt[NF]  + F
    dMxdt[NF-1]= dMxdt[NF] + F*x[NT]
    
    # Reboiler (assumed to be an equilibrium stage)
    dMdt[0] = L[1]      - V[0]      - B
    dMxdt[0]= L[1]*x[1] - V[0]*y[0] - B*x[0]
    
    # Total condenser (no equilibrium stage)
    dMdt[NT-1] = V[NT-2]         - LT         - D
    dMxdt[NT-1]= V[NT-2]*y[NT-2] - LT*x[NT-1] - D*x[NT-1]
    
    # Compute the derivative for the mole fractions from d(Mx) = x dM + M dx
    ceq = []
    for i in range(2*NT+2):
        ceq_k  = MX.sym('ceq_' + str(i))
        ceq   += [ceq_k]
        
    # CSTR model
    k1          = 34.1/60.0
    dMdt[NT]  = F_0 + D - F
    dMxdt[NT] = F_0*zF + D*x[NT-1] - F*x[NT] - k1*x[2*NT+1]*x[NT]
    
    ceq  = vertcat(*ceq)
    for i in range(NT+1):
        ceq[i] = dMxdt[i]
        
    for i in range(NT+1):
        ceq[NT+i] = dMdt[i]
        
    # bound constraints
    lb_u = [0.1, 0.1, 0.1, 0.1, 0.1]
    ub_u = [10, 4.008, 10, 1.0, 1.0]
    
    # State bounds and initial guess
    x_min     = np.zeros(84)
    x_max     = np.ones(84)
    xB_max    = 0.1
    x_max[0]  = xB_max
    x_min[83] = 0.3
    x_max[83] = 0.7
    lbx       = vertcat(x_min,lb_u)
    ubx       = vertcat(x_max,ub_u)
    lbg       = np.zeros(2*(NT+1))
    ubg       = np.zeros(2*(NT+1))
    
    return J, ceq, lbx, ubx, lbg, ubg


if __name__ == "__main__":
    # parameter values
    NT = 41        # number of trays
    LT = 2.827     # reflux
    VB = 3.454     # boilup
    F  = 1.0       # feedrate
    zF = 1.0       # feed composition at CSTR
    D  = 0.5       # distilate flow
    B  = 0.5       # bottom flow
    qF = 1.0       # feed liquid fraction
    
    # dictionary for distillation
    dist = {"F_0":0.3}
    dist["NT"]  = NT
    dist["zF"]  = zF
    dist["qF"]  = qF
    
    # price setting
    price = {"pf":1}
    price["pV"] = 0.02
    price["pB"] = 2
    price["pD"] = 0
    
    # symbolic primitives
    x = []
    l = []
    for i in range(2*NT+2):
        xk  = MX.sym('x_' + str(i))
        x  += [xk]
        
        lk  = MX.sym('l_ + str(i)')
        l  += [lk]

    u1  = MX.sym('u1')   # LT
    u2  = MX.sym('u2')   # VB
    u3  = MX.sym('u3')   # F
    u4  = MX.sym('u4')   # D
    u5  = MX.sym('u5')   # B
    
    # concatenate states and controls 
    x   = vertcat(*x)
    x   = vertcat(x,u1,u2,u3,u4,u5)
    
    # decision variables are states and controls
    Xinit = 0.5*np.ones(84)
    Uinit = vertcat(Xinit,LT,VB,F,D,B)
    
    # define the dynamics as equality constraints and additional inequality
    # constraints (lbx, ubx, lbg, and ubg)
    obj,eq, lbx, ubx, lbg, ubg = buildModelEq(x,dist,price)
    
    # Create an NLP solver
    prob = {'f': obj, 'x': x, 'g': eq}
    solver = nlpsol('solver', 'ipopt', prob)
    
    # Solve the NLP
    sol = solver(x0=Uinit, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    x_opt = sol['x'].full().flatten()
    
    print('x_opt = ', x_opt)
    
