import numpy as np
import MSSutils as mss

def otter(x,n,mp,rp,V_c,beta_c):
    # [xdot,U] = otter(x,n,mp,rp,V_c,beta_c) returns the speed U in m/s (optionally) 
    # % and the time derivative of the state vector: 
    # %    x = [ u v w p q r x y z phi theta psi ]' 
    # % for the Maritime Robotics Otter USV, see www.maritimerobotics.com. 
    # % The length of the USV is L = 2.0 m, while the state vector is defined as:
    # %
    # %  u:     surge velocity          (m/s)
    # %  v:     sway velocity           (m/s)
    # %  w:     heave velocity          (m/s)
    # %  p:     roll velocity           (rad/s)
    # %  q:     pitch velocity          (rad/s)
    # %  r:     yaw velocity            (rad/s)
    # %  x:     position in x direction (m)
    # %  y:     position in y direction (m)
    # %  z:     position in z direction (m)
    # %  phi:   roll angle              (rad)
    # %  theta: pitch angle             (rad)
    # %  psi:   yaw angle               (rad)
    # %
    # % The other inputs are:
    # %
    # % n = [ n(1) n(2) ]' where
    # %    n(1): propeller shaft speed, left (rad/s)
    # %    n(2): propeller shaft speed, right (rad/s)
    # %
    # % mp = payload mass (kg), maximum 45 kg
    # %  rp = [xp, yp, zp]' (m) is the location of the payload w.r.t. the CO
    # %  V_c:     current speed (m/s)
    # %  beta_c:  current direction (rad)
    # %
    # % See, ExOtter.m and demoOtterUSVHeadingControl.slx
    # %
    # % Author:    Thor I. Fossen
    # % Date:      2019-07-17
    # % Revisions: 2021-04-25 added call to new function crossFlowDrag.m
    # %            2021-06-21 Munk moment in yaw is neglected
    # %            2021-07-22 Added a new state for the trim moment
    # %            2021-12-17 New method Xudot = -addedMassSurge(m,L,rho)


    trim_moment = 0
    trim_setpoint = 280

    #Otter - Main data
    g = 9.81
    rho = 1025
    L = 2.0
    B = 1.08
    m = 55.0
    rg = np.transpose(np.array([[0.2,0,-0.2]]))
    r44 = 0.4 * B
    r55 = 0.25 * L
    r66 = 0.25 * L
    T_yaw = 1
    U_max = 6 * 0.5144

    #Otter - Data for one pontoon
    B_pont = 0.25
    y_pont = 0.395
    Cw_pont = 0.75
    Cb_pont = 0.4

    if(x.shape[0] != 12): error("x vector must have dimension 12")
    if(n.shape[0] != 2): error('n vector must have dimension 2')

    # State and Current variables
    nu = x[:6]
    nu1 = x[:3]
    nu2 = x[3:6]
    eta = x[6:]

    U = np.sqrt(nu[0,0]**2 + nu[1,0]**2 + nu[2,0]**2)
    u_c = V_c * np.cos(beta_c - eta[5,0])
    v_c = V_c * np.sin(beta_c - eta[5,0])
    nu_r = nu - np.transpose(np.array([[u_c, v_c, 0, 0, 0, 0]]))

    # Inertia dyadic, volume displacement and draft
    nabla = (m + mp)/rho       #Volume
    T = nabla/(2 * Cb_pont * B_pont * L)    #Draft
    Ig_CG = m * np.diag(np.array([r44**2 , r55**2 , r66**2]))
    rg = (m * rg + m * rp)/(m + mp)
    Ig = Ig_CG - m * np.dot(mss.Smtrx(rg),mss.Smtrx(rg)) - mp * np.dot(mss.Smtrx(rp),mss.Smtrx(rp))


    # Experimental propeller data including lever arms
    l1 = -y_pont 
    l2 = y_pont
    k_pos = 0.02216/2
    k_neg = 0.01289/2
    n_max = np.sqrt((0.5 * 24.4 * g)/k_pos)
    n_min = -np.sqrt((0.5 * 13.6 * g)/k_neg)


    # MRG and CRB (Fossen 2021)
    I3 = np.eye(3, dtype = int)
    O3 = np.zeros((3,3),dtype = int)

    # MRB_CG = np.array([[(m + mp) * I3, O3],
    #                    [   O3        , Ig]])

    MRB_CG = np.concatenate((np.concatenate(((m+mp)*I3,O3),axis=1),np.concatenate((O3,Ig),axis=1)),axis=0)

    # CRB_CG = np.array([[(m + mp) * mss.Smtrx(nu2), O3],
    #                    [   O3        , -mss.Smtrx(Ig*nu2)]])

    CRB_CG = np.concatenate((np.concatenate(((m+mp)*mss.Smtrx(nu2),O3),axis=1),np.concatenate((O3,-mss.Smtrx(Ig*nu2)),axis=1)),axis=0)


    H = mss.Hmtrx(rg)    # Transform MRB and CRB from CG to CO
    
    MRB = np.linalg.multi_dot([np.transpose(H), MRB_CG, H])
    CRB = np.linalg.multi_dot([np.transpose(H), CRB_CG, H])

    


    # Hydrodynamic added mass
    Xudot = -mss.addedMassSurge(m,L,rho)
    Yvdot = -1.5 * m
    Zwdot = -1.0 * m
    Kpdot = -0.2 * Ig[0,0]
    Mqdot = -0.8 * Ig[1,1]
    Nrdot = -1.7 * Ig[2,2]


    MA = -np.diag([Xudot, Yvdot, Zwdot, Kpdot, Mqdot, Nrdot])
    CA = mss.m2c(MA, nu_r)                                     #Coriolis-centripetal matrix

    CA[5,0] = 0             # Assuming munk moment in yaw can be neglected
    CA[5,1] = 0             # These terms, if non-zero, must be balanced by adding nonlinear damping


    # System mass and coriolis centriperal matrices
    M = MRB + MA
    C = CRB + MA


    # Hydrostatic quantities
    Aw_pont = Cw_pont * L * B_pont
    I_T = 2 * (1/12) * L * (B_pont ** 3) * (6 * (Cw_pont ** 3)/((1 + Cw_pont)*(1 + 2 * Cw_pont))) + 2 * Aw_pont * (y_pont ** 2)
    I_L = 0.8 * 2 * (1/12) * B_pont * (L ** 3)
    KB = (1/3) * (5 * T/2 - 0.5 * nabla/(L * B_pont))
    BM_T = I_T / nabla
    BM_L = I_L / nabla
    KM_T = KB + BM_T
    KM_L = KB + BM_L
    KG = T - rg[2,0]
    GM_T = KM_T - KG
    GM_L = KM_L - KG

    G33 = rho * g * (2 * Aw_pont)     # Spring stiffness
    G44 = rho * g * nabla * GM_T
    G55 = rho * g * nabla * GM_L

    G_CF = np.diag([0, 0, G33, G44, G55, 0])                # Spring stiffness matrix in the CF
    LCF = -0.2
    H = mss.Hmtrx(np.transpose(np.array([[LCF, 0, 0]])))
    G = np.linalg.multi_dot([np.transpose(H),G_CF, H])


    # Natural frequencies
    w3 = np.sqrt(G33/M[2,2])
    w4 = np.sqrt(G44/M[3,3])
    w5 = np.sqrt(G55/M[4,4])


    # Linear damping terms (Hydrodynamic derivatives)
    Xu = -24.4 * g / U_max
    Yv = 0
    Zw = -2 * 0.3 * w3 * M[2,2]
    Kp = -2 * 0.2 * w4 * M[3,3]
    Mq = -2 * 0.4 * w5 * M[4,4]
    Nr = -M[5,5] / T_yaw


    # Control forces and moments - with propeller revolution saturation
    Thrust = np.zeros((2,1))

    for i in [0,1]:
        if n[i,0] > n_max: n[i,0] = n_max
        elif n[i,0] < n_min: n[i,0] = n_min

        if n[i,0] > 0: Thrust[i,0] = k_pos * n[i,0] * abs(n[i,0])
        else: Thrust[i,0] = k_neg * n[i,0] * abs(n[i,0])


    tau = np.transpose(np.array([[Thrust[0,0] + Thrust[1,0], 0, 0, 0, 0, -l1 * Thrust[0,0] - l2 * Thrust[1,0]]]))

    # Linear damping using relative velocities + nonlinear yaw damping
    Xh = Xu * nu_r[0,0]
    Yh = Yv * nu_r[1,0]
    Zh = Zw * nu_r[2,0]
    Kh = Kp * nu_r[3,0]
    Mh = Mq * nu_r[4,0]
    Nh = Nr * (1 + 10 * abs(nu_r[5,0])) * nu_r[5,0]

    tau_damp = np.transpose(np.array([[Xh, Yh, Zh, Kh, Mh, Nh]]))


    # Strip theory cross-flow drag integrals for Yh and Nh 
    tau_crossflow = mss.crossFlowDrag(L, B_pont, T, nu_r)

    # Ballast 
    g_0 = np.transpose(np.array([[0, 0, 0, 0, trim_moment, 0]]))


    # Kinematics
    J = mss.eulerang(eta[3,0], eta[4,0], eta[5,0])

    # print(Xu,Yv,Nr)
    
    xdot = np.concatenate((np.dot(np.linalg.inv(M),(tau + tau_damp + tau_crossflow - np.dot(C, nu_r) - np.dot(G, eta) - g_0)),np.dot(J, nu)),axis = 0)

    trim_moment = trim_moment + 0.05 * (trim_setpoint - trim_moment)

    return np.take(tau_damp + tau_crossflow - np.dot(C, nu_r) - np.dot(G, eta) - g_0,[0,5])

def error(error):
    print(error)



# x = np.transpose(np.array([[0,0,0,0,0,0,0,0,0,0,0,0]]))
# n = np.transpose(np.array([[100,100]]))
# mp = 0
# rp = np.transpose(np.array([[0,0,0]]))
# V_c = 0
# beta_c = 0

# otter(x,n,mp,rp,V_c,beta_c)