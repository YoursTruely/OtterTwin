import numpy as np


def ssa(angle, *unit):

    if(len(unit) == 0): angle = (angle + np.pi) % (2 * np.pi) - np.pi
    elif (unit[0] == 'deg'): angle = (angle + 180) % 360 - 180
    
    return angle



def Smtrx(a):
    return np.array([[0      , -a[2,0], a[1,0]],
                     [ a[2,0], 0      , -a[0,0]],
                     [-a[1,0], a[0,0] , 0]])



def Hmtrx(a):
    return np.concatenate((np.concatenate((np.eye(3)      , np.transpose(Smtrx(a))), axis = 1),
                          np.concatenate((np.zeros((3,3)), np.eye(3)),              axis = 1)), axis = 0)





def addedMassSurge(m,L,*rho):
    # % A11 = addedMassSurge(m,L,rho) approximates the added mass in surge by
    # % the formula of Söding (1982):
    # %
    # %   A11 = -Xudot = 2.7 * rho * nabla^(5/3) / L^2
    # %
    # % Output:  
    # %   A11: added mass in surge due to a linear acceleration in surge (kg)
    # %   ratio: optional output for computation of the mass ratio A11/m,
    # %          typical values are ratios < 0.20.
    # %
    # % Inputs:   
    # %   m: ship/vehicle mass (kg)
    # %   L: ship/vehicle length (m)
    # %   rho: density of water, default value 1025 (kg/m3)
    # %
    # % Examples:
    # %   A11 = addedMassSurge(m,L)         - use default rho = 1025 (kg/m3)
    # %   A11 = addedMassSurge(m,L,rho)
    # %   [A11,ratio] = addedMassSurge(m,L,rho)
    # %
    # % Reference: H. Söding (1982). Prediction of Ship Steering Capabilities. 
    # %   Schiffstechnik, 3-29.
    # %  
    # % Author:    Thor I. Fossen
    # % Date:      17 Dec 2021
    # % Revisions: 

    if (len(rho) == 0): rho = 1025
    else: rho = rho[0]

    nabla = m / rho
    A11 = 2.7 * rho * (nabla ** (5/3)) / (L ** 2)
    ratio = A11/m

    return A11



def m2c(M,nu):
    # % C = m2c(M,nu) computes the Coriolis-centripetal matrix C(nu) from the
    # % the system inertia matrix M > 0 for varying velocity nu. 
    # % If M is a 6x6 matrix and nu = [u, v, w, p, q, r]', the output is a 6x6 C matrix
    # % If M is a 3x3 matrix and nu = [u, v, r]', the output is a 3x3 C matrix.
    # %
    # % Examples: CRB = m2c(MRB,nu)     
    # %           CA  = m2c(MA, nu)
    # % Output:
    # %  C         - Coriolis-centripetal matrix C = C(nu) 
    # %
    # % Inputs:
    # %  M        - 6x6 or 3x3 rigid-body MRB or added mass MA system marix 
    # %  nu       - nu = [u, v, w, p, q, r]' or nu = [u, v, r]'
    # %
    # % The Coriolis and centripetal matrix depends on nu1 = [u,v,w]' and nu2 =
    # % [p,q,r]' as shown in Fossen (2021, Theorem 3.2). It is possible to
    # % compute C = C(nu2) where nu2 = [p,q,r]' using the linear velocity-
    # % independent representation, see
    # %
    # % [MRB,CRB] = rbody(m,R44,R55,R66,nu2,r_bp) 
    # %
    # % Author:    Thor I. Fossen
    # % Date:      14 Jun 2001
    # % Revisions: 26 Jun 2002, M21 = M12 is corrected to M12'
    # %            10 Jan 2004, the computation of C(nU) is generalized to a 
    # %                         nonsymmetric M > 0 (experimental data)
    # %            22 Oct 2020, generalized to accept 3-DOF hirizontal-plane models
    # %            24 Apr 2021, improved the documentation


    M = 0.5 * (M + np.transpose(M))             #Symmetrization of inertia matrix

    if (np.shape(nu)[0] == 6):

        M11 = M[:3,:3]
        M12 = M[:3,3:]
        M21 = M[3:,:3]
        M22 = M[3:,3:]

        nu1 = nu[:3]
        nu2 = nu[3:]
        dt_dnu1 = np.dot(M11, nu1) + np.dot(M12, nu2)
        dt_dnu2 = np.dot(M21, nu1) + np.dot(M22, nu2)

        C = np.concatenate((np.concatenate((np.zeros((3,3))      , -Smtrx(dt_dnu1)), axis = 1),
                            np.concatenate((-Smtrx(dt_dnu1)      , -Smtrx(dt_dnu2)), axis = 1)), axis = 0)


        return C



def crossFlowDrag(L, B, T, nu_r):
    # % tau_crossflow = crossFlowDrag(L,B,T,nu_r) computes the cross-flow drag 
    # % integrals for a marine craft using strip theory. Application:
    # %
    # %  M d/dt nu_r + C(nu_r)*nu_r + D*nu_r + g(eta) = tau + tau_crossflow
    # %
    # % Inputs: L:  length
    # %         B:  beam
    # %         T:  draft 
    # %         nu_r = [u-u_c, v-v_c, w-w_c, p, q, r]': relative velocity vector
    # %
    # % Output: tau_crossflow = [0 Yh 0 0 0 Nh]:  cross-flow drag in sway and yaw
    # %
    # % Author:     Thor I. Fossen 
    # % Date:       25 Apr 2021, Horizontal-plane drag of ships
    # % Revisions:  30 Jan 2021, Extended to include heave and pitch for AUVs

    rho = 1025
    n = 20

    dx = L/n

    Cd_2D = Hoerner(B, T)   #2D - drag coefficient based on Hoerner's curve

    Yh = 0
    Zh = 0
    Mh = 0
    Nh = 0

    for xL in range(len(np.arange(-L/2, L/2, dx))):
        v_r = nu_r[1,0]             # relative sway velocities
        w_r = nu_r[2,0]
        r = nu_r[5,0]               # yaw rate
        U_h = abs(v_r + xL * r) * (v_r * xL * r)
        U_v = abs(w_r + xL * r) * (v_r * xL * r)
        Yh = Yh - 0.5 * rho * T * Cd_2D * U_h * dx          # sway force
        Zh = Zh - 0.5 * rho * T * Cd_2D * U_v * dx          # heave force
        Mh = Mh - 0.5 * rho * T * Cd_2D * xL * U_v * dx     # pitch moment
        Nh = Nh - 0.5 * rho * T * Cd_2D * xL * U_h * dx     # yaw moment
        


    return np.transpose(np.array([[0, Yh, Zh, 0, Mh, Nh]]))




def Hoerner(B,T):
    # % Hoerner computes the 2D Hoerner cross-flow form coeff. as a function of B and T.
    # % he data is digizied and interpolation is used to compute other data points 
    # % than those in the table
    # %
    # %  CY_2D = Hoerner(B,T)
    # %
    # % Output: 
    # %    CY_2D:    2D Hoerner cross-flow form coefficient
    # %
    # % Inputs:
    # %    T:      draft (m)
    # %    B:      beam  (m)
    # %
    # % Author: Thor I. Fossen
    # % Date:   2007-12-01
    # %
    # % Reference:
    # % A. J. P. Leite, J. A. P. Aranha, C. Umeda and M. B. conti (1998). 
    # % Current force in tankers and bifurcation of equilibrium of turret
    # % systems: hydrodynamic model and experiments. 
    # % Applied Ocean Research 20, pp. 145-256.

    CD_DATA = np.array([[0.0108623, 1.966080],
                        [0.1766060, 1.965730],
                        [0.3530250, 1.897560],
                        [0.4518630, 1.787180],
                        [0.4728380, 1.583740],
                        [0.4928770, 1.278620],
                        [0.4932520, 1.210820],
                        [0.5584730, 1.083560],
                        [0.6464010, 0.998631],
                        [0.8335890, 0.879590],
                        [0.9880020, 0.828415],
                        [1.3080700, 0.759941],
                        [1.6391800, 0.691442],
                        [1.8599800, 0.657076],
                        [2.3128800, 0.630693],
                        [2.5999800, 0.596186],
                        [3.0087700, 0.586846],
                        [3.4507500, 0.585909],
                        [3.7379000, 0.559877],
                        [4.0030900, 0.559315]])
    
    return np.interp(B/(2*T),CD_DATA[:,0],CD_DATA[:,1])



def eulerang(phi, theta, psi):
    # % [J,Rzyx,Tzyx] = eulerang(phi,theta,psi) computes the Euler angle
    # % transformation matrix
    # %
    # %  J = [ Rzyx     0
    # %           0  Tzyx ]
    # %
    # % where J1 = Rzyx and J2 = Tzyx, see Rzyx.m and Tzyx.m.
    # %
    # % Author:    Thor I. Fossen
    # % Date:      14th June 2001
    # % Revisions: 8 May 2021, added calls to Rzyx and Tzyx 

    J1 = Rzyx(phi, theta, psi)
    J2 = Tzyx(phi, theta)

    J = np.concatenate((np.concatenate((J1             , np.zeros((3,3))), axis = 1), 
                        np.concatenate((np.zeros((3,3)), J2             ), axis = 1)),axis = 0)

    return J



def Rzyx(phi, theta, psi):
    # % R = Rzyx(phi,theta,psi) computes the Euler angle
    # % rotation matrix R in SO(3) using the zyx convention
    # %
    # % Author:   Thor I. Fossen
    # % Date:     14th June 2001
    # % Revisions: 

    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth  = np.cos(theta)
    sth  = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
 
     
    R = np.array([[cpsi*cth, -spsi*cphi+cpsi*sth*sphi, spsi*sphi+cpsi*cphi*sth],
                  [spsi*cth, cpsi*cphi+sphi*sth*spsi , -cpsi*sphi+sth*spsi*cphi],
                  [-sth    , cth*sphi                , cth*cphi]])

    return R



def Tzyx(phi, theta):
    # % T = Tzyx(phi,theta) computes the Euler angle
    # % transformation matrix T for attitude using the zyx convention
    # %
    # % Author:   Thor I. Fossen
    # % Date:     4th August 2011
    # % Revisions: 
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth  = np.cos(theta)
    sth  = np.sin(theta)
 
    if cth == 0 : error('Tzyx is singular for theta = +-90 degrees')
    
   
    T = np.array([[1, sphi*sth/cth, cphi*sth/cth],
                  [0, cphi        , -sphi],
                  [0, sphi/cth    , cphi/cth]])

    return T


def EKF_5States(GNSS1,GNSS2,h_samp,Z,frame,Qd,Rd):
    # % EKF_5states estimates SOG, COG, and course rate from GNSS positions
    # % (xn[k], yn[k]) expressed in NED or latitude-longitude (mu[k], l[k]) using 
    # % a 5-states discrete-time extended Kalamn filter (EKF). The output is the 
    # % predicted state vector x_hat[k+1] as defined below.
    # %
    # % Outputs:
    # %  [x,y,U,chi omega_chi]  = EKF_5states(x,y,h,Z,'NED',Qd,Rd) 
    # %  [mu,l,U,chi,omega_chi] = EKF_5states(mu,l,h,Z,'LL',Qd,Rd)
    # %
    # % Inputs:
    # %  GNSS1,GNSS2: North-East positions (m) or Latitude-Longitude (rad)
    # %  h_samp:      EKF sampling time (s), h_samp = 1 / measurement frequency
    # %  Z:           h_samp * GNSS measurement frequency (Hz) (must be integer)
    # %  frame:       'NED' (North-East) or 'LL' (Latitude-Longitude)
    # %  Qd:          EKF 2x2 process cov. matrix for speed and course rate
    # %  Rd:          EKF 2x2 GNSS measurement cov. matrix
    # %
    # % Reference: S. Fossen and T. I. Fossen (2021). Five-state extended Kalman 
    # % filter for estimation of speed over ground(SOG), course over ground (COG) 
    # % and course rate of surface vehicles. Journal of Marine Science and 
    # % Applications. Submitted.
    # %
    # % Author:   Thor I. Fossen
    # % Date:     25 July 2021
    # % Revisions: 
    x_prd = None
    P_prd = None
    I5 = np.identity(5,dtype = int)

    if x_prd == None:
        x_prd = np.transpose(np.array([[GNSS1, GNSS2, 0, 0, 0]]))
        P_prd = I5
        count = 1

    a = 6378137
    f = 1/298.257223563
    e = np.sqrt(2*f - f**2)

    alpha_1 = 0.01
    alpha_2 = 0.1

    Cd = np.array([[1,0,0,0,0],[0,1,0,0,0]]) 
    Ed = h_samp * np.array([[0,0],[0,0],[1,0],[0,0],[0,1]])

    if (count == 1):
        y = np.array([[GNSS1],[GNSS2]])

        K = np.dot(np.dot(P_prd, np.transpose(Cd)),np.linalg.inv(np.linalg.multi_dot([Cd, P_prd, np.transpose(Cd)])+Rd))
        IKC = I5 - np.dot(K,Cd)
        P_hat = np.linalg.multi_dot([IKC,P_prd,np.transpose(IKC)]) + np.linalg.multi_dot([K,Rd,np.transpose(K)])
        eps =  y - np.dot(Cd,x_prd)
        if frame == 'LL':
            eps = ssa(eps,'deg')

        x_hat = x_prd + np.dot(K,eps)
        count = Z
    
    else:
        x_hat = x_prd
        P_hat = P_prd
        count = count - 1
    
    if frame == 'NED':

        f = np.array([[x_hat[2,0]*np.cos(x_hat[3,0])],
                      [x_hat[2,0]*np.sin(x_hat[3,0])],
                      [-alpha_1*x_hat[2,0]],
                      [x_hat[4,0]],
                      [-alpha_1*x_hat[4,0]]])

        Ad = I5 + h_samp * np.array([[0,0,np.cos(x_hat[3,0]),-x_hat[2,0]*np.sin(x_hat[3,0]),0],
                                     [0,0,np.sin(x_hat[3,0]),x_hat[2,0]*np.cos(x_hat[3,0]),0],
                                     [0,0,-alpha_1,0,0],
                                     [0,0,0,0,1],
                                     [0,0,0,0,-alpha_2]])

    
    if frame == 'LL':

        Rn = a/np.sqrt(1-(e**2)*(np.sin(x_hat[0,0]))**2)
        Rm = Rn * ((1-(e**2))/(1-((e**2)*((np.sin(x_hat[0,0]))**2))))

        f = np.array([[(1/Rm)*x_hat[2,0]*np.cos(x_hat[3,0])],
                      [(1/(Rn*np.cos(x_hat[0,0])))*x_hat[2,0]*np.sin(x_hat[3,0])],
                      [-alpha_1*x_hat[2,0]],
                      [x_hat[4,0]],
                      [-alpha_2*x_hat[4,0]]])

        Ad = I5 + h_samp * np.array([[0,0,(1/Rm)*np.cos(x_hat[3,0]),-(1/Rm)*x_hat[2,0]*np.sin(x_hat[3,0]),0],
                                     [np.tan(x_hat[0,0])/(Rn*np.cos(x_hat[0,0]))*x_hat[2,0]*np.sin(x_hat[3,0]),0,np.sin(x_hat[3,0]),(1/(Rn*np.cos(x_hat[0,0])))*x_hat[2,0]*np.cos(x_hat[3,0]),0],
                                     [0,0,-alpha_1,0,0],
                                     [0,0,0,0,1],
                                     [0,0,0,0,-alpha_2]])


    x_prd = x_hat + h_samp * f
    P_prd = np.linalg.multi_dot([Ad,P_hat,np.transpose(Ad)]) + np.linalg.multi_dot([Ed,Qd,np.transpose(Ed)])


    return x_hat.flatten()




def error(str):
    pass






