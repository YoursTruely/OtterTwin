

#Library import
import numpy as np
import math

import pandas as pd
import matplotlib.pyplot as plt

import MSSutils as mss
from scipy import signal
from OtterModel import otter



class VirtualUSV:

    #Initialisation
    def __init__(self,data):
       
        self.data = data
        self.time = self.data['time']
       
        
        self.A = np.identity(4,dtype=int)
        self.B = np.transpose(np.array([[0,0,1,0],[0,0,0,1]]))
        self.C = np.identity(4, dtype=int)
    
        self.QF = 0.01*np.identity(np.linalg.matrix_rank(self.A))  #Process noise
       
        self.RF = 0.04*np.identity(np.linalg.matrix_rank(self.C))   #Measurement noise

        self.xhat = np.transpose(np.array([[0,0,self.data['SOG'].iloc[0],mss.ssa(self.data['COG'].iloc[0])]]))
        self.theta = np.zeros((2,1))
        self.thetahat = np.zeros((2,1))
        self.Pplus = np.identity(np.linalg.matrix_rank(self.A))


        self.u = np.transpose(np.array([[0,0]]))
        self.Psi = -np.dot(self.B,np.diag(self.u.flatten()))
        self.S = 0.1*np.identity(2, dtype = int)

        self.UpsilonPlus = np.zeros(np.shape(self.B))
        self.lambDa = 0.995
        self.a = 0.999

        self.Inputs = np.zeros((len(self.time),2))
        self.Thrust_True = np.ones((len(self.time),2))
        self.Thrust_Faulty = np.zeros((len(self.time),2))
        self.Thrust_faults = np.zeros((len(self.time),2))

        self.FaultEstimates = np.zeros((len(self.time),2))
        self.StateEstimates = np.zeros((len(self.time),4))
        self.StateEstimatesLB = np.zeros((len(self.time),4))
        self.StateEstimatesUB = np.zeros((len(self.time),4))

        self.input1_fault_max = [0,0]
        self.input2_fault_max = [0,0]
        self.thrust1_fault_max = [0,0]
        self.thrust2_fault_max = [0,0]

        self.M11 = 60.28
        self.M66 = 45.265
        self.b = 0.395
        self.M = np.array([[self.M11,0],[0,self.M66]])
        
       
       

        

    #Simulation
    def run(self):
    
        for i in range(len(self.time)):

            #Observations      
            self.y = np.dot(self.C,np.transpose(np.array([[self.data['x'].iloc[i],self.data['y'].iloc[i],self.data['U'].iloc[i],self.data['theta'].iloc[i]]])))           
           
            #Input
            self.u = np.transpose(np.array([[self.data['a_f'].iloc[i],self.data['r_f'].iloc[i]]]))


            #Sample time
            if np.isnan(self.data['h'].iloc[i]): self.dt = 0.2
            else: self.dt = self.data['h'].iloc[i]

            #Fault Matrix
            self.Psi = -self.dt * np.dot(self.B,np.diag(self.u.flatten()))
      
            #Jacobian
            self.FX = self.A + self.dt * np.array([[0,0,math.sin(self.xhat[3,0]),self.xhat[2,0]*math.cos(self.xhat[3,0])],[0,0,math.cos(self.xhat[3,0]),-self.xhat[2,0]*math.sin(self.xhat[3,0])],[0,0,0,0],[0,0,0,0]])
         
            #Extended Kalman Filter
            self.Pmin = np.linalg.multi_dot([self.FX,self.Pplus,np.transpose(self.FX)]) + self.QF
            self.Sigma = np.linalg.multi_dot([self.C,self.Pmin,np.transpose(self.C)]) + self.RF
            self.KF    = np.linalg.multi_dot([self.Pmin,np.transpose(self.C),np.linalg.inv(self.Sigma)])
            self.Pplus = np.dot((np.identity(np.linalg.matrix_rank(self.A)) - np.dot(self.KF,self.C)),self.Pmin)

            self.yTilde = self.y - np.dot(self.C,self.xhat)
            self.QF = np.dot(self.a,self.QF) + (1-self.a)*np.linalg.multi_dot([self.KF,self.yTilde,np.transpose(self.yTilde),np.transpose(self.KF)])
            self.RF = np.dot(self.a,self.RF) + (1-self.a)*(np.dot(self.yTilde,np.transpose(self.yTilde)) + np.linalg.multi_dot([self.C,self.Pmin,np.transpose(self.C)])) 
         
            #Adaptive Kalman Filter
            self.Upsilon = np.linalg.multi_dot([np.identity(np.linalg.matrix_rank(self.A))-np.dot(self.KF,self.C),self.FX,self.UpsilonPlus]) + np.dot((np.identity(np.linalg.matrix_rank(self.A))-np.dot(self.KF,self.C)),self.Psi)
            self.Omega = np.linalg.multi_dot([self.C,self.FX,self.UpsilonPlus])+np.dot(self.C,self.Psi)
            self.LambDa = np.linalg.inv(self.lambDa*self.Sigma+np.dot(np.dot(self.Omega,self.S),np.transpose(self.Omega)))
            self.Gamma = np.dot(np.dot(self.S,np.transpose(self.Omega)),self.LambDa)
            self.S = (1/self.lambDa)*self.S - (1/self.lambDa)*np.dot(np.linalg.multi_dot([np.dot(self.S,np.transpose(self.Omega)),self.LambDa,self.Omega]),self.S)
            self.thetahat = self.thetahat + np.dot(self.Gamma,self.yTilde)
            self.xhat = np.dot(self.A,self.xhat) + self.dt * np.transpose(np.array([[self.xhat[2,0]*math.sin(self.xhat[3,0]),self.xhat[2,0]*math.cos(self.xhat[3,0]),0,0]])) + self.dt * np.dot(self.B,self.u) + np.dot(self.Psi,self.thetahat) + np.dot(self.KF,self.yTilde) + np.linalg.multi_dot([self.Upsilon,self.Gamma,self.yTilde])
            self.UpsilonPlus = self.Upsilon


            #Output vectors
            self.Inputs[i,:] = [self.data['a_f'].iloc[i],self.data['r_dot_f'].iloc[i]]
            self.FaultEstimates[i,:] = self.thetahat.flatten()
            self.StateEstimates[i,:] = self.xhat.flatten()
            
            #Thrust
            self.Thrust_True[i,:] = (np.linalg.multi_dot([np.linalg.inv(np.array([[1,self.b],[1,-self.b]])),self.M,np.transpose(np.array([self.Inputs[i,:]]))])).flatten() - np.dot(np.linalg.inv(np.array([[1,self.b],[1,-self.b]])),self.data['forces'].iloc[i])
            self.Thrust_Faulty[i,:] = (np.linalg.multi_dot([np.linalg.inv(np.array([[1,self.b],[1,-self.b]])),self.M,np.array([[1-self.FaultEstimates[i,0],0],[0,1-self.FaultEstimates[i,1]]]),np.transpose(np.array([self.Inputs[i,:]]))])).flatten() - np.dot(np.linalg.inv(np.array([[1,self.b],[1,-self.b]])),self.data['forces'].iloc[i])

            #Thrust faults coefficients
            self.Thrust_faults[i,:] = 1 - self.Thrust_Faulty[i,:]/self.Thrust_True[i,:]
            self.Thrust_faults[i,:] = np.where(self.Thrust_faults[i,:]<0,0,self.Thrust_faults[i,:])
            self.Thrust_faults[i,:] = np.where(self.Thrust_faults[i,:]>1,1,self.Thrust_faults[i,:])


    #Input faults
    def if_plot(self):

        self.fig = plt.figure(constrained_layout = True)
        gs = self.fig.add_gridspec(4,5)
        ax1 = self.fig.add_subplot(gs[:4,:-2])
        ax1.set_title("Tragejtory")
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")

        ax2 = self.fig.add_subplot(gs[:2,-2:])
        ax2.set_ylabel("Fault 1(a)")
        ax2.set_xlabel("Time (s)")

        ax3 = self.fig.add_subplot(gs[-2:,-2:])
        ax3.set_ylabel("Fault 2(r)")
        ax3.set_xlabel("Time (s)")

        ax1.plot(self.data['x'],self.data['y'],'k--',label='Measured',linewidth=2)
        ax1.plot(self.StateEstimates[:,0],self.StateEstimates[:,1],'b-',label='Estimated',linewidth=3,alpha=0.5)
        ax1.legend()

    
        ax2.plot(self.time,self.FaultEstimates[:,0],'r-',label='Fault(a)')
        ax2.legend(loc='upper center',bbox_to_anchor=(0.2,-0.1))
        label = '{:.2f}'.format(self.input1_fault_max[1])
        ax2.annotate(label,self.input1_fault_max,textcoords='offset points',xytext=(0,1),ha='center')
        
        ax21 = ax2.twinx()
        ax21.plot(self.time, self.data['a_f'],'b--',alpha=0.5,label='Input(a)')
        ax21.set_ylabel("Input(a)")
        ax21.legend(loc='upper center',bbox_to_anchor=(0.8,-0.1))
        

        ax3.plot(self.time,self.FaultEstimates[:,1],'r-',label='Fault(r)')
        ax3.legend(loc='upper center',bbox_to_anchor=(0.2,-0.1))
        label = '{:.2f}'.format(self.input2_fault_max[1])
        ax3.annotate(label,self.input2_fault_max,textcoords='offset points',xytext=(0,1),ha='center')

        ax31 = ax3.twinx()
        ax31.plot(self.time, self.data['r_f'],'b--',alpha=0.5,label='Input(r)')
        ax31.set_ylabel("Input(r)")
        ax31.legend(loc='upper center',bbox_to_anchor=(0.8,-0.1))

        
    #Actuator faults
    def af_plot(self):

        self.fig = plt.figure(constrained_layout = True)
        
        gs = self.fig.add_gridspec(2,1)

        ax2 = self.fig.add_subplot(gs[:1,:1])
       
        ax2.set_ylabel("STB Thruster fault")
        ax2.set_xlabel("Time (s)")

        ax3 = self.fig.add_subplot(gs[1:,:1])
       
        ax3.set_ylabel("Port Thruster fault")
        ax3.set_xlabel("Time (s)")

        ax2.plot(self.time,self.Thrust_faults[:,0],'r-')
        label = '{:.2f}'.format(self.thrust1_fault_max[1])
        ax2.annotate(label,self.thrust1_fault_max,textcoords='offset points',xytext=(0,1),ha='center')
        
        ax3.plot(self.time,self.Thrust_faults[:,1],'r-')
        label = '{:.2f}'.format(self.thrust2_fault_max[1])
        ax3.annotate(label,self.thrust2_fault_max,textcoords='offset points',xytext=(0,1),ha='center')
        

    def FD(self):

        self.input1_fault_max[0] = self.time.iloc[np.argmax(self.FaultEstimates[:,0],axis=0)]
        self.input1_fault_max[1] = np.max(self.FaultEstimates[:,0],axis=0)

        self.input2_fault_max[0] = self.time.iloc[np.argmax(self.FaultEstimates[:,1],axis=0)]
        self.input2_fault_max[1] = np.max(self.FaultEstimates[:,1],axis=0)

        self.thrust1_fault_max[0] = self.time.iloc[np.argmax(self.Thrust_faults[:,0],axis=0)]
        self.thrust1_fault_max[1] = np.max(self.Thrust_faults[:,0],axis=0)

        self.thrust2_fault_max[0] = self.time.iloc[np.argmax(self.Thrust_faults[:,1],axis=0)]
        self.thrust2_fault_max[1] = np.max(self.Thrust_faults[:,1],axis=0)
    

#Data processing
def processData(data):
    #Lat long to X Y
    l = data['Long']
    mu = data['Lat'] 

    e = 0.0818
    a = 6378137

    Rn = a/math.sqrt(1-(e**2)*(math.sin(mu.iloc[0])**2))
    Rm = Rn - (1-e**2)/(math.sqrt(1-(e**2)*(math.sin(mu.iloc[0])**2)))

    data['time'] = data['TimeStamp']-data['TimeStamp'].iloc[0]
    data['x'] = (l-l.iloc[0])/math.atan2(1,Rn*math.cos(mu.iloc[0]))
    data['y'] = (mu-mu.iloc[0])/math.atan2(1,Rm)
    data['U'] = data['SOG']
    data['theta'] = mss.ssa(data['COG'])
   
    #Low-pass Fossen:
    T_a = 10
    T_r = 10
    a_max = 4
    a_min = -4
    r_max = 1.5
    r_min = -1.5

    data['h'] = data['TimeStamp'].diff()
    data['a_c'] = (data['U'].shift(-1) - data['U'])/data['h']
    data['r_c'] = (data['theta'].shift(-1) - data['theta'])/data['h']
    data['r_dot'] = (data['r_c'].shift(-1) - data['r_c'])/data['h']
    
    
    mp = 0
    rp = np.transpose(np.array([[0,0,0]]))
    V_c = 0
    beta_c = 0
    
    data = data.fillna(0)

    #Butter-worth:
 
    fs = 5
    cutoff_a = 1 #1.2 works
    cutoff_r = 1
    cutoff_rd = 0.9
    order = 5
    

    data['a_f'] = butter_lowpass_filter(data['a_c'],cutoff_a,fs,order)
    data['r_f'] = butter_lowpass_filter(data['r_c'],cutoff_r,fs,order)
    data['r_dot_f'] = butter_lowpass_filter(data['r_dot'],cutoff_rd,fs,order)

    
    

    data['state'] = data.apply(lambda row:np.transpose(np.array([[row.SOG,0,0,0,0,row.r_f,row.x,row.y,0,0,0,row.COG]])), axis = 1)
    data['rpm'] = data.apply(lambda row:np.transpose(np.array([[row.RPM_STB,row.RPM_PORT]])),axis = 1)

    data['forces'] = data.apply(lambda row:otter(row.state,row.rpm,mp,rp,V_c,beta_c),axis = 1)
 
    data = data.fillna(0)
    

    return data


def butter_lowpass_filter(data,cutoff,fs,order):
    normal_cutoff = cutoff/(0.5*fs)
    b,a = signal.butter(order,normal_cutoff,btype='low',analog = False)
    # print(a)
    y = signal.lfilter(b,a,data)
    return y


def sat(data,max,min):
    sat_data = np.where(data>max,max,data)
    sat_data = np.where(data<min,min,data)
    return sat_data
    

    



if __name__ == '__main__':  
        
    
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRSip2FMYM2u6EQxZpyw26RtLllLNIN4fv6_rEZUtlMC-c2QzAdiLtND7VzYDaqvnGz8q-dcUvre2qg/pub?gid=663831922&single=true&output=csv"
    data= pd.read_csv(url)
    data['RPM_STB'] = pd.to_numeric(data['RPM_STB'],errors = 'coerce')

    data['COG'] = data['COG'].interpolate()
    data['SOG'] = data['SOG'].interpolate()
    data['RPM_STB'] = data['RPM_STB'].interpolate()
    data['RPM_PORT'] = data['RPM_PORT'].interpolate()

    data.astype({'RPM_STB':'float32','RPM_PORT':'float32'}).dtypes

    for i in range(1,9):
        dataSet = data[(data['TEST'] == i) & (data['TEST1'] == i*10)].copy() 
        data_new = processData(dataSet)
        asset = VirtualUSV(data_new)
        asset.run()
        asset.FD()
        asset.af_plot()
        asset.if_plot()
        plt.show()







  
 
    


    







