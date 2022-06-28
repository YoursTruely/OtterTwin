
#Library import
import numpy as np
import math
import matplotlib.pyplot as plt




class VirtualUSV:

    #Initialisation
    def __init__(self):
        self.tf = 20
        self.dt = 0.001
        self.time = np.arange(0,self.tf,self.dt)

    

        self.A = np.identity(4,dtype=int)
        self.B = np.transpose(np.array([[0,0,1,0],[0,0,0,1]]))
        self.C = np.identity(4, dtype=int)
    
        self.QF = 0.01*np.identity(np.linalg.matrix_rank(self.A))  
       
        self.RF = 0.04*np.identity(np.linalg.matrix_rank(self.C))   

    
        self.x = np.transpose(np.array([[0,0,0,0]]))
        self.xhat = np.transpose(np.array([[0,0,0,0]]))
        self.theta = np.zeros((2,1))
        self.thetahat = np.zeros((2,1))
        self.Pplus = np.identity(np.linalg.matrix_rank(self.A))


        self.u = np.transpose(np.array([[1,1]]))
        self.Psi = -self.dt * np.dot(self.B,np.diag(self.u.flatten()))
        self.S = 0.1*np.identity(2, dtype = int)

        self.EpsilonPlus = np.zeros(np.shape(self.B))
        self.lambDa = 0.995
        self.a = 0.999

        self.TrueFaults = np.zeros((len(self.time),2))
        self.Inputs = np.zeros((len(self.time),2))
        self.TrueStates = np.zeros((len(self.time),4))
        

        self.FaultEstimates = np.zeros((len(self.time),2))
        self.StateEstimates = np.zeros((len(self.time),4))

        self.StateEstimatesLB = np.zeros((len(self.time),4))
        self.StateEstimatesUB = np.zeros((len(self.time),4))

    #Simulation
    def run(self):

        for i in range(len(self.time)):

            if i > int(5/self.dt): #5, 10
                self.theta = np.transpose(np.array([[0.5,0.3]]))

            if i > int(5.2/self.dt): #10, 150
                self.u = np.transpose(np.array([[0,0]]))  

            if i > int(8/self.dt): #10, 150
                self.u = np.transpose(np.array([[2,0.5]]))
                # self.theta = np.transpose(np.array([[0.1,0.5]]))

            if i > int(15/self.dt): #15, 240
                self.u = np.transpose(np.array([[-0.5,0.4]]))
                # self.theta = np.transpose(np.array([[0.1,0.3]]))



            
            # self.u = np.random.randn(2,1)

            # self.u = 0.2*np.sin(np.transpose(np.array([[10*np.pi*i/250,15*np.pi*i/500+np.pi/2]]))) + 0.3*np.cos(np.transpose(np.array([[5*np.pi*i/250,7*np.pi*i/500+np.pi/2]])))

            # self.u = np.transpose([self.Inputs[i,:]])

            

            self.Psi = -self.dt * np.dot(self.B,np.diag(self.u.flatten()))
            
            self.x =  np.dot(self.A,self.x) + self.dt * np.transpose(np.array([[self.x[2,0]*math.sin(self.x[3,0]),self.x[2,0]*math.cos(self.x[3,0]),0,0]])) + self.dt * np.dot(self.B,self.u) + np.dot(self.Psi,self.theta) + np.dot(self.QF,self.dt*np.random.randn(4,1))
            
            self.y = np.dot(self.C,self.x) + np.dot(self.RF,self.dt*np.random.randn(4,1))     
        
            self.FX = self.A + self.dt * np.array([[0,0,math.sin(self.xhat[3,0]),self.xhat[2,0]*math.cos(self.xhat[3,0])],[0,0,math.cos(self.xhat[3,0]),-self.xhat[2,0]*math.sin(self.xhat[3,0])],[0,0,0,0],[0,0,0,0]])
        
            self.Pmin = np.linalg.multi_dot([self.FX,self.Pplus,np.transpose(self.FX)]) + self.QF

            self.Sigma = np.linalg.multi_dot([self.C,self.Pmin,np.transpose(self.C)]) + self.RF

            self.KF    = np.linalg.multi_dot([self.Pmin,np.transpose(self.C),np.linalg.inv(self.Sigma)])

            self.Pplus = np.dot((np.identity(np.linalg.matrix_rank(self.A)) - np.dot(self.KF,self.C)),self.Pmin)
        
            self.yTilde = self.y - np.dot(self.C,self.xhat)

            self.QF = np.dot(self.a,self.QF) + (1-self.a)*np.linalg.multi_dot([self.KF,self.yTilde,np.transpose(self.yTilde),np.transpose(self.KF)])
    
            self.RF = np.dot(self.a,self.RF) + (1-self.a)*(np.dot(self.yTilde,np.transpose(self.yTilde)) + np.linalg.multi_dot([self.C,self.Pmin,np.transpose(self.C)]))  
        
            self.Epsilon = np.linalg.multi_dot([np.identity(np.linalg.matrix_rank(self.A))-np.dot(self.KF,self.C),self.FX,self.EpsilonPlus]) + np.dot((np.identity(np.linalg.matrix_rank(self.A))-np.dot(self.KF,self.C)),self.Psi)
            
            self.Omega = np.linalg.multi_dot([self.C,self.FX,self.EpsilonPlus])+np.dot(self.C,self.Psi)
    
            self.LambDa = np.linalg.inv(self.lambDa*self.Sigma+np.dot(np.dot(self.Omega,self.S),np.transpose(self.Omega)))

            self.Gamma = np.dot(np.dot(self.S,np.transpose(self.Omega)),self.LambDa)

            self.S = (1/self.lambDa)*self.S - (1/self.lambDa)*np.dot(np.linalg.multi_dot([np.dot(self.S,np.transpose(self.Omega)),self.LambDa,self.Omega]),self.S)

            self.EpsilonPlus = self.Epsilon

            self.thetahat = self.thetahat + np.dot(self.Gamma,self.yTilde)

            self.xhat = np.dot(self.A,self.xhat) + self.dt * np.transpose(np.array([[self.xhat[2,0]*math.sin(self.xhat[3,0]),self.xhat[2,0]*math.cos(self.xhat[3,0]),0,0]])) + self.dt * np.dot(self.B,self.u) + np.dot(self.Psi,self.thetahat) + np.dot(self.KF,self.yTilde) + np.linalg.multi_dot([self.Epsilon,self.Gamma,self.yTilde])
            
            self.TrueFaults[i,:] = self.theta.flatten()
            self.FaultEstimates[i,:] = self.thetahat.flatten()

            self.Inputs[i,:] = self.u.flatten()

            self.TrueStates[i,:] = self.x.flatten()
            self.StateEstimates[i,:] = self.xhat.flatten()

    
    def plot(self):
        self.fig = plt.figure(constrained_layout = True) 
        gs = self.fig.add_gridspec(4,5)
        ax1 = self.fig.add_subplot(gs[:4,:3])
        ax1.set_title("Trajectory")
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")

        ax2 = self.fig.add_subplot(gs[:2,-2:])
        ax2.set_ylabel("Fault 1")
        ax2.set_xlabel("Time (s)")

        ax3 = self.fig.add_subplot(gs[-2:,-2:])
        ax3.set_ylabel("Fault 2")
        ax3.set_xlabel("Time (s)")

        ax1.plot(self.TrueStates[:,0],self.TrueStates[:,1],'k--',label='Measured',linewidth=2)
        ax1.plot(self.StateEstimates[:,0],self.StateEstimates[:,1],'b-',label='Estimated',linewidth=3,alpha=0.5)
        ax1.legend()


        ax2.plot(self.time,self.TrueFaults[:,0],'k-',label = 'Fault Given')
        ax2.plot(self.time,self.FaultEstimates[:,0],'r-', label = 'Fault Estimated')
        ax21 = ax2.twinx()
        ax21.plot(self.time, self.Inputs[:,0],'b--',label='Input1',)
        ax21.set_ylabel("Input 1(a)")
        ax2.legend(loc=(1.2,0.5))
        ax21.legend(loc=(1.2,0.35))
        
        
        ax3.plot(self.time,self.TrueFaults[:,1],'k-',label='Fault Given')
        ax3.plot(self.time,self.FaultEstimates[:,1],'r-',label='Fault Estimated')
        ax31 = ax3.twinx()
        ax31.plot(self.time, self.Inputs[:,1],'b--',label='Input2')
        ax31.set_ylabel("Input 2(r)")
        ax3.legend(loc=(1.2,0.5))
        ax31.legend(loc=(1.2,0.35))

        

    def plot1(self): 
        self.fig = plt.figure(constrained_layout = True)
        
        gs = self.fig.add_gridspec(2,1)

        ax2 = self.fig.add_subplot(gs[:1,:1])
       
        ax2.set_ylabel("Fault 1(a)")
        ax2.set_xlabel("Time (s)")

        ax3 = self.fig.add_subplot(gs[1:,:1])
       
        ax3.set_ylabel("Fault 2(r)")
        ax3.set_xlabel("Time (s)")

        ax2.plot(self.time,self.TrueFaults[:,0],'k-',label='Fault',alpha=0.9)
        ax2.plot(self.time,self.FaultEstimates[:,0],'r--',label='Fault Est.')
        ax2.legend()
        ax21 = ax2.twinx()
        ax21.plot(self.time, self.Inputs[:,0],'b--',label='a')
        ax21.set_ylabel("Input 1(a)")
        ax21.legend()
       
        ax3.plot(self.time,self.TrueFaults[:,1],'k-',label='Fault',alpha=0.9)
        ax3.plot(self.time,self.FaultEstimates[:,1],'r--',label='Fault Est.')
        ax3.legend()
        ax31 = ax3.twinx()
        ax31.plot(self.time, self.Inputs[:,1],'b--',label='r')
        ax31.set_ylabel("Input 2(r)")
        ax31.legend()
        

    
if __name__ == '__main__':  
    asset = VirtualUSV()
    asset.run()
    asset.plot1()
    plt.show()
 
    


    







