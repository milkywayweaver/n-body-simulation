import numpy as np
import matplotlib.pyplot as plt
import time

class NBodySim():
    def __init__(self,N=2,DIM=3,POS=None,VEL=None,M=None,G=1):
        self.N = N
        self.DIM = DIM
        if POS is None:
            self.POS = [np.random.normal(0,1,size=(self.N,self.DIM))]
        else:
            self.POS = [POS]
        if VEL is None:
            self.VEL = [np.random.normal(0,1,size=(self.N,self.DIM))]
        else:
            self.VEL = [VEL]
        if M is None:
            self.M = np.ones(N)
        else:
            self.M = M

        self.G = G
        self.has_run = False
    
    def _acceleration(self,pos):
        A = np.zeros(shape=(self.N,self.DIM))
        E_pot = 0
        for i in range(self.N):
            for j in range(i+1,self.N):
                r_diff = pos[j]-pos[i]
                r_norm = np.linalg.norm(r_diff)
                A[i] += self.G*self.M[j]*r_diff/r_norm**3
                A[j] -= self.G*self.M[i]*r_diff/r_norm**3
                E_pot -= self.G*self.M[j]*self.M[i]/r_norm
        return A,E_pot
    
    def _run_euler(self):
        init_acc = self._acceleration(self.POS[-1])
        self.ACC = [init_acc[0]]
        E_pot_i = init_acc[1]
        E_kin_i = 1/2*np.sum(self.M*np.linalg.norm(self.VEL[-1],axis=1)**2)
        self.E = [E_kin_i+E_pot_i]
        for t in range(int(self.T/self.dt)):
            vel = self.VEL[-1] + self.ACC[-1]*self.dt
            pos = self.POS[-1] + vel*self.dt
            acc,epot = self._acceleration(pos)
            ekin = 1/2*np.sum(self.M*np.linalg.norm(vel,axis=1)**2)

            self.POS.append(pos)
            self.VEL.append(vel)
            self.ACC.append(acc)
            self.E.append(ekin+epot)
    
    def _run_leapfrog(self):
        init_acc = self._acceleration(self.POS[-1])
        self.ACC = [init_acc[0]]
        E_pot_i = init_acc[1]
        E_kin_i = 1/2*np.sum(self.M*np.linalg.norm(self.VEL[-1],axis=1)**2)
        self.E = [E_kin_i+E_pot_i]

        for t in range(int(self.T/self.dt)):
            vel_half = self.VEL[-1] + self.ACC[-1]*self.dt/2
            pos = self.POS[-1] + vel_half*self.dt
            acc,epot = self._acceleration(pos)
            vel = vel_half + acc*self.dt/2

            ekin = 1/2*np.sum(self.M*np.linalg.norm(vel,axis=1)**2)

            self.POS.append(pos)
            self.VEL.append(vel)
            self.ACC.append(acc)
            self.E.append(ekin+epot)
    
    def _run_hermite(self):
        def _acc_jerk(pos,vel):
            acc = np.zeros(shape=(self.N,self.DIM))
            jerk = np.zeros(shape=(self.N,self.DIM))
            e_pot = 0
            for i in range(self.N):
                for j in range(i+1,self.N):
                    r_diff = pos[j] - pos[i]
                    r_norm = np.linalg.norm(r_diff)

                    v_diff = vel[j] - vel[i]
                    rv = np.sum(r_diff*v_diff)/r_norm**2

                    acc[i] += self.G*self.M[j]*r_diff/r_norm**3
                    acc[j] -= self.G*self.M[i]*r_diff/r_norm**3
                    jerk[i] += self.G*self.M[j]*(v_diff-3*rv*r_diff)/r_norm**3
                    jerk[j] -= self.G*self.M[i]*(v_diff-3*rv*r_diff)/r_norm**3
                    e_pot -= self.G*self.M[j]*self.M[i]/r_norm
            return acc,jerk,e_pot
        init_acc,init_jerk,init_epot = _acc_jerk(self.POS[-1],self.VEL[-1])
        init_ekin = 1/2*np.sum(self.M*np.linalg.norm(self.VEL[-1],axis=1)**2)

        self.ACC = [init_acc]
        self.JERK = [init_jerk]
        self.E = [init_epot + init_ekin]

        for t in range(int(self.T/self.dt)):
            old_pos = self.POS[-1].copy()
            old_vel = self.VEL[-1].copy()
            old_acc = self.ACC[-1].copy()
            old_jerk = self.JERK[-1].copy()
            pos = old_pos + old_vel*self.dt + 1/2*old_acc*self.dt**2 + 1/6*old_jerk*self.dt**3
            vel = old_vel + old_acc*self.dt + 1/2*old_jerk*self.dt**2

            acc,jerk,epot = _acc_jerk(pos,vel)

            vel = old_vel + 1/2*(old_acc + acc)*self.dt + 1/12*(old_jerk-jerk)*self.dt**2
            pos = old_pos + 1/2*(old_vel + vel)*self.dt + 1/12*(old_acc-acc)*self.dt**2

            ekin = 1/2*np.sum(self.M*np.linalg.norm(vel,axis=1)**2)

            self.POS.append(pos)
            self.VEL.append(vel)
            self.ACC.append(acc)
            self.JERK.append(jerk)
            self.E.append(ekin+epot)

    def run(self,T,dt,method:str='leapfrog'):
        self.has_run = True
        self.T = T
        self.dt = dt
        self.method = method

        t0 = time.time()
        if self.method == 'leapfrog':
            self._run_leapfrog()
        elif self.method == 'euler':
            self._run_euler()
        elif self.method == 'hermite':
            self._run_hermite()
        else:
            raise ValueError('Unknown method! Use "euler", "leapfrog", or "hermite"!')
        t1 = time.time()
    
        self.POS = np.array(self.POS)
        self.VEL = np.array(self.VEL)
        self.ACC = np.array(self.ACC)
        if self.method == 'hermite':
            self.JERK = np.array(self.JERK)
        self.E = np.array(self.E)
        self.running_time = t1-t0
    
    def com_convert(self):
        if self.has_run:
            M_sum = np.sum(self.M)
            COM = np.sum(self.M[:,None]*self.POS,axis=1)/M_sum
            self.POS_COM = self.POS - COM[:,None,:]
        else:
            raise ValueError('Run simulation first!')
    
    def energy_diff(self):
        return (self.E[-1]-self.E[0])/self.E[0]
    
    def plot(self,LIM=None,savepath:str=None):
        if self.has_run:
            plt.figure(figsize=(26,7))
            plt.subplot(1,4,1)
            for i in range(self.N):
                plt.plot(self.POS_COM[:,i,0], self.POS_COM[:,i,1], label=f'Star {i+1}, M = {self.M[i]:.2f}',color=f'C{i}',zorder=1)
                plt.scatter(self.POS_COM[0,i,0], self.POS_COM[0,i,1],marker='o',color=f'C{i}',s=50,zorder=3)
            plt.legend(loc='upper right')
            plt.xlabel('X (COM)')
            plt.ylabel('Y (COM)')
            if LIM is not None:
                plt.xlim(-LIM,LIM)
                plt.ylim(-LIM,LIM)
            plt.title(f'X vs Y')

            plt.subplot(1,4,2)
            for i in range(self.N):
                plt.plot(self.POS_COM[:,i,0], self.POS_COM[:,i,2], label=f'Star {i+1}, M = {self.M[i]:.2f}',color=f'C{i}',zorder=1)
                plt.scatter(self.POS_COM[0,i,0], self.POS_COM[0,i,2],marker='o',color=f'C{i}',s=50,zorder=3)
            plt.legend(loc='upper right')
            plt.xlabel('X (COM)')
            plt.ylabel('Z (COM)')
            if LIM is not None:
                plt.xlim(-LIM,LIM)
                plt.ylim(-LIM,LIM)
            plt.title(f'X vs Z')

            plt.subplot(1,4,3)
            for i in range(self.N):
                plt.plot(self.POS_COM[:,i,1], self.POS_COM[:,i,2], label=f'Star {i+1}, M = {self.M[i]:.2f}',color=f'C{i}',zorder=1)
                plt.scatter(self.POS_COM[0,i,1], self.POS_COM[0,i,2],marker='o',color=f'C{i}',s=50,zorder=3)
            plt.legend(loc='upper right')
            plt.xlabel('Y (COM)')
            plt.ylabel('Z (COM)')
            if LIM is not None:
                plt.xlim(-LIM,LIM)
                plt.ylim(-LIM,LIM)
            plt.title(f'Y vs Z')

            plt.subplot(1,4,4)
            plt.plot((self.E-self.E[0])/self.E[0])
            plt.xlabel('T')
            plt.ylabel(fr'$\Delta E_{{rel}}$')
            plt.title(fr'$\Delta E_{{rel}}$ = {(self.E[-1]-self.E[0])/self.E[0]:.5f}')

            plt.suptitle(fr'{self.N} Body Simulation (dt = {self.dt}, runtime = {self.running_time:.5f} s)',fontsize=16,y=0.98)
            plt.tight_layout()
            if savepath is not None:
                plt.savefig(savepath)
            plt.show()
        else:
            raise ValueError('Run simulation first!')
    
