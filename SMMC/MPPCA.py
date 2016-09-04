import numpy as np
from .Kmeans import kmeans

class MPPCA():
    
    def __init__(self,data,d,M,kmeans_init = False):
        self.data = data
        self.N = data.shape[0]
        self.D = data.shape[1]
        self.d = d
        self.M = M
        self.L_old = None
        self.random_init()
        if kmeans_init:
            self.kmeans_init()
            
    def random_init(self):
        self.V = np.random.randn(self.M,self.D,self.d)
        self.U = np.random.randn(self.M,self.D,1)
        self.o2 = np.random.rand(self.M)
        self.pi = np.ones(self.M)/self.M
    
    def kmeans_init(self):
        locs = kmeans(self.data,self.M)
        for m in range(self.M):
            clustered_data = self.data[locs[:,m],:]
            cov_mat = np.cov(clustered_data.T)
            w,v = np.linalg.eig(cov_mat)
            v = v[:,np.argsort(w)]
            self.V[m] = v[:,-self.d:]
            self.U[m] = clustered_data.mean(axis=0).reshape((-1,1))
            self.o2[m] = np.var(clustered_data)
            self.pi[m] = clustered_data.shape[0]/self.N
    
    def get_likelihood(self,P):
        L = np.log(np.sum(self.pi*P,axis=1)).sum()
        return L
    
    def predict(self,data):
        P = self.get_prob(data)
        return P
    
    def get_prob(self,data):
        P = np.ones((data.shape[0],self.M))
        for m in range(self.M):
            Cm = self.o2[m]*np.eye(self.D) + self.V[m,:,:].dot(self.V[m,:,:].T)
            invCm = np.linalg.inv(Cm)
            centralized_data = data - self.U[m].T
            P[:,m] = np.sum(centralized_data.dot(invCm)*centralized_data,axis=1)
            P[:,m] = np.exp((-1/2)*P[:,m])
            P[:,m] *= (2*np.pi)**(-self.D/2) * np.linalg.det(Cm)**(-1/2)
        return P
    
    def one_step(self,managed):
        P = self.get_prob(self.data)
        R = (self.pi*P)/np.sum(self.pi*P,axis=1).reshape((-1,1))
        pi_new = R.sum(axis=0)/self.N
        U_new = np.zeros_like(self.U)
        V_new = np.zeros_like(self.V)
        o2_new = np.zeros_like(self.o2)
        for m in range(self.M):
            U_new[m] = np.sum(R[:,m].reshape((-1,1))*self.data,axis=0).reshape((-1,1))/R[:,m].sum()
            centralized_data_new = self.data - U_new[m].T
            Sm = (centralized_data_new*R[:,m].reshape((-1,1))).T.dot(centralized_data_new)/(pi_new[m]*self.N)
            Tm = self.o2[m]*np.eye(self.d) + self.V[m].T.dot(self.V[m])
            invTm = np.linalg.inv(Tm)
            V_new[m] = Sm.dot(self.V[m]).dot(np.linalg.inv(self.o2[m]*np.eye(self.d) +\
                       invTm.dot(self.V[m].T).dot(Sm).dot(self.V[m])))
            o2_new[m] = np.max([1e-10,(1/self.d)*np.trace(Sm - Sm.dot(self.V[m]).dot(invTm).dot(V_new[m].T))])
        L = self.get_likelihood(P)
        
        if self.L_old is not None:
            if not managed or self.L_old < L:
                self.par_old = self.pi,self.U,self.V,self.o2
                self.pi,self.V,self.U,self.o2 = pi_new,V_new,U_new,o2_new
            else:
                self.pi,self.U,self.V,self.o2 = self.par_old
                L =  self.L_old
        else:
            self.par_old = self.pi,self.U,self.V,self.o2
            self.pi,self.V,self.U,self.o2 = pi_new,V_new,U_new,o2_new
        self.L_old = L
        return L
        
    def train(self,max_iter = 100,tol = 1e-2,managed = True):
        loss = 0
        for i in range(max_iter):
            loss_old = loss
            loss = self.one_step(managed)
            if np.abs(loss_old - loss) < tol:
                #print('tol reached')
                return
            print('epoch:%s, likelihood:%s'%(str(i),str(loss)),end = '\r')
        #print('max_iter reached')
