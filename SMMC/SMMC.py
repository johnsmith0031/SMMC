from .MPPCA import MPPCA
from .PrincipalAngle import PrincipalAngle
from .Kmeans import kmeans
import numpy as np

class SMMC():
    
    def __init__(self,data):
        '''
            Spectral Multi-Manifold Clustering
            
            data: Dataset

            The algorithm is described in:
                "Spectral clustering on multiple manifolds",
                Y Wang, Y Jiang, Y Wu, ZH Zhou,
                IEEE Transactions on Neural Networks, 2011, 22(7):1149-1161
        '''
        self.data = data
        
    def trian_mppca(self,d,M,max_iter = 100,tol = 1e-4,managed = True,kmeans_init = False):
        '''
            Train MPPCA Model
            
            d: Dimension of the latent space
            M: Number of probabilistic principal component analyzers
            max_iter: Max steps to run EM
            tol: A value to judge whether EM is converged
            managed: A flag to avoid diverge
            kmeans_init: Whether to use kmeans to initialize the parameters
        '''
        self.mppca = MPPCA(self.data,d,M,kmeans_init)
        self.mppca.train(max_iter = max_iter,tol = tol,managed = managed)
    
    def get_principal_angles(self):
        '''
            Get Principal Angle Matrix of local tangent spaces fitted by MPPCA
        '''
        V = self.mppca.V
        res = np.ones((len(V),len(V)))
        for i in range(len(V)-1):
            for j in range(i+1,len(V)):
                try:
                    temp = PrincipalAngle(V[i],V[j])
                    temp = np.prod(temp.get_pas())
                except:
                    temp = 1
                res[i,j] = temp
                res[j,i] = res[i,j]
        return res
    
    def get_P(self,o = 8):
        '''
            Get similarity matrix of local tangent spaces using Principal Angles
            
            o: A parameter to set the threshold for similarity between local tangent spaces
        '''
        ps = self.mppca.predict(self.data)
        angles = self.get_principal_angles()
        loc = ps.argmax(axis=1)
        P = np.ones((self.data.shape[0],self.data.shape[0]))
        for i in range(self.data.shape[0]):
            P[i,:] = angles[loc[i],loc]
        P = P**o        
        return P
        
    def get_Q(self,k = 20):
        '''
            Get local similarity matrix Q using Knn(k)
            
            k: The parameter of Knn
        '''
        res = np.zeros((len(self.data),len(self.data)))
        for i in range(len(self.data)):
            dis = np.sum((self.data - self.data[i,:])**2,axis=1)
            loc = dis.argsort()[1:(k+1)]
            res[i,loc] = 1
        return res
        
    def get_affinity(self,P,Q):
        res = P*Q
        return res
        
    def get_U(self,W):    
        '''
            Spectral Clustering and get singular vectors
            
            This part is a bit different from original article.
            The original article use the generalized eigon vectors of (E - W)u = rEu,
            but this part use eigon vectors of normalized Laplacian matrix instead
        '''
        E = np.diag(np.sum(W,axis=1))
        invE2 = np.linalg.inv(E)**(1/2)
        Z = invE2.dot(E - W).dot(invE2)
        U,w,V = np.linalg.svd(Z)
        return U
        
    def cluster(self,U,K):
        '''
            Cluster data into K clusters
            
        '''
        temp = U[:,-K:]*1
        temp /= np.sqrt(np.sum(temp**2,axis=1)).reshape((-1,1))
        locs = kmeans(temp,K)
        return locs
        
    def run_cluster(self,o,k,K):
        '''
            Complete Procedure of SMMC
            
            o: A parameter to set the threshold for similarity between local tangent spaces
            k: The parameter of Knn
            K: Number of clusters
        '''
        P = self.get_P(o = o)
        Q = self.get_Q(k = k)
        W = self.get_affinity(P,Q)
        U = self.get_U(W)
        locs = self.cluster(U,K = K)
        return locs