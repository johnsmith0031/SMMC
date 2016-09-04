import numpy as np

class PrincipalAngle():
    
    def __init__(self,A,B):
        self.A = A
        self.B = B
        self.QA = self.schmidt(A)
        self.QB = self.schmidt(B)
    
    def schmidt(self,base,tol = 1e-8):
        res = np.zeros((base.shape[0],np.linalg.matrix_rank(base,1e-3)))
        scale = np.zeros(res.shape[1])
        i = 0
        while i < res.shape[1]:
            res[:,i] = 0
            res[:,i] += base[:,i]
            if i > 0:
                res[:,i] -= np.sum(base[:,i].T.dot(res[:,:i])/scale[:i]*res[:,:i],axis=1)
            scale[i] = np.sum(res[:,i]**2)
            if scale[i] < tol:
                loc = np.repeat(True,base.shape[1])
                loc[i] = False
                base = base[:,loc]
                continue
            i += 1
        res /= np.sqrt(scale)
        return res
        
    def get_pas(self):
        res = np.linalg.svd(self.QA.T.dot(self.QB))[1]
        return res
    