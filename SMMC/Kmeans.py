import numpy as np

def kmeans(data, k=2, max_iter = 100, centers = None, tol = 1e-4):
    
    if centers is None:
        center_locs = []
        i = 0
        while i < k:
            loc = np.random.randint(0,len(data))
            while loc not in center_locs:
                center_locs.append(loc)
            i += 1
        centers = data[center_locs,:]*1
    
    for t in range(max_iter):
        
        dis = np.zeros((data.shape[0],centers.shape[0]))
        for i in range(centers.shape[0]):
                dis[:,i] = np.sum((data - centers[i,:])**2, axis=1)
        clusters = np.argmin(dis,axis=1)
        
        old_centers = centers*1
        for i in range(centers.shape[0]):
            centers[i,:] = np.mean(data[clusters == i,:], axis = 0)
                
        if np.sum((old_centers - centers)**2) < tol:
            break
    
    locs = np.repeat(False,len(data)*k).reshape((-1,k))
    for i in range(centers.shape[0]):
        locs[:,i] = clusters == i
    
    return locs
