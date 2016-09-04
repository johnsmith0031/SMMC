from SMMC.SMMC import SMMC
import matplotlib.pyplot as plt
import numpy as np

'''
    Example 1: Two lines
'''

k1 = np.random.rand()
k2 = np.random.rand()
x1 = np.random.rand(200)*2 - 1
x2 = np.random.rand(200)*2 - 1
y1 = x1*k1 + np.random.rand(200)*0.08
y2 = -k2*x2 + np.random.rand(200)*0.08
X = np.row_stack((np.column_stack((x1,y1)),np.column_stack((x2,y2))))

test = SMMC(X)
test.trian_mppca(d = 2, M = 2, max_iter = 200, tol = 1e-4, kmeans_init = False)
locs = test.run_cluster(8,20,2)
K = 2
if X.shape[1] == 2:
    for i in range(K):
        plt.plot(X[locs[:,i],0],X[locs[:,i],1],'.',alpha = 0.5)

'''
    Example 2: A line and a circle
'''

x1 = np.random.randn(200)*0.01
y1 = np.random.rand(200)*4 - 2
t = np.linspace(0,np.pi*2,200)
r = np.cos(t)
x2 = r*np.cos(t) + np.random.randn(200)*0.01 - 0.5
y2 = r*np.sin(t) + np.random.randn(200)*0.01
X = np.row_stack((np.column_stack((x1,y1)),np.column_stack((x2,y2))))

test = SMMC(X)
test.trian_mppca(d = 2, M = 20, max_iter = 200, tol = 1e-4, kmeans_init = False)
locs = test.run_cluster(6,22,2)
K = 2
if X.shape[1] == 2:
    for i in range(K):
        plt.plot(X[locs[:,i],0],X[locs[:,i],1],'.',alpha = 0.5)

'''
    Example 3: Two circles
'''

t = np.linspace(0,np.pi*2,200)
r = np.cos(t)
x1 = r*np.cos(t) + np.random.randn(200)*0.01 - 0.5
y1 = r*np.sin(t) + np.random.randn(200)*0.01
t = np.linspace(0,np.pi*2,200)
r = np.cos(t)
x2 = r*np.cos(t)*2 + np.random.randn(200)*0.01
y2 = r*np.sin(t)*2 + np.random.randn(200)*0.01
X = np.row_stack((np.column_stack((x1,y1)),np.column_stack((x2,y2))))

test = SMMC(X)
test.trian_mppca(d = 2, M = 40, max_iter = 200, tol = 1e-4, kmeans_init = False)
locs = test.run_cluster(8,20,2)
K = 2
if X.shape[1] == 2:
    for i in range(K):
        plt.plot(X[locs[:,i],0],X[locs[:,i],1],'.',alpha = 0.5)
        
'''
    Example 4: Two parabolas
'''

k1 = np.random.rand()
k2 = np.random.rand()
k1,k2 = np.max([k1,k2]),np.min([k1,k2])
x1 = np.random.rand(200)*2 - 1
x2 = np.random.rand(200)*2 - 1
y1 = x1**2*k1 + np.random.rand(200)*0.02 + 0.2
y2 = x2**2*k2 + np.random.rand(200)*0.02
X = np.row_stack((np.column_stack((x1,y1)),np.column_stack((x2,y2))))

test = SMMC(X)
test.trian_mppca(d = 2, M = 40, max_iter = 200, tol = 1e-4, kmeans_init = False)
locs = test.run_cluster(8,20,2)
K = 2
if X.shape[1] == 2:
    for i in range(K):
        plt.plot(X[locs[:,i],0],X[locs[:,i],1],'.',alpha = 0.5)
        
'''
    Example 5: Two Spirals
'''

X = np.array([]).reshape((0,2))
for i in range(4):
    t = np.linspace(0,np.pi,100)
    r = (np.pi/2)**t*0.2 - 0.2
    x1 = r*np.cos(t+i*np.pi/2) + np.random.randn(100)*0.002
    y1 = r*np.sin(t+i*np.pi/2) + np.random.randn(100)*0.002
    X = np.row_stack((X,np.column_stack((x1,y1))))
plt.plot(X[:,0],X[:,1],'.')

test = SMMC(X)
test.trian_mppca(d = 2, M = 25, max_iter = 200, tol = 1e-4, kmeans_init = False)
locs = test.run_cluster(6,22,2)
K = 2
if X.shape[1] == 2:
    for i in range(K):
        plt.plot(X[locs[:,i],0],X[locs[:,i],1],'.',alpha = 0.5)
