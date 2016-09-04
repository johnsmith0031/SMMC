# SMMC
Simple Installation of Spectral Multi-Manifold Clustering Algorithm

SMMC(Multi-Manifold Clustering Algorithm) is an Algorithm used to handle spectral clustering problems, expecially the problems in which significant intersections among different clusters exist.

This Package is a Simple Installation of SMMC method described in

Y Wang, Y Jiang, Y Wu, ZH Zhou, "Spectral clustering on multiple manifolds", IEEE Transactions on Neural Networks, 2011, 22(7):1149-1161

#How to use this package:

Here is the example to cluster two circles, in which X is the dataset:<br><br><br>

from SMMC.SMMC import SMMC<br>
<br>
test = SMMC(X)<br>
test.train_mppca(d = 2, M = 40, max_iter = 200, tol = 1e-4, kmeans_init = False)<br>
locs = test.run_cluster(8,20,2)<br><br><br>

And You can plot your result:<br><br><br>

![Image](https://github.com/hyychong/SMMC/raw/master/examples/example3.png)


More Examples are in directory examples, like this:<br>
![Image](https://github.com/hyychong/SMMC/raw/master/examples/example6_1.png)
![Image](https://github.com/hyychong/SMMC/raw/master/examples/example6_2.png)<br>
![Image](https://github.com/hyychong/SMMC/raw/master/examples/example6_3.png)<br>

It's cool.


