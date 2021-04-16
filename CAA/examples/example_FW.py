# %%
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy import sparse
from scipy.sparse.linalg import svds
from CAA.logreg import solver_logreg
from CAA.logcosh import solver_logcosh
from CAA.hinge import solver_hinge_smooth
from libsvmdata import fetch_libsvm
from CAA.utils.plot_utils import configure_plt, C_LIST
from CAA.utils.utils import power_method
from bcd4.data.synthetic import simu_linreg

configure_plt()
# data generation

# X, y = fetch_libsvm("phishing", normalize=False)
# X, y = fetch_libsvm("madelon", normalize=False)
# X, y = fetch_libsvm("german.numer", normalize=False) # good
# X, y = fetch_libsvm("liver-disorders", normalize=False) # good
# X, y = fetch_libsvm("phishing", normalize=False) # good

Z, y = fetch_libsvm("madelon", normalize=False) # good
# X, y = simu_linreg(n_samples=3000, n_features=2000, corr=0.999)

tol = 1e-10

all_algos = [
    # ('gd', False, None, None, 1),
    ('AA no reg', True, None, None, 1),
    ('CAA adapt', True, 10, None, 1),
    # ('AA reg 1e-7', True, None, 1e-7, 1),
    # ('AA reg 1e-8', True, None, 1e-8, 1),
    # ('AA reg 1e-9', True, None, 1e-9, 1),
    # ('AA reg 1e-10', True, None, 1e-10, 1),
    # ('AA reg 1e-11', True, None, 1e-11, 1),
    ]
all_Es = {}
all_Ts = {}


fgap = 1_00
#max_iter = 150_001
max_iter = 2000_001
verbose = True
w = 0

X = Z.toarray()
#X = Z
is_sparse = sparse.issparse(X)


print("Lipschitz constant computation started")

if is_sparse:
    print("sparse")
    L = svds(X, k=1)[1][0] ** 2 / 4
    # L = power_method(X, max_iter=100) ** 2/4
else:
    L = norm(X, ord=2) ** 2/4
#%%
print("Lipschitz constant computation done")
print(L)
for algo in all_algos:
    algo_name = algo[0]
    use_acc = algo[1]
    C = algo[2]
    reg = algo[3]
    iters = algo[4]
    start_time = time.time()
    # _, E, gaps = solver_enet(
    #     X, y, 1/C, rho=0.01, verbose=verbose,
    #     tol=tol, algo=algo_name, use_acc=use_acc, max_iter=max_iter, f_gap=fgap)
    w, E, T = solver_logreg(
        X, y, rho=1e-8*L, verbose=verbose,
        tol=tol, C0=C, adaptive_C=True, use_acc=use_acc, max_iter=max_iter*iters, f_grad=fgap, K=5, reg_amount=reg, max_time = 20)
    print("%s --- %s seconds ---" % (algo_name, time.time() - start_time))
    all_Es[algo] = E
    all_Ts[algo] = T



dict_color = {}
dict_color['gd'] = C_LIST[0]
dict_color['AA no reg'] = C_LIST[1]
dict_color['CAA adapt'] = C_LIST[2]
dict_color['AA reg 1e-9'] = C_LIST[3]

for algo in all_algos:
    Es = all_Es[algo]
    algo_name = algo[0]
    use_acc = algo[1]
    color = dict_color[algo_name]
    if use_acc:
        # marker = 'X'
        marker = None
    else:
        marker = None
    label = ' acc: %s' % (use_acc)
    plt.semilogy(
        fgap * np.arange(len(Es)), Es, label=label,
        color=color, marker=marker, markevery=5)
plt.ylabel("Norm of gradients")
plt.xlabel("Iterations")
plt.tight_layout()
plt.legend()
plt.show()

for algo in all_algos:
    Es = all_Es[algo]
    Ts = all_Ts[algo]
    algo_name = algo[0]
    use_acc = algo[1]
    color = dict_color[algo_name]
    if use_acc:
        marker = 'X'
    else:
        marker = None
    label = ' acc: %s' % (use_acc)
    plt.semilogy(
        Ts, Es, label=label,
        color=color, marker=marker, markevery=5)
plt.ylabel("Norm of gradients")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.legend()
plt.show()

# res = np.array([])
# for algo in all_algos:
#     res = np.hstack([res,all_Es[algo]])

# res = res.reshape((8,-1))
# for algo in all_algos:
#     res = np.vstack([res, all_Ts[algo]])
# res = np.vstack([res,fgap * np.arange(res.shape[1])])

# np.savetxt("logreg_C0_10_madelon-test_rho_1e-12.txt",res.T)
# %%
# Import packages.


# %%
