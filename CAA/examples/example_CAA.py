import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy import sparse
from scipy.sparse.linalg import svds
from CAA.logreg import solver_logreg
from CAA.utils.plot_utils import configure_plt, C_LIST
from libsvmdata import fetch_libsvm

configure_plt()
# data generation


# Z, y = fetch_libsvm("rcv1.binary", normalize=False)
Z, y = fetch_libsvm("madelon", normalize=False)

all_algos = [
    ('GD', False, None, None, 1),
    ('AA no reg', True, None, None, 1),
    ('CAA adapt', True, 10, None, 1),
    ('AA reg', True, None, 1e-9, 1),
    ]
all_Es = {}
all_Ts = {}

tol = 1e-10  # stop when gradient norm smaller than tol
fgap = 1_00  # frequency of storage of gradient norms
max_iter = 50_001  # maximal number of outter iterations
verbose = True
max_time = 15  # maximal running time for the methods

X = Z.toarray()

is_sparse = sparse.issparse(X)


print("Lipschitz constant computation started")

if is_sparse:
    print("sparse")
    L = svds(X, k=1)[1][0] ** 2 / 4
    # L = power_method(X, max_iter=100) ** 2/4
else:
    L = norm(X, ord=2) ** 2/4


mu = 1e-8 * L  # amount of l_2 regularization


print("Lipschitz constant computation done")
print(L)
for algo in all_algos:
    algo_name = algo[0]
    use_acc = algo[1]
    C = algo[2]
    reg = algo[3]
    iters = algo[4]
    start_time = time.time()
    w, E, T = solver_logreg(
        X, y, rho=mu, verbose=verbose,
        tol=tol, C0=C, adaptive_C=True, use_acc=use_acc,
        max_iter=max_iter*iters, f_grad=fgap, K=5,
        reg_amount=reg, max_time=max_time)
    print("%s --- %s seconds ---" % (algo_name, time.time() - start_time))
    all_Es[algo] = E
    all_Ts[algo] = T


dict_color = {}
dict_color['GD'] = C_LIST[0]
dict_color['AA no reg'] = C_LIST[1]
dict_color['CAA adapt'] = C_LIST[2]
dict_color['AA reg'] = C_LIST[3]

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
    label = algo_name
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
