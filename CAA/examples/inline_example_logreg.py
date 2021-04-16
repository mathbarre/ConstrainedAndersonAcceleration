# %%
import sys
import time
import numpy as np
from numpy.linalg import norm
from scipy import sparse
from scipy.sparse.linalg import svds
from CAA.logreg import solver_logreg
from libsvmdata import fetch_libsvm
from CAA.utils.utils import power_method

if __name__ == '__main__':
    if len(sys.argv) > 5:
        dataset = sys.argv[1]
        max_iter = int(sys.argv[2])
        fgap = int(sys.argv[3])
        is_sparse = bool(sys.argv[4])
        try:
            if len(sys.argv) < 5:
                conditionings = [1e-7, 1e-8, 1e-9]
            else:
                conditionings = []
                for i in range(5, len(sys.argv)):
                    conditionings.append(float(sys.argv[i]))
            X, y = fetch_libsvm(dataset, normalize=False)  # good
            #is_sparse = sparse.issparse(X)
            if is_sparse:
                L = svds(X, k=1)[1][0] ** 2 / 4
            else:
                L = norm(X, ord=2) ** 2 / 4
                X = X.toarray()
            for conditioning in conditionings:
                print(conditioning)
                tol = 1e-10
                C0 = 10
                all_algos = [
                    ('gd', False, None, None, 1),
                    ('AA no reg', True, None, None, 1),
                    ('CAA adapt', True, C0, None, 1),
                    # ('AA reg 1e-7', True, None, 1e-7, 1),
                    # ('AA reg 1e-8', True, None, 1e-8, 1),
                    # ('AA reg 1e-9', True, None, 1e-9, 1),
                    # ('AA reg 1e-10', True, None, 1e-10, 1),
                    # ('AA reg 1e-11', True, None, 1e-11, 1),
                    ]
                all_Es = {}
                all_Ts = {}

                # max_iter = 150_001
                max_iter = max_iter
                verbose = True
                n = int(np.floor(max_iter/fgap))+1

                for algo in all_algos:
                    algo_name = algo[0]
                    use_acc = algo[1]
                    C = algo[2]
                    reg = algo[3]
                    iters = algo[4]
                    start_time = time.time()
                    # _, E, gaps = solver_enet(
                    #     X, y, 1/C, rho=0.01, verbose=verbose,
                    #     tol=tol, algo=algo_name, use_acc=use_acc,
                    #     max_iter=max_iter, f_gap=fgap)
                    w, E, T = solver_logreg(
                        X, y, rho=conditioning*L, verbose=False,
                        tol=tol, C0=C, adaptive_C=True, use_acc=use_acc,
                        max_iter=max_iter*iters, f_grad=fgap, K=5,
                        reg_amount=reg, max_time=24*3600)
                    print("%s --- %s seconds ---" % (algo_name, time.time() -
                                                     start_time))
                    all_Es[algo] = np.hstack([np.array(E), np.zeros(n-len(E))])
                    all_Ts[algo] = np.hstack([np.array(T), np.zeros(n-len(T))])

                res = np.array([])
                for algo in all_algos:
                    res = np.hstack([res, all_Es[algo]])

                res = res.reshape((len(all_algos), -1))
                for algo in all_algos:
                    res = np.vstack([res, all_Ts[algo]])
                res = np.vstack([res, fgap * np.arange(res.shape[1])])

                name = ("./CAA/examples/results/logreg_%s_C0_%i_rho_%e.txt" %
                        (dataset, C0, conditioning))
                np.savetxt(name, res.T)
        except ValueError:
            print("dataset not known")
    else:
        print("please enter a datasets name, a number of" +
              " iterations and a frequence.")
