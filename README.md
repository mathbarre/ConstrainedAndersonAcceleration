# Convergence of Constrained Vector Extrapolation

Implementation of the Constrained Vector Extrapolation (or Constrained Anderson Acceleration) method from ['reference'](https://arxiv.org/abs/2010.15482)

### Authors

- [**Mathieu Barré**](https://mathbarre.github.io/)
- [**Adrien Taylor**](https://www.di.ens.fr/~ataylor/)
- [**Alexandre d'Aspremont**](https://www.di.ens.fr/~aspremon/)

### Requirement

To be able to run the code you need the following packages
- numpy
- scipy
- numba
- libsvmdata (https://github.com/mathurinm/libsvmdata)

### Install
Before using the code you should run in the folder ConstrainedAndersonAcceleration
```console
pip install -e .
```
Then, to test the method you can execute
```console
python ./CAA/examples/example_CAA.py
```
