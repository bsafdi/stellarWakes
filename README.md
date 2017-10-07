# stellarContrails

This code uses stellar kinematic data to search for dark matter (DM) subhalos through their gravitational perturbations to the stellar phase-space distribution.

## Authors

- Malte Buschmann; buschman at umich.edu
- Joachim Kopp; jkopp at uni-mainz dot de
- Benjamin R. Safdi; bsafdi at umich dot edu
- Chih-Liang Wu; cliang at mit.edu

If you make use of `stellarContrails` in a publication, please cite 1710.xxxxx.

## Compiling and Running

This code package is written in `python` and `cython`.  The easiest way to install `stellarContrails`, along with it's dependent Python packages, is to use the setup script

```
python setup.py install
```

which also builds the `cython` modules. To just compile the `cython` modules locally:  

```
make build
```

## Examples

An example of how to use `stellarContrails` is provided in the `examples/` folder in the form of a `jupyter` notebook.  That notebook uses example stellar kinematic data, with and without a subhalo, provided in the `examples/data/` folder.  