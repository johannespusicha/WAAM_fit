# WAAM_fit
Evaluates the fitness of stuctures for WAAM manufacturing.

## How to use
The main part of the application is a python program which is invoked from the command line with:
```
python -m main <input-file> -o <output-file> -s <mesh-size>
```

However, to run the python program, compilation and installation of the rust bindings is needed.
It is recommended to install the rust_methods module in a python virtual envrionment. 
In the venv the python package `maturin` has to be installed which will handle compilation and installation of the rust module.
```
pip install maturin
```
Afterwards, 
```
maturin develop --release
``` 
has to be called from the directory `WAAM_fit/rust-methods` to compile and install the rust bindings as a python module (see also the [maturin package documentation](https://www.maturin.rs)).
