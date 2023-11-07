# WAAM_fit
Evaluates the fitness of stuctures for WAAM manufacturing.

## How to use
The main part of the application is a python program which is invoked from the command line with, while `<input-file>` is a geometry file in the `.stp`-format:
```
waam_fit <input-file> -o <output-file> -s <mesh-size>
```
See the examples dir for example input files.

## Installation
However, to run the python program, compilation and installation of the rust bindings and the command-line-tool is needed, which is handeld by maturin
It is recommended to install the tool in a python virtual envrionment (either `venv` or `conda`). 
In the virtual environment install the python dependencies specified in `requirements.txt`. Fo example with using `pip` call :
```
pip install -r requirements.txt
```
in the project directory.

Afterwards, the commands
```
maturin build --release
```
and
```
pip install waam_fit
``` 
have to be executed to compile the rust bindings and install the python script as command-line-tool (see also the [maturin package documentation](https://www.maturin.rs)).
