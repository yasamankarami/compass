# Background

> Understanding Communication Networks in Molecular Complexes 


**compass** is an advanced computational tool that analyzes the communication networks between protein-protein and protein-nucleic acid complexes. It leverages molecular dynamics (MD) simulation data to extract essential inter-residue properties, including dynamical correlations, interactions, and distances. For a comprehensive methodological introduction, please refer to the following [paper]().


## Installation
The installation of **compass** is straightforward. Just type the following commands in your terminal (assuming you've got Anaconda or Miniconda up and running):

```bash
   git clone https://github.com/rglez/compass.git
   cd compass
   conda env create -f environment.yml
   conda activate compass  
````
   
## Basic usage
As a command-line tool, **compass** requires only the path to a well-structured configuration file (refer to the [documentation](#documentation) for instructions on preparing it). Once installed, you can run it using the following command:

```bash
   compass path-to-config-file.cfg
``` 

## Documentation

For comprehensive documentation and tutorials, visit [the wiki](../../wiki).

## Contributing
If you are interested in contributing to the project, please review the guidelines in the [CONTRIBUTING.rst](CONTRIBUTING.rst) file.


## License
This project is licensed under the MIT License. For more details, check the [LICENSE](LICENSE.txt) file.


## Citation
If you use **compass** in your research, please let us knoow by citing the following paper::

```
(soon available)
```
