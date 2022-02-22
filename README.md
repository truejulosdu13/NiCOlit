# Prediction of reaction yields on a diverty calibrated dataset

This is the code for the "Title to find" paper to be published in ???.

A preprint version can be found on ChemArxiv[link]

# Description :

A description of the different folders is given below :

- aqc_utils : this folder contains usefull tools released by Auto-QChem[https://github.com/PrincetonUniversity/auto-qchem/]
- data : 
  * this folder contains the NiCOLit dataset 
- descriptors :
- notebook :
- images :
- results :

 https://github.com/PrincetonUniversity/auto-qchem/



Install requirements

Create a new conda environment:

# Description :

- aqc_utils : outils récupérés sur https://github.com/PrincetonUniversity/auto-qchem/

# Environnement :

Best use with following requirements :
```
conda create --name dft_for_sm python=3.9
conda install jupyter pandas scipy matplotlib pymongo pyyaml fabric xlrd appdirs openpyxl
conda install -c conda-forge openbabel=3.1.1
conda install -c conda-forge rdkit=2021.03.5
python -m pip install "pymongo[srv]"
```

