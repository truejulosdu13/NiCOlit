# Prediction of reaction yields on a chemical diversity calibrated dataset: the NiCOLit dataset

This is the code for the "Title to find" paper to be published in ???.

A preprint version can be found on [ChemArxiv](link)

# Description :

A description of the different folders is given below :

- **aqc_utils** : this folder contains usefull tools released by [Auto-QChem](https://github.com/PrincetonUniversity/auto-qchem/)
- **data** : 
  * the NiCOLit dataset downloadable [here](https://github.com/truejulosdu13/DFT_for_SM/blob/main/data/Data_test11262021.csv).
  * HTE : HTE datasets published by [D. T. Anheman *et al.*](https://www.science.org/doi/10.1126/science.aar5169) and [A. B. Santanilla *et al.*](https://www.science.org/doi/10.1126/science.1259203) and used in a publication of [P. Scwhaller](https://rxn4chemistry.github.io/rxn_yields/)
  * utils : csv files of dft molecular featurization needed for dft-featurization.
  * rxnfp_featurization : csv files of Anhemand,  Santilla and NiCOlit datasets featurized with the RXNFP method
- **descriptors** :
  * preprocessing of the NiCOlit datase python script.
  * dft and rdkit featurisation python scripts.
- **notebook** :
  * description on going
  * description going
- **images** : All images form visualisation...
- **results** :




# Install requirements

Best use with following requirements :
```
conda create --name dft_for_sm python=3.9
conda install jupyter pandas scipy matplotlib pymongo pyyaml fabric xlrd appdirs openpyxl
conda install -c conda-forge openbabel=3.1.1
conda install -c conda-forge rdkit=2021.03.5
python -m pip install "pymongo[srv]"
```

