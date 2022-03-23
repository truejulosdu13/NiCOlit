# Prediction of reaction yields on a chemical diversity calibrated dataset: the NiCOLit dataset

This is the code for "Can Organic Chemistry Literature Enable Machine
Learning Yield Prediction ?".

A preprint version will soon be found on [ChemArxiv](link)
The NiCOlit dataset is accessible [here](https://github.com/truejulosdu13/NiCOlit/blob/main/data/NiCOlit.csv)

# Description :

A description of the different folders is given below :

- **aqc_utils** : this folder contains usefull tools released by [Auto-QChem](https://github.com/PrincetonUniversity/auto-qchem/)
- **data** : 
  * the NiCOlit dataset downloadable [here](https://github.com/truejulosdu13/NiCOlit/blob/main/data/NiCOlit.csv).
  * HTE : HTE datasets published by [D. T. Ahneman *et al.*](https://www.science.org/doi/10.1126/science.aar5169) and [A. B. Santanilla *et al.*](https://www.science.org/doi/10.1126/science.1259203) and used in a publication of [P. Scwhaller](https://rxn4chemistry.github.io/rxn_yields/)
  * utils : csv files of dft molecular featurization needed for dft-featurization.
  * rxnfp_featurization : csv files of Ahneman,  Santanilla and NiCOlit datasets featurized with the RXNFP method
- **descriptors** :
  * preprocessing of the NiCOlit dataset.
  * DFT and RDKit featurisation.
- **images** : All images displayed in the article.


# Install requirements

Best use with following requirements :
```
pip install -r requirements.txt
```

