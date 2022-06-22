import pandas as pd
import numpy as np 
from drfp import DrfpEncoder
from descriptors.dictionnaries import dict_solvent, Ni0, Ni2

def process_dataframe(df):
        # a bit more of preprocessing
    # remove NaN
    df.fillna('', inplace=True)
    # change solvent names to SMILES
    solvent_smiles = [dict_solvent[df.loc[i, 'solvent']] for i in range(len(df))]
    df.loc[:,'solvent'] = solvent_smiles
    # change precrusors names to [Ni] or [Ni++] 
    prec_smiles = [precursor_mapping(prec) for prec in df.loc[:, 'catalyst_precursor']]
    df.loc[:, 'catalyst_precursor'] = prec_smiles
    # change NoLigand to ''
    df.replace('NoLigand', '', inplace=True)
    # converting all reactants to one string
    df['all_reactants'] = df[reaction_reagents_reactants].agg('.'.join, axis=1)
    # make reaction smiles
    df['reaction_smiles'] = df[['all_reactants', 'product']].agg('>>'.join, axis=1)
    
    # get reaction_smiles and featurize them with the DRFP encoder
    RXN_SMILES = np.array(df['reaction_smiles'])
    drfp_encoder = DrfpEncoder()
    X = drfp_encoder.encode(RXN_SMILES)
    
    # get DOIs origin
    DOIs = df.DOI.to_list()
    # get CP class
    cps = df.coupling_partner_class.to_list()
    # get data origin
    origins = df.origin.to_list()
    
    yields = []
    for i, row in df.iterrows():
        yield_isolated = process_yield(row["isolated_yield"])
        yield_gc = process_yield(row['analytical_yield'])
        # If both yields are known, we keep the isolated yield
        if yield_gc is not None:
            y = yield_gc
        if yield_isolated is not None:
            y = yield_isolated
        yields.append(y)
    
    return np.array(X), np.array(yields), np.array(DOIs), np.array(cps), np.array(origins)


# reaction parameters used for the DRFP featurisation
reaction_reagents_reactants = ['substrate', 'effective_coupling_partner',
       'solvent', 'catalyst_precursor', 
       'effective_reagents', 'reductant',
       'effective_ligand']

    
# Maps a precursor to a category representing it's oxidation state; if precursor is unknown, returns the precursor itself 
def precursor_mapping(precursor):
    if precursor in Ni0:
        return "[Ni]"
    elif precursor in Ni2:
        return "[Ni++]"
    else:
        return str(precursor)

# takes a yield (with potential information as a string e.g. "not detected") and returns a float (e.g. 0)
def process_yield(y):
    if y in ['not detected', 'trace', 'ND', '<1', '<5', 'nd']:
        return 0
    if y in ['>95']:
        return 100
    try:
        # check if is not NaN
        if float(y)==float(y):
            return float(y)
        else:
            return None
    except:
        return None 