import pandas as pd
import numpy as np 
from drfp import DrfpEncoder

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

# solvent mapping
dict_solvent={'toluene': 'Cc1ccccc1',
             'Et2O': 'CCOCC', 
             'THF': 'C1COCC1', 
             'dioxane' : 'C1COCCO1',
             'DME': 'COCCOC', 
             'DMA': 'O=C(C)N(C)C', 
             'tBuOH': 'OC(C)(C)C',
             'm-xylene': 'Cc1cc(C)ccc1',
             'o-xylene': 'Cc1c(C)cccc1',
             'p-xylene': 'Cc1cc(C)ccc1',
             'DMF': 'O=CN(C)C',
             'NMP': 'O=C1N(C)CCC1',
             'DCE': 'ClCCCl',
             'EtOH': 'CCO',
             'CPME': 'COC1CCCC1', 
             'sBuOH': 'OC(C)CC', 
             'iPrOH': 'OC(C)C',
             'MeOH': 'OC',
             't-amyl alcohol': 'OC(C)(C)CC',
             'benzene': 'c1ccccc1',
             'CH3CN': 'N#CC',
             'hexane': 'CCCCCC',
             'nBu2O': 'CCCCOCCCC',
             'iPr2O': 'CC(C)OC(C)C', 
             '(EtO)2CH2': 'CCOCOCC',
             'tAmOMe': 'COC(C)(C)CC',
             'tBuOMe': 'COC(C)(C)C',
             'tAmOMe + Et2O': 'COC(C)(C)CC.CCOCC', 
             '(EtO)2CH2 + Et2O': 'CCOCOCC.CCOCC'}
    
# Maps a precursor to a category representing it's oxidation state; if precursor is unknown, returns the precursor itself 
def precursor_mapping(precursor):
    if precursor in Ni0:
        return "[Ni]"
    elif precursor in Ni2:
        return "[Ni++]"
    else:
        return str(precursor)
    
Ni0 = ['Ni(cod)2', 'Ni(dcypbz)(CO)2', 'Ni(dcype)(CO)2', 'Ni(dcypt)(CO)2', 'Ni(dppe)(CO)2', 'Ni(L1)(CO)2',
            'Ni(L2)(CO)2', 'Ni(L3)(CO)2', 'Ni(L4)(CO)2', 'Ni(L5)(CO)2', 'Ni(L6)(CO)2', 'Ni(PCy3)2', 'Ni(PPh3)4']
Ni2 = ['Ni(acac)2', 'NiBr2', 'NiBr2(glyme)', 'NiBr2(diglyme)', 'NiBr2(IPr)2', 'NiBr2(PCy3)2', 'NiBr2(PCy3)(IPr)', 'NiBr2(PCy3)', 
       '(ItBu)', 'NiBr2(PPh3)2', 'NiBr2(PPh3)(IPr)', 'NiBr2(PPh3)(ItBu)', 'Ni(2-CF3Ph)Br(dcypf)', 'NiCl2', 'NiCl2(dme)', 
       'NiCl2(dme)2', 'NiCl2(dppb)', 'NiCl2(dppe)', 'NiCl2(dppf)', 'NiCl2(dppp)', 'NiCl2(glyme)', 'NiCl2(IPr) (PPh3)',
       'NiCl2(PBu3)2', 'NiCl2(PCy3)2', 'NiCl2(phen)', 'NiCl2(PEt3)2', 'NiCl2(Ph2PCy)2', 'NiCl2(PhPCy2)2', 'NiCl2(PiBu3)2',
       'NiCl2(PiPr3)2', 'NiCl2(PMe3)2', 'NiCl2(PPh3)2', 'NiCl2(PPh3)(ItBu)', 'NiCl2(Py)2', 'Ni(2-ethylPh)Br', 'NiF2',
       'NiI2', 'NiO', 'Ni(OAc)2-4H2O', 'Ni(2-OMePh)Br ', 'Ni(OTf)2', 'Ni(o-tol)Br', 'Ni(o-tol)Cl', 'Ni(naphtyl)Br ',
       'Ni(PCy3)2(C2H4)', 'Ni(2,4-xylyl)Br', 'Ni(2,6-xylyl)Br', 'NiCl2(dme)2', 'Ni(OAc)2', 'NiBr2(bipy)2',
       'CCCCN4c1ccccc1N5c6cccc7N3c2ccccc2N(CCCC)C3[Ni](Br)(C45)[n+]67.[Br-]', 'CN2C=CN3c4cccc5N1C=CN(C)C1[Ni](Br)(C23)[n+]45.[Br-]',
       'Ni(o-tol)Cl(dppf)', 'Ni(o-tol)Cl(dippf)', 'Ni(o-tol)Cl(dcypf)', 'Ni(2-OMePh)Br(dcypf)', 'Ni(2,4-xylyl)Br(dcypf)',
       'Ni(naphtyl)Br(dcypf)', 'Ni(2,6-xylyl)Br(dcypf)', 'Ni(o-tol)Br(dcypf)', 'Ni(2-ethylPh)Br(dcypf)', 'NiCl2(Pph3)2',
       'Cl[Ni](Cl)([P+](C1CCCCC1)(C2CCCCC2)C(Nc3ccccc3n5nc(c4ccccc4)cc5c6ccccc6)c7ccccc7)[P+](C8CCCCC8)(C9CCCCC9)C(Nc%10ccccc%10n%12nc(c%11ccccc%11)cc%12c%13ccccc%13)c%14ccccc%14',
       'CC(C)[P+](C(C)C)(C(Nc1ccccc1n3nc(c2ccccc2)cc3c4ccccc4)c5ccccc5)[Ni](Cl)(Cl)[P+](C(C)C)(C(C)C)C(Nc6ccccc6n8nc(c7ccccc7)cc8c9ccccc9)c%10ccccc%10',
       'Cl[Ni](Cl)([P+](c1ccccc1)(c2ccccc2)C(Nc3ccccc3n5nc(c4ccccc4)cc5c6ccccc6)c7ccccc7)[P+](c8ccccc8)(c9ccccc9)C(Nc%10ccccc%10n%12nc(c%11ccccc%11)cc%12c%13ccccc%13)c%14ccccc%14', 'NiBr2(PCy3)(ItBu)', 'NiCl(PCy3)2(para-trifluorophenyl)', 'NiCl2(dppp)']

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