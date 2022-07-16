from rdkit import Chem
from rdkit.Chem import AllChem, Draw, BRICS, rdChemReactions
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import math
import pandas as pd
import copy
from descriptors.dictionnaries import descritpors_to_remove_al, descritpors_to_remove_ax, descritpors_to_remove_lig, Ni0, Ni2

def process_dataframe_dft(df, data_path = '/data/utils', origin=False, dim=False, AX_sub_only=False):
    """ Featurize the preprocessed NiCOLit dataset with DFT and physico-chemical descriptors already computed and stored as csv files and return additionnal data as yields, DOIs, coupling partner class and scope-optimization origin of the reaction :
    1. Physico-chemical featurization of solvents
    2. DFT featurization of Ligands
    3. DFT featurization of Substrates
    4. DFT featurization of Coupling Partner
    5. DFT featurization of Lewis Acids
    6. Concatenation of Time, Temperature and Molar Ratios
    7. One-Hot encoding of precursors
    8. One-Hot encoding of data origin (scope-optimization)
    9. Yield selection and concatenation of all informations.
    10. Standard scaling of the reaction descriptors
    11. Definition of a scope and an optimization subspace
    
            Parameters:
                    df     (dataframe): dataframe obtain from the NiCOLit csv file  
                    data_path    (str): name of the folder with featurization informations
                    origin      (bool): specify if the scope/optimization origin of the reaction is encoded or not. (8.)
                    dim         (bool): defines vectors for projection of the data on scope and optimization subspaces (9.)
                    AX_sub_only (bool): reduces the featurization to substrate and coupling partners only.
            Returns:
                    X          (np.array): DFT Featurization of the reaction
                    yields     (np.array): yield of the reactions
                    DOIs       (np.array): DOI of the reactions
                    mechanisms (np.array): coupling partner class of the reactions
                    origins    (np.array): scope/optimization origin of the reactions
                    (v_scope, v_optim) (np.array, np.array): vectors for projection of the data on scope and optimization subspaces
    """
    
    df = copy.copy(df)
    class_lig(df)
    class_sub(df)
    # 1.
    solv = pd.read_csv(data_path + "solvents.csv", sep = ',', index_col=0)
    solv.drop(columns=["polarisabilite", "Unnamed: 9"], inplace=True)
    solvents = [np.array(solv.loc[solvent]) for solvent in df["solvent"]]

    # 2.
    # issue : what should we put for nan ? 
    ligs = pd.read_csv(data_path + "ligand_dft.csv", sep = ',', index_col=0)
    ligs.drop(columns=descritpors_to_remove_lig, inplace=True)
    ligs.index.to_list()
    canon_rdkit = []
    for smi in ligs.index.to_list():
        try:
            canon_rdkit.append(Chem.CanonSmiles(smi))
        except:
            canon_rdkit.append(smi)
    ligs["can_rdkit"] = canon_rdkit
    ligs.set_index("can_rdkit", inplace=True)
    ligands = [np.array(ligs.loc[ligand]) for ligand in df["effective_ligand"]]
    
    # 3.
    substrate = pd.read_csv(data_path + "substrate_dft.csv", sep = ',', index_col=0)
    substrate.drop(columns=descritpors_to_remove_lig, inplace=True)
    canon_rdkit = [Chem.CanonSmiles(smi_co) for smi_co in substrate.index.to_list() ]
    substrate["can_rdkit"] = canon_rdkit
    substrate.set_index("can_rdkit", inplace=True)
    substrate = substrate[substrate.duplicated(keep='first') != True]
    substrate = substrate[~substrate.index.duplicated(keep='first')]
    substrates = [np.array(substrate.loc[sub]) for sub in df["substrate"]]
    
    # 4.
    AX = pd.read_csv(data_path + "AX_dft.csv", sep = ',', index_col=0)
    AX.drop(columns=descritpors_to_remove_ax, inplace=True)
    canon_rdkit = [Chem.CanonSmiles(smi_co) for smi_co in AX.index.to_list() ]
    AX["can_rdkit"] = canon_rdkit
    AX.set_index("can_rdkit", inplace=True)
    AXs = [np.array(AX.loc[ax]) for ax in df["effective_coupling_partner"]]
    
    # 5.
    AL = pd.read_csv(data_path + "AL_dft.csv", sep = ',', index_col=0)
    AL.drop(columns=descritpors_to_remove_al, inplace=True)
    canon_rdkit = []
    for smi in AL.index.to_list():
        try:
            canon_rdkit.append(Chem.CanonSmiles(smi))
        except:
            canon_rdkit.append(smi)
    AL["can_rdkit"] = canon_rdkit
    AL.set_index("can_rdkit", inplace=True)
    ALs = [np.array(AL.loc[al]) for al in df["Lewis Acid"]]
    
    # 6.
    time = times(df)
    temp = temperatures(df)
    equiv = equivalents(df)

    # 7.
    precursors = one_hot_encoding(np.array([precursor_mapping(precursor) for precursor in df["catalyst_precursor"]]).reshape(-1, 1))
    additives = one_hot_encoding(np.array([additives_mapping(precursor) for precursor in df["reagents"]]).reshape(-1, 1))
    
    # 8.
    if origin is True:
        Origin = one_hot_encoding(np.array(df["origin"]).reshape(-1, 1))
        
    # 9. 
    X, yields, DOIs, mechanisms, origins = [], [], [], [], []
    for i, row in df.iterrows():
        yield_isolated = process_yield(row["isolated_yield"])
        yield_gc = process_yield(row['analytical_yield'])
        # If both yields are known, we keep the isolated yield
        if yield_gc is not None:
            y = yield_gc
        if yield_isolated is not None:
            y = yield_isolated  
        if origin is True:
                feature_vector = np.concatenate((substrates[i], AXs[i], solvents[i], ligands[i], precursors[i], ALs[i], [temp[i]], equiv[i], [time[i]], Origin[i]))
        else:
            if AX_sub_only==True:
                feature_vector = np.concatenate((substrates[i], AXs[i]))  
            else:
                feature_vector = np.concatenate((substrates[i], AXs[i], solvents[i], ligands[i], precursors[i], ALs[i], [temp[i]], equiv[i], [time[i]]))
        
        X.append(feature_vector)
        yields.append(y)
        DOIs.append(row["DOI"])
        mechanisms.append(row["coupling_partner_class"])
        origins.append(row["origin"])
    
    # 10. 
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 11.
    if dim == True:
        d_scope = len(substrates[0]) + len(AXs[0])
        d_optim = len(solvents[0]) + len(ligands[0]) + len(precursors[0]) + len(ALs[0]) + 2 + len(equiv[0])
        d_tot = d_scope + d_optim
        v_scope = [1 if i < d_scope else 0 for i in range(d_tot)]
        v_optim = [0 if i < d_scope else 1 for i in range(d_tot)]
        
        return np.array(X), np.array(yields), np.array(DOIs), np.array(mechanisms), np.array(origins), (v_scope, v_optim), np.array(df.substrate_cat), np.array(df.ligand_cat)
    
    else : 
        return np.array(X), np.array(yields), np.array(DOIs), np.array(mechanisms), np.array(origins), np.array(df.substrate_cat), np.array(df.ligand_cat)


    
# Classifiction of substrates and ligands:
def class_lig(df):
    ligand_cat = ['Phos' ,'DiPhos', 'NHC', 'others']
    lig_cat = []
    for smi in df.effective_ligand:
        if 'P' in smi:
            if smi.count('P') == 1:
                lig_cat.append('Phos')
            else:
                lig_cat.append('DiPhos')
        elif '[C]' in smi:
            lig_cat.append('NHC')
        else:
            lig_cat.append('others')  
    df['ligand_cat'] = lig_cat

def class_sub(df):
    mols = [Chem.MolFromSmiles(smi) for smi in df.substrate]
    sub_class = []
    for mol in mols:
        if mol.HasSubstructMatch(Chem.MolFromSmiles('c1ncncn1')) or mol.HasSubstructMatch(Chem.MolFromSmiles('C1=NC=NC=N1')):
            sub_class.append('Otriazine')
        elif mol.HasSubstructMatch(Chem.MolFromSmiles('c1ccccc1OC(=O)C(C)(C)C')):
            sub_class.append('OPiv')
        elif mol.HasSubstructMatch(Chem.MolFromSmiles('c1ccccc1OC(=O)N')) or mol.HasSubstructMatch(Chem.MolFromSmarts('*1****c1OC(=O)N')) or mol.HasSubstructMatch(Chem.MolFromSmarts('*1***c1OC(=O)N')) :
            sub_class.append('OC(=O)N')
        elif mol.HasSubstructMatch(Chem.MolFromSmiles('c1ccccc1OC(=O)O')):
            sub_class.append('OC(=O)O')
        elif mol.HasSubstructMatch(Chem.MolFromSmiles('c1ccccc1O[Si](C)(C)C')) or mol.HasSubstructMatch(Chem.MolFromSmarts('c1ccccc1o[Si](C)(C)C')):
            sub_class.append('OSi(C)(C)C')
        else:
            mol = Chem.AddHs(mol)
            if mol.HasSubstructMatch(Chem.MolFromSmiles('c1ccccc1OC(=O)C')):
                sub_class.append('OAc')
            elif mol.HasSubstructMatch(Chem.MolFromSmiles('c1ccccc1Oc1ccccc1')):
                sub_class.append('OPh')
            elif mol.HasSubstructMatch(Chem.MolFromSmiles('OCOC')):
                sub_class.append('OCOC')
            elif mol.HasSubstructMatch(Chem.MolFromSmiles('OC([H])([H])[H]')):
                sub_class.append('OCH3')
            else:
                sub_class.append('others')           
    sub_classes = np.unique(sub_class)
    df['substrate_cat'] = sub_class


# Converts a list of integers or strings to a one hot featurisation
def one_hot_encoding(x):
    enc = OneHotEncoder(sparse=False)
    enc.fit(x)
    return enc.transform(x)

# Maps a precursor to a category representing it's oxidation state; if precursor is unknown, returns the precursor itself 
def precursor_mapping(precursor):
    if precursor in Ni0:
        return "Ni0"
    elif precursor in Ni2:
        return "Ni2"
    else:
        return str(precursor)

# Maps an additive to its category
def additives_mapping(add):
    add = str(add)
    add = add.replace('[Sc+++]', '[Sc+3]').replace('[Ti++++]', '[Ti+4]').replace('[Al+++]', '[Al+3]').replace('[Fe+++]', '[Fe+3]').replace('[HO-]', '[O-]')
    if Chem.MolFromSmiles(add):
        return Chem.CanonSmiles(add)
    else:
        return 'nan'

def ligand_mapping(ligand):
    try:
        if math.isnan(ligand):
            return "None"
    except:
        return ligand       

    
# takes a yield (with potential information as a string e.g. "not detected") and returns a float (e.g. 0)
# this cleaning will have to take place in a separate part of the code 
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

# fonction to add a suffix coressponding to the descriptor category
def add_suffix(list_descriptor, suf):
    list_descriptor_suf = []
    for descriptor in list_descriptor:
        list_descriptor_suf.append(str(descriptor + '_' + suf))
    return list_descriptor_suf

# function to canonicalize additives : removes all duplicates in smi.split('.')
def canonicalize_additives(smiles):
    uniques = set(smiles.split('.'))
    s = ""
    for add in uniques:
        s += add
        s += '.'     
    return s[:-1]

def one_hot_encoding_with_names(x):
    enc = OneHotEncoder(sparse=False)
    enc.fit(x)
    return enc.transform(x), enc.get_feature_names_out()

# add temperatures to the featurisation
def temperatures(df):
    temp = df["temperature"].to_list()
    temp = ['25' if x == 'rt' else x for x in temp]
    temp = [str(x).replace('°C', '') for x in temp]
    replacements = {'23-100':'60', '23-65':'44', '60-100':'80', '80-120':'100', '110-130':120}
    replacer = replacements.get
    temp = [float(replacer(n, n)) for n in temp]
    return np.array(temp)

# adds equivalents to the featurisation
def equivalents(df):
    df = df[['eq_substrate','eq_coupling_partner', 'eq_catalyst', 'eq_ligand','eq_reagent']]
    return df.values.astype(float)

def is_float(value):
    try:
        float(value)
        return True
    except:
        return False
    
def times(df_t):
    df_t["time"] = df_t["time"].map(lambda x : x.replace('h', ''))
    df_t["time"] = df_t["time"].map(lambda x : float(x) if is_float(x) else x )
    df_t["time"] = df_t["time"].map(lambda x : float(x.replace('min',''))/60 if 'min' in str(x) else x)
    replacements = {'2-15':'8.5', '6-12':'9', '>12':'24', '5-20':'12.5'}
    replacer = replacements.get
    time = [float(replacer(n, n)) for n in df_t["time"].values]
    return np.array(time)