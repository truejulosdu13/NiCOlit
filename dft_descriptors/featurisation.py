from rdkit import Chem
from rdkit.Chem import AllChem, Draw, BRICS, rdChemReactions
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import math
import pandas as pd
import copy

def process_dataframe_dft(df, data_path = '/data/utils', origin=False, dim=False, AX_sub_only=False):
    """ Featurize the preprocessed NiCOLit dataset with DFT and physico-chemical descriptors already computed and stored as csv files.
    1. Physico-chemical featurization of solvents
    2. DFT featurization of Ligands
    3. DFT featurization of Substrates
    3. DFT featurization of Substrates
    3. DFT featurization of Substrates
    
            Parameters:
                    df (dataframe): dataframe obtain from the NiCOLit csv file  
            Returns:
                    np.array(X)
                    np.array(yields)
                    np.array(DOIs)
                    np.array(mechanisms)
                    np.array(origins)
                    (v_scope, v_optim)
    """
    
    df = copy.copy(df)
    # 1.
    solv = pd.read_csv(data_path + "solvents.csv", sep = ',', index_col=0)
    solv.drop(columns=["polarisabilite", "Unnamed: 9"], inplace=True)
    solvents = [np.array(solv.loc[solvent]) for solvent in df["Solvent"]]

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
            print(smi)
    ligs["can_rdkit"] = canon_rdkit
    ligs.set_index("can_rdkit", inplace=True)
    ligands = [np.array(ligs.loc[ligand]) for ligand in df["Ligand effectif"]]
    
    # 3.
    substrate = pd.read_csv(data_path + "substrate_dft.csv", sep = ',', index_col=0)
    substrate.drop(columns=descritpors_to_remove_lig, inplace=True)
    #canon_rdkit = [Chem.CanonSmiles(smi_co) for smi_co in substrate.index.to_list() ]
    #substrate["can_rdkit"] = canon_rdkit
    #substrate.set_index("can_rdkit", inplace=True)
    substrate = substrate[substrate.duplicated(keep='first') != True]
    substrate = substrate[~substrate.index.duplicated(keep='first')]
    substrates = [np.array(substrate.loc[sub]) for sub in df["Reactant Smile (C-O)"]]
    
    # dft description for AX
    AX = pd.read_csv(data_path + "AX_dft.csv", sep = ',', index_col=0)
    AX.drop(columns=descritpors_to_remove_ax, inplace=True)
    canon_rdkit = [Chem.CanonSmiles(smi_co) for smi_co in AX.index.to_list() ]
    AX["can_rdkit"] = canon_rdkit
    AX.set_index("can_rdkit", inplace=True)
    AXs = [np.array(AX.loc[ax]) for ax in df["A-X effectif"]]
    
    # dft for Lewis Acid
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
    
    # temperatures
    temp = temperatures(df)
    
    # equivalents
    equiv = equivalents(df)
    
    # time
    time = times(df)
    
    
    precursors = one_hot_encoding(np.array([precursor_mapping(precursor) for precursor in df["Precurseur Nickel"]]).reshape(-1, 1))
    additives = one_hot_encoding(np.array([additives_mapping(precursor) for precursor in df["Base/additif après correction effective"]]).reshape(-1, 1))
    
    if origin is True:
        Origin = one_hot_encoding(np.array(df["type of data (Optimisation or scope)"]).reshape(-1, 1))
    
    X = []
    yields = []
    DOIs = []
    mechanisms = []
    origins = []
    
    for i, row in df.iterrows():
        yield_isolated = process_yield(row["Isolated Yield"])
        yield_gc = process_yield(row['GC/NMR Yield'])
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
        mechanisms.append(row["Mechanism"])
        origins.append(origin_mapping(row["type of data (Optimisation or scope)"]))
    

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    if dim == True:
        d_scope = len(substrates[0]) + len(AXs[0])
        d_optim = len(solvents[0]) + len(ligands[0]) + len(precursors[0]) + len(ALs[0]) + 2 + len(equiv[0])
        d_tot = d_scope + d_optim
        v_scope = [1 if i < d_scope else 0 for i in range(d_tot)]
        v_optim = [0 if i < d_scope else 1 for i in range(d_tot)]
        return np.array(X), np.array(yields), np.array(DOIs), np.array(mechanisms), np.array(origins), (v_scope, v_optim)
    else : 
        return np.array(X), np.array(yields), np.array(DOIs), np.array(mechanisms), np.array(origins)



# Dft descriptors that have been removed from the description :
descritpors_to_remove_al = ["converged", "stoichiometry", "ES_root_molar_volume", "X_0", "Y_0", "Z_0", "at_0", "ES_transition_7", "ES_transition_8", "ES_transition_9", 'ES_osc_strength_7', 'ES_osc_strength_8', 'ES_osc_strength_9', 'ES_<S**2>_7', 'ES_<S**2>_8', 'ES_<S**2>_9']

descritpors_to_remove_ax = ["number_of_atoms", "charge", "multiplicity", "molar_mass", "molar_volume", "E_scf", "zero_point_correction", "E_thermal_correction","H_thermal_correction", "G_thermal_correction", "E_zpe", "E", "H", "G", "stoichiometry", "converged", "ES_root_molar_volume", "ES_root_electronic_spatial_extent",
                        "X_0", "X_1", "X_2", "X_3",
                        "Y_0", "Y_1", "Y_2", "Y_3",
                        "Z_0", "Z_1", "Z_2", "Z_3",
                        "at_0", "at_1", "at_2", "at_3",
                        'ES_root_Mulliken_charge_0', 'ES_root_Mulliken_charge_1', 'ES_root_Mulliken_charge_2', 'ES_root_Mulliken_charge_3',
                         'ES_root_NPA_charge_0', 'ES_root_NPA_charge_1', 'ES_root_NPA_charge_2', 'ES_root_NPA_charge_3',
                         'ES_root_NPA_core_0', 'ES_root_NPA_core_1', 'ES_root_NPA_core_2', 'ES_root_NPA_core_3', 
                         'ES_root_NPA_valence_0', 'ES_root_NPA_valence_1', 'ES_root_NPA_valence_2', 'ES_root_NPA_valence_3',
                         'ES_root_NPA_Rydberg_0', 'ES_root_NPA_Rydberg_1', 'ES_root_NPA_Rydberg_2', 'ES_root_NPA_Rydberg_3',
                         'ES_root_NPA_total_0', 'ES_root_NPA_total_1', 'ES_root_NPA_total_2', 'ES_root_NPA_total_3',
                         'ES_transition_0', 'ES_transition_1', 'ES_transition_2', 'ES_transition_3', 'ES_transition_4', 'ES_transition_5', 'ES_transition_6', 'ES_transition_7', 'ES_transition_8', 'ES_transition_9',
                         'ES_osc_strength_0', 'ES_osc_strength_1', 'ES_osc_strength_2', 'ES_osc_strength_3', 'ES_osc_strength_4', 'ES_osc_strength_5', 'ES_osc_strength_6', 'ES_osc_strength_7', 'ES_osc_strength_8', 'ES_osc_strength_9',
                         'ES_<S**2>_0', 'ES_<S**2>_1', 'ES_<S**2>_2', 'ES_<S**2>_3', 'ES_<S**2>_4', 'ES_<S**2>_5', 'ES_<S**2>_6', 'ES_<S**2>_7', 'ES_<S**2>_8', 'ES_<S**2>_9']

descritpors_to_remove_lig = ["number_of_atoms", "charge", "multiplicity", "molar_mass", "molar_volume", "E_scf", "zero_point_correction", "E_thermal_correction","H_thermal_correction", "G_thermal_correction", "E_zpe", "E", "H", "G", "stoichiometry", "converged", "ES_root_molar_volume", "ES_root_electronic_spatial_extent",
    "X_0", "X_1", "X_2", "X_3", "X_4", "X_5", "X_6", "X_7",
    "Y_0", "Y_1", "Y_2", "Y_3", "Y_4", "Y_5", "Y_6", "Y_7", 
    "Z_0", "Z_1", "Z_2", "Z_3", "Z_4", "Z_5", "Z_6", "Z_7",
    "at_0", "at_1", "at_2", "at_3", "at_4", "at_5", "at_6", "at_7",                   'ES_root_Mulliken_charge_0','ES_root_Mulliken_charge_1','ES_root_Mulliken_charge_2','ES_root_Mulliken_charge_3','ES_root_Mulliken_charge_4','ES_root_Mulliken_charge_5','ES_root_Mulliken_charge_6',
'ES_root_Mulliken_charge_7',
'ES_root_NPA_charge_0','ES_root_NPA_charge_1', 'ES_root_NPA_charge_2', 'ES_root_NPA_charge_3', 'ES_root_NPA_charge_4', 'ES_root_NPA_charge_5','ES_root_NPA_charge_6','ES_root_NPA_charge_7',
 'ES_root_NPA_core_0', 'ES_root_NPA_core_1', 'ES_root_NPA_core_2', 'ES_root_NPA_core_3', 'ES_root_NPA_core_4', 'ES_root_NPA_core_5', 'ES_root_NPA_core_6', 'ES_root_NPA_core_7',
 'ES_root_NPA_valence_0', 'ES_root_NPA_valence_1', 'ES_root_NPA_valence_2', 'ES_root_NPA_valence_3', 'ES_root_NPA_valence_4', 'ES_root_NPA_valence_5', 'ES_root_NPA_valence_6', 'ES_root_NPA_valence_7',
 'ES_root_NPA_Rydberg_0', 'ES_root_NPA_Rydberg_1', 'ES_root_NPA_Rydberg_2', 'ES_root_NPA_Rydberg_3', 'ES_root_NPA_Rydberg_4', 'ES_root_NPA_Rydberg_5', 'ES_root_NPA_Rydberg_6', 'ES_root_NPA_Rydberg_7',
 'ES_root_NPA_total_0', 'ES_root_NPA_total_1', 'ES_root_NPA_total_2', 'ES_root_NPA_total_3', 'ES_root_NPA_total_4', 'ES_root_NPA_total_5', 'ES_root_NPA_total_6', 'ES_root_NPA_total_7',
 'ES_transition_0', 'ES_transition_1', 'ES_transition_2', 'ES_transition_3', 'ES_transition_4', 'ES_transition_5', 'ES_transition_6', 'ES_transition_7', 'ES_transition_8', 'ES_transition_9',
 'ES_osc_strength_0', 'ES_osc_strength_1', 'ES_osc_strength_2', 'ES_osc_strength_3', 'ES_osc_strength_4', 'ES_osc_strength_5', 'ES_osc_strength_6', 'ES_osc_strength_7', 'ES_osc_strength_8', 'ES_osc_strength_9',
 'ES_<S**2>_0', 'ES_<S**2>_1', 'ES_<S**2>_2', 'ES_<S**2>_3', 'ES_<S**2>_4', 'ES_<S**2>_5', 'ES_<S**2>_6', 'ES_<S**2>_7', 'ES_<S**2>_8','ES_<S**2>_9']

# Mapping to go from precursor to simplified category (oxidation state of the nickel) 
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
    
    
# list of additives 
# TODO: clean 
dict_additifs = {
    # Li
    'C[Li]',
    '[Li+].[Li+].[O-]C([O-])[O-]',
    '[Li+].[O-]C(C)(C)C',
    '[Li+].[Cl-]',
    # Li + Mg
    'c1cccc(C)c1[Mg]Br.[Li+].[Cl-]',
    'c1ccccc1[Mg]Br.[Li+].[Cl-].[Cl-]',
    'c1cc(C)ccc1[Mg]Br.[Li+].[Cl-]',
    'c1ccccc1[Mg]Br.[Li+].[Cl-]'
    # Mg
    '[Mg]',
    'c1cccc(C)c1[Mg]Br',
    'c1ccccc1[Mg]Br',
    'c1ccccc1[Mg]Br.[Mg++].[Cl-].[Cl-]',
    'C[Mg]Br',
    'CC[Mg]Br',
    'C1CCCCC1[Mg]Br',
    'c1ccccc1C[Mg]Br',
    'c1cc(C)ccc1[Mg]Br',
    'C[Mg]I',
    'Cc1cccc(C)c1[Mg]Br',
    'Cc1cc(C)cc(C)c1[Mg]Br',
    'c1ccccc1[Mg]Cl',
    'c1cccc(C)c1[Mg]Br.CC1(CCCC(N1[O-])(C)C)C',
    'c1cccc(C)c1[Mg]Br.C=C(c1ccccc1)c2ccccc2',
    '[Mg++].[F-].[F-]',
    # Mg + Sc
    'c1ccccc1[Mg]Br.[Sc+++].CC(C)(C)S(=O)(=O)[O-].CC(C)(C)S(=O)(=O)[O-].CC(C)(C)S(=O)(=O)[O-]',
    # Mg + Ti
    'c1ccccc1[Mg]Br.[Ti++++].CC(C)[O-].CC(C)[O-].CC(C)[O-].CC(C)[O-]',
    # Mg + Al
    'c1ccccc1[Mg]Br.[Al+++].[Cl-].[Cl-].[Cl-]',
    # Al
    '[Al]',
    # Zn
    '[Zn]',
    '[Zn].[Zn++].[Cl-].[Cl-]',
    '[Zn].c1ccccc1O',
    '[Zn].C[Si](C)(C)Cl',
    '[Zn].CC[N+](CC)(CC)CC.[I-]',
    '[Zn].CCN(CC)CC',
    '[Zn].OC(=O)C(C)(C)C',
    '[Zn++].[F-].[F-]',
    '[Zn].O',
    # Zn + K
    '[Zn].[K+].O=P(O)(O)[O-]',
    '[K+].[K+].[K+].[O-]P(=O)([O-])[O-].O.[Zn]',
    # Zn + Mg
    '[Zn].[Mg++].[Cl-].[Cl-]',
    # Zn + Na
    '[Zn].[Na+].[I-]',
    '[Zn].[Na+].OC([O-])[O-]',
    '[Zn].[Na+].O=P(O)(O)[O-]',
    # Mn
    '[Mn]',
    '[Mn].CC[N+](CC)(CC)CC.[I-]',
    '[Mn].CCC[N+](CCC)(CCC)CCC.[I-]',
    '[Mn].CCCC[N+](CCCC)(CCCC)CCCC.[I-]',
    '[Mn].CC[N+](CC)(CC)CC.[Br-]',
    '[Mn].CCC[N+](CCC)(CCC)CCC.[Cl-]',
    '[Mn].C[N+](C)(C)C.[I-]',
    '[Mn].C1C=CCCC=CC1',
    # Mn + Li
    '[Mn].[Li+].[I-]',
    '[Mn].[Li+].[O-]C(=O)C(C)(C)C',
    # Mn + Mg
    '[Mn].[Mg++].[Cl-].[Cl-]',
    '[Mn].[Mg++].[Br-].[Br-].CCOCC',
    # Mn + Na
    '[Mn].[Na+].[I-]',
    # Mn + K
    '[Mn].[K+].[I-]',
    # K
    '[K+].[K+].[O-]C([O-])[O-]',
    '[K+].[O-]C(C)(C)C',
    '[K+].[K+].[K+].[O-]P(=O)([O-])[O-]',
    '[K+].[K+].[K+].[O-]P([O-])([O-])=O',
    '[K+].[F-]',
    '[K+].[O-]',
    '[K+].[O-]C(=O)C',
    '[K+].OC([O-])[O-]',
    '[K+].[K+].[K+].[O-]P(=O)([O-])[O-].O.O.O.O.O.O',
    '[K+].[K+].[K+].[O-]P([O-])([O-])=O.O.O.O.O',
    '[K+].[HO-].O.O.O.O',
    '[K+].[O-]C(C)(C)C.O.O.O.O',
    'C(=O)([O-])[O-].[K+].[K+].O.O.O.O',
    '[K+].[K+].[K+].[O-]P([O-])([O-])=O.O.O',
    '[K+].[K+].[K+].[O-]P([O-])([O-])=O.O.O.O.O.O.O',
    '[K+].[K+].[K+].[O-]P([O-])([O-])=O.O.O.O.O',
    '[K+].[K+].[K+].[O-]P([O-])([O-])=O.O.O.O.O.O.O.O.O',
    '[K+].[K+].[K+].[O-]C(C)(C)C.O.O.O.O',
    '[K+].[F-].O.O.O.O',
    '[K+].[K+].[K+].[O-]P(=O)([O-])[O-].O',
    '[K+].[K+].[K+].[O-]P(=O)([O-])[O-].O.O.O',
    '[K+].[K+].OP(=O)([O-])[O-].O.O.O',
    '[K+].[K+].[O-]C([O-])[O-].O',
    '[K+].[K+].[K+].O=C([O-])[O-].O=C([O-])[O-].O',
    # Cs
    '[Cs+].[Cs+].[O-]C([O-])[O-]',
    '[Cs+].[O-]C(=O)C(C)(C)C',
    '[Cs+].[F-]',
    '[Cs+].[Cs+].[O-]C([O-])=O',
    '[Cs+].[F-].O.O.O.O',
    '[Cs+].[Cs+].[O-]C([O-])[O-].O.O.O.O',
    # Cs + Mg
    'c1ccccc1[Mg]Br.[Cs+].[Cl-]',
    'c1ccccc1[Mg]Br.[Cs+].[Cl-][Cl-]',
    # Na + Cs
    '[Na+].[O-]C(C)(C)C.[Cs+].[F-]',
    # Na
    '[Na+].[Na+].[Na+].[O-]P(=O)([O-])[O-]',
    '[Na+].[Na+].[O-]C([O-])[O-]',
    '[Na+].[O-]C(C)(C)C',
    '[Na+].[O-]',
    '[Na+].[O-]CC',
    '[Na+].[O-]C(C)(C)C.CC2(C)OB(c1ccccc1)OC2(C)C',
    'C(=O)([O-])[O-].[Na+].[Na+].O.O.O',
    # Rb
    '[Rb+].[Rb+].[O-]C([O-])[O-]',
    # Ag
    '[Ag+].[F-]',
    # Cu
    '[Cu++].[F-].[F-]',
    '[Cu++].[F-].[F-].CCCC[N+](CCCC)(CCCC)CCCC.[F-]',
    '[Cu++].[Br-].[Br-]',
    '[Cu++].[O-]S(=O)(=O)[O-]',
    '[Cu+].[I-]',
    '[Cu++].[O-]C(C)=O.[O-]C(C)=O',
    '[Cu++].[O-]S(=O)(=O)[O-]',
    # Cu + K
    '[Cu++].[F-].[F-].[K+].[K+].[K+].[O-]P(=O)([O-])[O-]',
    '[Cu++].[F-].[F-].[K+].[F-]',
    '[Cu++].[F-].[F-].[K+].[O-]P(O)(O)=O',
    # Cu + Cs
    '[Cu++].[F-].[F-].[Cs+].[F-]',
    '[Cu++].[F-].[F-].[Cs+].[F-].Cc1ccccc1',
    '[Cu++].[Br-].[Br-].[Cs+].[F-]',
    '[Cu++].[Cl-].[Cl-].[Cs+].[F-]',
    '[Cu++].FC(F)(F)S(=O)(=O)[O-].FC(F)(F)S(=O)(=O)[O-].[Cs+].[F-] ',
    '[Cu++].[O-]S(=O)(=O)[O-].[Cs+].[F-]',
    '[Cu+].[I-].[Cs+].[F-]',
    '[Cu++].[O-]C(=O)C.[Cs+].[F-]',
    # Cu + Al
    '[Cu++].[F-].[F-].[Al](C)(C)C',
    # Fe
    '[Fe+++].[F-].[F-].[F-]',
    # Cu + Sr
    '[Cu++].[F-].[F-].[Sr++].[F-].[F-]',
    # autres
    'nan',
    'CCN(CC)CC',
    'C2CCCN1CCCN=C1C2',
    'CCCC[N+](CCCC)(CCCC)CCCC.[F-]',
    'CC1(CCCC(N1[O-])(C)C)C' # TEMPO
}

def rxnfp(rxn_smarts, radius=2):
    rxn = rdChemReactions.ReactionFromSmarts(rxn_smarts)
    rxnfp = list(rdChemReactions.CreateDifferenceFingerprintForReaction(rxn))
    return rxnfp


# Converts single SMILES to Morgan Fingerprint 
def ecfp(smiles, radius=2):
    Chem.MolFromSmiles(smiles)
    ecfp = list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius))
    return ecfp

# Converts list of SMILES to list of Morgan Fingerprint 
def ecfp_list(smiles_list, radius=2):
    return [ecfp(smiles, radius=radius) for smiles in smiles_list]

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

# To rewrite 
def categorie_add_base(liste_additifs) :
    #extraction de toutes les smiles contenant des especes chargees negativement
    Liste_base = []
    dict_additifs_corr = [additives_mapping(smiles) for smiles in dict_additifs]
    
    for add in dict_additifs_corr:
        if add != 'nan':
            add = Chem.MolToSmiles(Chem.MolFromSmiles(add))
            if add not in Liste_base:
                if '-' in add:
                    Liste_base.append(add)
    # creation d'une liste des especes chargees negativement rencontrées
    Liste_base_unique = []
    for add in Liste_base:
        unique_add = add.split('.')
        for uni_add in unique_add:
            if uni_add not in Liste_base_unique:
                if '-' in uni_add:
                    Liste_base_unique.append(uni_add)
    # categorisation des additifs
    cat_add_base = []
    additives = [additives_mapping(smiles) for smiles in liste_additifs]
    for add in additives:
        if str(add) == 'nan':
            cat_add_base.append(None)
        else:
            add = Chem.MolToSmiles(Chem.MolFromSmiles(add))
            for i in Liste_base_unique:
                if i in add.split('.'):
                    cat = i
                    break
                else:
                    cat = None
            cat_add_base.append(cat)
    return cat_add_base
    
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
    
# Maps information on whether the reaction was from a scope/optimization table to a binary category optimization./scope
optimisation = ["Optimisation table", "optimisation - changement de ligand", "optimization", "Optimisation Table", "optimisation", *
                "optimisation table" ,"Optimisation", "Table d'optimisation", "Table Optimisation"]

def origin_mapping(information):
    if information in optimisation:
        return "optimisation"
    else:
        return "scope"
    
    
#featurisation of the dft reactions in order to perform a permutation analysis performance

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
    temp = df["Temperature"].to_list()
    temp = ['25' if x == 'rt' else x for x in temp]
    temp = [str(x).replace('°C', '') for x in temp]
    replacements = {'23-100':'60', '23-65':'44', '60-100':'80', '80-120':'100', '110-130':120}
    replacer = replacements.get
    temp = [float(replacer(n, n)) for n in temp]
    return np.array(temp)

# adds equivalents to the featurisation
def equivalents(df):
    df = df[['eq CO','eq A-X', 'eq Ni', 'eq Lig (lig + prec)','eq B (si reducteur pas pris en c0mpte)']]
    return df.values.astype(float)

# add temperatures to the featurisation
def is_float(value):
    try:
        float(value)
        return True
    except:
        return False
    
def times(df_t):
    df_t["Time"] = df_t["Time"].map(lambda x : x.replace('h', ''))
    df_t["Time"] = df_t["Time"].map(lambda x : float(x) if is_float(x) else x )
    df_t["Time"] = df_t["Time"].map(lambda x : float(x.replace('min',''))/60 if 'min' in str(x) else x)
    replacements = {'2-15':'8.5', '6-12':'9', '>12':'24', '5-20':'12.5'}
    replacer = replacements.get
    time = [float(replacer(n, n)) for n in df_t["Time"].values]
    return np.array(time)



# allmost duplicate function : to be remove
def dft_ft(df, data_path = '../data_csv/'):
    # physico-chemical description of solvents
    solv = pd.read_csv(data_path + "solvents.csv", sep = ',', index_col=0)
    solv.drop(columns=["polarisabilite", "Unnamed: 9"], inplace=True)
    solvents = [np.array(solv.loc[solvent]) for solvent in df["Solvent"]]
    col_solv = solv.columns.to_list()
    
    # dft description of ligands 
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
            print(smi)
    ligs["can_rdkit"] = canon_rdkit
    ligs.set_index("can_rdkit", inplace=True)
    ligands = [np.array(ligs.loc[ligand]) for ligand in df["Ligand effectif"]]
    col_lig = add_suffix(ligs.columns.to_list(), 'lig')
    
    # dft description for suubstrates
    substrate = pd.read_csv(data_path + "substrate_dft.csv", sep = ',', index_col=0)
    substrate = substrate[substrate.duplicated(keep='first') != True]
    substrate.drop(columns=descritpors_to_remove_lig, inplace=True)
    canon_rdkit = [Chem.CanonSmiles(smi_co) for smi_co in substrate.index.to_list() ]
    substrate["can_rdkit"] = canon_rdkit
    substrate.set_index("can_rdkit", inplace=True)
    substrates = [np.array(substrate.loc[sub]) for sub in df["Reactant Smile (C-O)"]]
    col_sub = add_suffix(substrate.columns.to_list(), 'sub')
    
    # dft description for AX
    AX = pd.read_csv(data_path + "AX_dft.csv", sep = ',', index_col=0)
    AX.drop(columns=descritpors_to_remove_ax, inplace=True)
    canon_rdkit = [Chem.CanonSmiles(smi_co) for smi_co in AX.index.to_list() ]
    AX["can_rdkit"] = canon_rdkit
    AX.set_index("can_rdkit", inplace=True)
    AXs = [np.array(AX.loc[ax]) for ax in df["A-X effectif"]]
    col_ax = add_suffix(AX.columns.to_list(), 'ax')
    
    
    ohe_precursors = one_hot_encoding_with_names(np.array([precursor_mapping(precursor) for precursor in df["Precurseur Nickel"]]).reshape(-1, 1))
    precursors = ohe_precursors[0]
    col_prec = ohe_precursors[1]
    ohe_additives = one_hot_encoding_with_names(np.array([additives_mapping(precursor) for precursor in df["Base/additif après correction effective"]]).reshape(-1, 1))
    additives = ohe_additives[0]
    col_add = ohe_additives[1]
    
    X = []

    for i, row in df.iterrows():
        yield_isolated = process_yield(row["Isolated Yield"])
        yield_gc = process_yield(row['GC/NMR Yield'])
        # If both yields are known, we keep the isolated yield
        if yield_gc:
            y = yield_gc
        if yield_isolated:
            y = yield_isolated
        
        feature_vector = np.concatenate((solvents[i], ligands[i], precursors[i], additives[i], substrates[i], AXs[i], np.array([y])))
        X.append(feature_vector)
        
    columns = np.concatenate((col_solv, col_lig, col_prec, col_add, col_sub, col_ax, np.array(["yield"])))
    df = pd.DataFrame(data=X, columns=columns)
    
    return df