import pandas as pd
import os
import sys
sys.path.append('../')

from rdkit import Chem
import dft_descriptors.numbering_CO as nb
import dft_descriptors.prepocessing as pp
from aqc_utils.molecule import molecule
from aqc_utils.db_functions import *
from aqc_utils.openbabel_functions import *


import hashlib
import logging
logging.basicConfig(level=logging.INFO)

    # path for file storage
path = '/Users/julesschleinitz/Desktop/These/Projet_stage_M1_Max/Code/SM_DFT/DFT_for_SM/log_data'
    
def generates_descriptors(mol_df, parameter):
    
    """ parameter indicates the member of the reaction you want to generate descriptors """
    
    # keep the infos on the parameter you need
    tags_coll = db_connect("tags")
    mols_coll = db_connect("molecules")
    log_files_coll = db_connect("log_files")
    
    data_df = pd.read_csv('../data_csv/Data_test10222021.csv', sep = ',') 
    
    if parameter == "substrate":
        unik_smi = np.unique(data_df["Reactant Smile (C-O)"].tolist())
        can_smis = np.unique([Chem.CanonSmiles(smi) for smi in unik_smi])
        num_df = pd.read_csv('../data_csv/fragments_0-7.csv')
        
    elif parameter == "ligand":
        data_df = data_df[data_df['Ligand effectif'].notna()]
        unik_lig = [pp.dict_ligand[i] for i in np.unique(data_df['Ligand effectif'])]
        can_smis = np.unique([Chem.CanonSmiles(smi) for smi in unik_lig])
        num_df = pd.read_csv("../data_csv/num_ligands.csv")
        
    # drop molecules that you don't want
    mol_sub_df = drop_non_needed_mols(mol_df, can_smis)
    
    print("we have ", len(mol_sub_df), " molecules to extract")
    # add fs_name to the table
    mol_sub_df['file_base_name'] = mol_sub_df['can'].map(mol_fs_name)
    X = [Chem.CanonSmiles(list(mol_sub_df["can"])[i]) for i in range(len(mol_sub_df))]
    mol_sub_df['can_rdkit'] = X
    

    # L = [eval(mol_sub_df._ids.tolist()[i])[0] for i in range(len(mol_sub_df))]
    # print(L)
    # get a cursor that iterates over the log files
    
    cursor = log_files_coll.find({'molecule_id' : {"$in": mol_sub_df.molecule_id.to_list()}}, {'log': 1, 'can': 1})

    N = 0
    for l in cursor:
        can, log = l['can'], l['log']
        print(can)
        fs_name = mol_fs_name(can) 
        smi_obabel, smi_shared = get_smis(can, mol_sub_df, num_df)
        shared_to_obabel = shared_to_obabel_idx(smi_shared, smi_obabel, parameter)   
        mol_desc = get_moldescriptors(can, log, fs_name, shared_to_obabel)
        if N == 0:
            full_df = mol_desc
        else:
            full_df = full_df.append(mol_desc)
        N += 1
    
    return full_df     
        
def get_moldescriptors(can, log, fs_name, shared_to_obabel):
    print(fs_name)
    with open(f"{path}/{fs_name}_0.log", "w") as f:
        f.write(log)
        log_name = f"{path}/{fs_name}_0.log"
        extractor = gaussian_log_extractor(log_name)
        extractor.__init__
        mol_descriptors = extractor.get_descriptors()
        dict_global = mol_descriptors['descriptors']
        atom_descriptors = mol_descriptors['atom_descriptors'] 
        dict_atoms = select_at_desc(shared_to_obabel, atom_descriptors)
        dict_global.update(dict_atoms)
        dict_trans = get_transitions(mol_descriptors['transitions'])
        dict_global.update(dict_trans)
        dict_labels = get_labels(shared_to_obabel, mol_descriptors['labels'])
        dict_global.update(dict_labels)
        df = pd.DataFrame.from_dict(dict_global, orient ='index', columns=[f"{can}"]).T
        os.remove(log_name)
    return df

def select_at_desc(shared_to_obabel, atom_descriptors):
    select_at_descriptors = dict()
    for key in atom_descriptors.keys():
        key_d = atom_descriptors[key]
        reduced_descriptor = [key_d[shared_to_obabel[i]] for i in range(8)]
        for i in range(len(reduced_descriptor)):
            select_at_descriptors.update({str(key + f"_{i}"): reduced_descriptor[i]})
    return select_at_descriptors

def get_transitions(transitions):
    transitions_d = dict()
    for key in transitions.keys():
        key_d = transitions[key]
        for i in range(len(key_d)):
            transitions_d.update({str(key + f"_{i}"): key_d[i]})
    return transitions_d

def get_labels(shared_to_obabel, labels):
    ret_labels = dict()
    reduced_labels = [labels[shared_to_obabel[i]] for i in range(8)]
    for i in range(len(reduced_labels)):
        ret_labels.update({f"at_{i}" : reduced_labels[i]})
    return ret_labels  

def drop_non_needed_mols(mol_df, can_smis):
    idx_todrop = []
    can_db = []
    for j, smi in enumerate(mol_df["can"]):
        try:
            can_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi)) 
            if can_smi not in can_smis:
                idx_todrop.append(j)
            else:
                can_db.append((smi, can_smi))
        except:
            idx_todrop.append(j)
        
    good_df = mol_df.drop(axis=0, index=idx_todrop)

    # drop all the molecules that don't have the good calculation setup
    dft_set = {'gaussian_config': {'theory': 'b3lyp',
                               'light_basis_set': '6-31G*',
                               'heavy_basis_set': 'LANL2DZ',
                               'generic_basis_set': 'genecp',
                               'max_light_atomic_number': 36},
               'gaussian_tasks': ['opt b3lyp/6-31G* scf=(xqc,tight)',
                                  'freq b3lyp/6-31G* volume NMR pop=NPA density=current Geom=AllCheck Guess=Read',
                                  'TD(NStates=10, Root=1) b3lyp/6-31G* volume pop=NPA density=current Geom=AllCheck Guess=Read'],
               'max_num_conformers': 1,
               'class': '',
               'subclass': '',
               'type': '',
               'subtype': ''}

    idx_todrop = []
    for j, dft in enumerate(good_df["metadata"]):
        if dft != dft_set:
        #if eval(dft)['gaussian_config'] != dft_set['gaussian_config']:
            idx_todrop.append(good_df.iloc[[j]].index[0])

    good_df = good_df.drop(axis=0, index=idx_todrop)
    return good_df


# return smi_shared and smi_obabel for smi from dataframe
def get_smis(smi, obabel_df, num_df):
    smi_obabel = list(obabel_df["can"])[list(obabel_df["can_rdkit"]).index(Chem.CanonSmiles(smi))]
    smi_shared = list(num_df["C0C7_num"])[list(num_df["react"]).index(Chem.CanonSmiles(smi))]
    return smi_obabel, smi_shared

# returns the indexes of the autqchem atoms to extract for CO substrates
def shared_to_obabel_idx(smi_shared, smi_obabel, parameter):
    s_t_r = shared_to_rdkit_can(smi_shared)
    r_t_o = rdkit_to_obabel_can(smi_obabel)
    s_t_o = []
    if parameter == 'substrate':
        for i in range(8):
            idx_rdkit = s_t_r[i][1]
            idx_obabel = r_t_o[idx_rdkit][1]
            #couple = (i, idx_obabel)
            s_t_o.append(idx_obabel)
    elif parameter == 'ligand':
        if smi_L_type(smi_obabel) == 'NHC':
            for i in range(7):
                idx_rdkit = s_t_r[i][1]
                idx_obabel = r_t_o[idx_rdkit][1]
                #couple = (i, idx_obabel)
                s_t_o.append(idx_obabel)
            idx_rdkit = s_t_r[8][1]
            idx_obabel = r_t_o[idx_rdkit][1]
            s_t_o.append(idx_obabel)
        elif smi_L_type(smi_obabel) == 'Phos':
            for i in range(4):
                idx_rdkit = s_t_r[i][1]
                idx_obabel = r_t_o[idx_rdkit][1]
                #couple = (i, idx_obabel)
                s_t_o.append(idx_obabel)
                s_t_o.append(idx_obabel)
        elif smi_L_type(smi_obabel) == 'DiPhos':
            for i in range(8):
                idx_rdkit = s_t_r[i][1]
                idx_obabel = r_t_o[idx_rdkit][1]
                #couple = (i, idx_obabel)
                s_t_o.append(idx_obabel)
        else:
            s_t_o = [i for i in range(8)]
            
    return s_t_o

def shared_to_rdkit_can(smi_shared):
    atmaptidx = [] 
    m = Chem.MolFromSmiles(smi_shared)
    for a in m.GetAtoms():
        a.SetProp("foo", str(a.GetAtomMapNum()))
    nb.remove_at_map(m)
    # get the atoms in the smiles string order
    order = m.GetPropsAsDict(True,True)["_smilesAtomOutputOrder"]
    # print("canonical order:", list(order))
    m_canonical = Chem.RenumberAtoms(m, order)
    for a in m_canonical.GetAtoms():
        atmaptidx.append((int(a.GetProp("foo")), a.GetIdx()))
    atmaptidx.sort(key=takeFirst)
    return atmaptidx

def rdkit_to_obabel_can(smi_obabel):
    atmaptidx = [] 
    m = Chem.MolFromSmiles(smi_obabel)
    for a in m.GetAtoms():
        a.SetProp("foo", str(a.GetIdx()))
    nb.remove_at_map(m)
    # get the atoms in the smiles string order
    order = m.GetPropsAsDict(True,True)["_smilesAtomOutputOrder"]
    # print("canonical order:", list(order))
    m_canonical = Chem.RenumberAtoms(m, order)
    for a in m_canonical.GetAtoms():
        atmaptidx.append((a.GetIdx(), int(a.GetProp("foo"))))
    return atmaptidx

def smi_L_type(smi):
    m = Chem.MolFromSmiles(smi)
    if m.HasSubstructMatch(Chem.MolFromSmiles('N[C]N')):
        typ = 'NHC'
    elif m.HasSubstructMatch(Chem.MolFromSmiles('P')):
        typ = 'Phos'
        if len(m.GetSubstructMatches(Chem.MolFromSmiles('P'))) == 2:
            typ = 'DiPhos'
    else:
        typ = 'other'
    return typ

# helper function to create file names the same way ACQ does
def mol_fs_name(can):
    mol = input_to_OBMol(can, "string", "smi")
    return f"{mol.GetFormula()}_{hashlib.md5(can.encode()).hexdigest()[:4]}"

def takeFirst(elem):
    return elem[0]