from rdkit import Chem
from rdkit.Chem import Draw

## main CO numbering functions ##

# numerote le carbone aromatique 0 et l'oxygène 1
def number_C0O1(mol, CO_frag, ar_frag):
    mol = mol_with_atom_index(mol)
    L = mol.GetSubstructMatches(Chem.MolFromSmarts('cO'))
    if len(L) == 0:
        L = mol.GetSubstructMatches(Chem.MolFromSmarts('co'))
        
    c = None
    for couple in L:
        if couple[0] in mol.GetSubstructMatch(Chem.MolFromSmarts(CO_frag.replace('*', ''))):
            if couple[1] in mol.GetSubstructMatch(Chem.MolFromSmarts(ar_frag.replace('(*)','').replace('*',''))):
                c = couple
        elif couple[0] in mol.GetSubstructMatch(Chem.MolFromSmarts(ar_frag.replace('(*)','').replace('*',''))):
            if couple[1] in mol.GetSubstructMatch(Chem.MolFromSmarts(CO_frag.replace('*', 'c'))):
                c = couple
        
    if c == None:
        print(Chem.MolToSmiles(mol))
        
    else:            
        for at in mol.GetAtoms():
            a = at.GetIdx()
            if a in c and at.GetAtomicNum() == 8:
                reset_atom_map(mol, 1)
                at.SetAtomMapNum(1)
            if a in c and at.GetAtomicNum() == 6:
                reset_atom_map(mol, 0)
                at.SetAtomMapNum(0)
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    return mol
    
# not sure this functions is usefull    ???
def number_at_by_shared_ArOC(mol, ar_frag, CO_frag):
    mol = mol_with_atom_index(mol)
    n_share = 3
    n_other = 8
    shared_frag_1 = Chem.MolFromSmiles(CO_frag.replace('*', 'c8ccccc8'))
    for at in mol.GetAtoms():
        if at.GetIdx() not in mol.GetSubstructMatch(shared_frag_1):
            reset_atom_map(mol, n_other)
            at.SetAtomMapNum(n_other)
            n_other += 1
                    
        elif at.GetIdx() in mol.GetSubstructMatch(Chem.MolFromSmarts(CO_frag.replace('*',''))):
            if mol.HasSubstructMatch(Chem.MolFromSmarts('Oc1ccccc1')):
                if at.GetIdx() in mol.GetSubstructMatch(Chem.MolFromSmarts('Oc1ccccc1')): 
                    if (at.GetIdx(),) in mol.GetSubstructMatches(Chem.MolFromSmiles('O')):
                        reset_atom_map(mol, 1)
                        at.SetAtomMapNum(1)  
                elif at.GetIdx() in mol.GetSubstructMatch(Chem.MolFromSmarts('COc')):
                    reset_atom_map(mol, 2)
                    at.SetAtomMapNum(2) 
                elif at.GetIdx() in mol.GetSubstructMatch(Chem.MolFromSmarts('[Si]Oc')):
                    reset_atom_map(mol, 2)
                    at.SetAtomMapNum(2) 
                elif at.GetIdx() in mol.GetSubstructMatch(Chem.MolFromSmarts('cOc')):
                    reset_atom_map(mol, 2)
                    at.SetAtomMapNum(2) 
                else:
                    reset_atom_map(mol, n_other)
                    at.SetAtomMapNum(n_other)
                    n_other += 1
                    
            elif mol.HasSubstructMatch(Chem.MolFromSmarts('oc1ccccc1')):
                if at.GetIdx() in mol.GetSubstructMatch(Chem.MolFromSmarts('oc1ccccc1')): 
                    if (at.GetIdx(),) in mol.GetSubstructMatches(Chem.MolFromSmiles('o')):
                        reset_atom_map(mol, 1)
                        at.SetAtomMapNum(1)  
                elif at.GetIdx() in mol.GetSubstructMatch(Chem.MolFromSmarts('coc')):
                    reset_atom_map(mol, 2)
                    at.SetAtomMapNum(2) 
                elif at.GetIdx() in mol.GetSubstructMatch(Chem.MolFromSmarts('Coc')):
                    reset_atom_map(mol, 2)
                    at.SetAtomMapNum(2) 
                elif at.GetIdx() in mol.GetSubstructMatch(Chem.MolFromSmarts('[Si]Oc')):
                    reset_atom_map(mol, 2)
                    at.SetAtomMapNum(2) 
                else:
                    reset_atom_map(mol, n_other)
                    at.SetAtomMapNum(n_other)
                    n_other += 1
            
        elif at.GetIdx() in mol.GetSubstructMatch(Chem.MolFromSmarts(ar_frag.replace('(*)','').replace('*',''))):
            if at.GetIdx() in mol.GetSubstructMatch(shared_frag_1):   
                if at.GetIdx() in mol.GetSubstructMatch(Chem.MolFromSmarts('[O:1]c')):
                    reset_atom_map(mol, 0)
                    at.SetAtomMapNum(0)
                
            elif at.GetIdx() in mol.GetSubstructMatch(Chem.MolFromSmarts('Oc1ccccc1')):
                    reset_atom_map(mol, n_share)
                    at.SetAtomMapNum(n_share)
                    n_share += 1
            else:
                print(ar_frag, CO_frag, Chem.MolToSmiles(mol))

    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    verif = verif_num(mol)
    return mol, verif, shared_frag_1

## check potential errors functions ##

def verif_num(mol):
    verif = True
    at_num_list = []
    for at in mol.GetAtoms():
        at_num_list.append(at.GetAtomMapNum())
    for at in mol.GetAtoms():
        if at.GetAtomMapNum() == 1 and at.GetAtomicNum() != 8:
            verif = False
        if at.GetAtomMapNum() == 0 and at.GetAtomicNum() != 6:
            verif = False
    return verif

## small renumbering usefull functions ##

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

def mol_with_atom_num(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetAtomMapNum())
    return mol

def remove_at_map(mol):
    for at in mol.GetAtoms():
        at.SetAtomMapNum(0)
    return Chem.MolFromSmiles(Chem.MolToSmiles(mol))

def list_at_nums(mol):
    at_nums = []
    for i in list(mol.GetAtoms()):
        at_nums.append(i.GetAtomMapNum())
    return at_nums

def reset_atom_map(mol, n):   
    n_max = 0
    for at in mol.GetAtoms():
        if at.GetAtomMapNum() >= n_max :
            n_max = at.GetAtomMapNum() + 1
    for at in mol.GetAtoms():
        if at.GetAtomMapNum() == n:
            at.SetAtomMapNum(n_max)
            
def rescale_atom_map(mol):
    l = list_at_nums(mol)
    L = len(l)
    if max(l) >= L:
        mini = 0
        while mini in l:
            mini += 1
        print(mini)
        for at in mol.GetAtoms():
            if at.GetAtomMapNum() == max(l):
                at.SetAtomMapNum(mini)