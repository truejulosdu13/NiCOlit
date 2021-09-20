from rdkit import Chem
from rdkit.Chem import Draw

## main CO numbering functions ##

# numerote le carbone aromatique 0 et l'oxygène 1
def number_C0O1(mol, CO_frag, ar_frag):
    well_numbered = True
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
        #print(Chem.MolToSmiles(mol))
        well_numbered == False
        
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
    return mol, well_numbered

def number_C2C3(m):
    n_at = 2
    OCC = choose_goods_Occ(m)
    OC = choose_good_OC(m)
    for at in m.GetAtoms():
        l = []
        for j in OCC:
            for k in j:
                if k not in l:
                    l.append(k)
    for at in m.GetAtoms():
        if at.GetIdx() in l:
            if at.GetIdx() not in OC:
                reset_atom_map(m, n_at)
                at.SetAtomMapNum(n_at)
                n_at +=1      

    #m = Chem.MolFromSmarts(Chem.MolToSmarts(m))
    m = Chem.MolFromSmiles(Chem.MolToSmiles(m))
    return(m)

def number_C456(m):
    OBenz = choose_goods_Oc1ccccc1(m)
    Occ_ats = []
    for trouple in choose_goods_Occ(m):
        for k in trouple:
            if k not in Occ_ats:
                Occ_ats.append(k)
    n_at = 4            
    for at in m.GetAtoms():
        if at.GetIdx() in OBenz:
            if at.GetIdx() not in Occ_ats:
                reset_atom_map(m, n_at)
                at.SetAtomMapNum(n_at)
                n_at +=1
            
    m = Chem.MolFromSmiles(Chem.MolToSmiles(m))
    return(m)

def number_7(m):  
    cOC = list(m.GetSubstructMatches(Chem.MolFromSmarts('cOC')))
    for trouple in cOC:
        good = False
        trouple_map = [m.GetAtomWithIdx(at_idx).GetAtomMapNum() for at_idx in trouple]
        if 1 in trouple_map:
            if 0 in trouple_map:
                good = True
        if good == True:
            for at_idx in trouple:
                if m.GetAtomWithIdx(at_idx).GetAtomMapNum() != 1:
                    if m.GetAtomWithIdx(at_idx).GetAtomMapNum() != 0:
                        reset_atom_map(m, 7)
                        m.GetAtomWithIdx(at_idx).SetAtomMapNum(7)
    
    cOc = list(m.GetSubstructMatches(Chem.MolFromSmarts('cOc')))
    for trouple in cOc:
        good = False
        trouple_map = [m.GetAtomWithIdx(at_idx).GetAtomMapNum() for at_idx in trouple]
        if 1 in trouple_map:
            if 0 in trouple_map:
                good = True
        if good == True:
            for at_idx in trouple:
                if m.GetAtomWithIdx(at_idx).GetAtomMapNum() != 1:
                    if m.GetAtomWithIdx(at_idx).GetAtomMapNum() != 0:
                        reset_atom_map(m, 7)
                        m.GetAtomWithIdx(at_idx).SetAtomMapNum(7)
                        
    cOSi = list(m.GetSubstructMatches(Chem.MolFromSmarts('cO[Si]')))
    for trouple in cOSi:
        good = False
        trouple_map = [m.GetAtomWithIdx(at_idx).GetAtomMapNum() for at_idx in trouple]
        if 1 in trouple_map:
            if 0 in trouple_map:
                good = True
        if good == True:
            for at_idx in trouple:
                if m.GetAtomWithIdx(at_idx).GetAtomMapNum() != 1:
                    if m.GetAtomWithIdx(at_idx).GetAtomMapNum() != 0:
                        reset_atom_map(m, 7)
                        m.GetAtomWithIdx(at_idx).SetAtomMapNum(7)
    return m

  
# functions usefull to identify aromatics atoms
def choose_good_OC(m):
    OC = list(m.GetSubstructMatches(Chem.MolFromSmarts('[O:1][c:0]')))
    for couple in OC:
        is_good = True
        for at_idx in couple:
            if m.GetAtomWithIdx(at_idx).GetAtomMapNum() > 1:
                is_good = False
        if is_good == True:
            OC = couple
    return OC

def choose_goods_Occ(m):
    OCC = list(m.GetSubstructMatches(Chem.MolFromSmarts('[O:1][c:0]c')))
    good_Occs = []
    for trouple in OCC:
        if 0 in trouple and 1 in trouple:
            good_Occs.append(trouple)
    return good_Occs

def choose_goods_Oc1ccccc1(m):
    Obenz = list(m.GetSubstructMatches(Chem.MolFromSmarts('Oc1ccccc1')))
    sub_ats = []
    for trouple in choose_goods_Occ(m):
        for k in trouple:
            if k not in sub_ats:
                sub_ats.append(k)
    for OBz in Obenz:
        is_good = True
        for at_idx in sub_ats:
            if at_idx not in OBz:
                is_good = False
        if is_good == True:
            Obenz = OBz
    return Obenz

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

def numbering_arom_fine(mol):
    good = True
    sub_struct = Chem.MolFromSmiles('Oc1ccccc1') 
    idx_matching = mol.GetSubstructMatch(sub_struct)
    map_matching = [mol.GetAtomWithIdx(idx_matching[i]).GetAtomMapNum() for i in range(len(idx_matching))]
    for i in range(7):
        if i not in map_matching:
            good = False
            
    if 0 in map_matching:
        if mol.GetAtomWithIdx(idx_matching[map_matching.index(0)]).GetSymbol() != 'C':
            good = False
    else:
        good = False
        
    if 1 in map_matching:
        if mol.GetAtomWithIdx(idx_matching[map_matching.index(1)]).GetSymbol() != 'O':
            good = False
    else:
        good = False
    return good

def numbering_8_fine(mol):
    good = True
    if mol.HasSubstructMatch(Chem.MolFromSmiles('COc1ccccc1')):
        sub_struct = Chem.MolFromSmiles('COc1ccccc1') 
        idx_matching = mol.GetSubstructMatch(sub_struct)
        map_matching = [mol.GetAtomWithIdx(idx_matching[i]).GetAtomMapNum() for i in range(len(idx_matching))]
        for i in range(8):
            if i not in map_matching:
                good = False
        if good == True:    
            if mol.GetAtomWithIdx(idx_matching[map_matching.index(0)]).GetSymbol() != 'C':
                good = False
            if mol.GetAtomWithIdx(idx_matching[map_matching.index(1)]).GetSymbol() != 'O':
                good = False
            if mol.GetAtomWithIdx(idx_matching[map_matching.index(7)]).GetSymbol() != 'C':
                good = False
    
    elif mol.HasSubstructMatch(Chem.MolFromSmarts('c1ccccc1O[Si]')):
        sub_struct = Chem.MolFromSmarts('[Si]Oc1ccccc1') 
        idx_matching = mol.GetSubstructMatch(sub_struct)
        map_matching = [mol.GetAtomWithIdx(idx_matching[i]).GetAtomMapNum() for i in range(len(idx_matching))]
        for i in range(8):
            if i not in map_matching:
                good = False
        if good == True:    
            if mol.GetAtomWithIdx(idx_matching[map_matching.index(0)]).GetSymbol() != 'C':
                good = False
            if mol.GetAtomWithIdx(idx_matching[map_matching.index(1)]).GetSymbol() != 'O':
                good = False
            if mol.GetAtomWithIdx(idx_matching[map_matching.index(7)]).GetSymbol() != 'Si':
                good = False
                
        
    elif mol.HasSubstructMatch(Chem.MolFromSmarts('cOc1ccccc1')):
        sub_struct = Chem.MolFromSmiles('cOc1ccccc1') 
        idx_matching = mol.GetSubstructMatch(sub_struct)
        map_matching = [mol.GetAtomWithIdx(idx_matching[i]).GetAtomMapNum() for i in range(len(idx_matching))]
        for i in range(8):
            if i not in map_matching:
                good = False
        if good == True:    
            if mol.GetAtomWithIdx(idx_matching[map_matching.index(0)]).GetSymbol() != 'C':
                good = False
            if mol.GetAtomWithIdx(idx_matching[map_matching.index(1)]).GetSymbol() != 'O':
                good = False
            if mol.GetAtomWithIdx(idx_matching[map_matching.index(7)]).GetSymbol() != 'C':
                good = False
                         
    else:
        good = False
    return good

def show_DOIS(mol, df):
    smi_can = Chem.MolToSmiles(remove_at_map(mol)) 
    dois_id = []
    for i, smis in enumerate(df["Reactant Smile (C-O)"]):
        if Chem.MolToSmiles(Chem.MolFromSmiles(smis)) == smi_can:
            dois_id.append(df["DOI"][i])
    dois = []
    for doi in dois_id:
        if doi not in dois:
            dois.append(doi)
    return dois

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
    while max(l) > L:
        mini = 0
        while mini in l:
            mini += 1
        for at in mol.GetAtoms():
            if at.GetAtomMapNum() == max(l):
                at.SetAtomMapNum(mini)
        l = list_at_nums(mol)