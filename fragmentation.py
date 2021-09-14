from rdkit import Chem
from rdkit.Chem import AllChem
import gc

## MAIN FUNCTION ##
#returns a list of smiles fragments list from a reactant and product list

def gen_fragments_on_CO_break(df_column_react, df_column_prod):
    mol_frags = [] #liste des listes de fragments des molécules
    for i, smi in enumerate(df_column_react):
        frags = gen_frag_on_CO_break(smi, df_column_prod[i]) 
        mol_frags.append(frags)
    return mol_frags

def gen_frag_on_CO_break(smi, prod):
    react_w = Chem.rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(smi)) # poids du reactif
    m = Chem.MolFromSmiles(smi)
    bis = m.GetSubstructMatches(Chem.MolFromSmarts('[c][O]'))  # renvoie les couples d'atomes correspondants aux liaisons C-aromatique/Oxygène
    bis = remove_OH_bad_frag(m, bis)
    if len(bis) == 1: # cas où il y a une seule liaison C-aromatique/Oxygène : pas d'ambiguité !
        frags = gen_frags(m, react_w, bis)   
    elif len(bis) == 0: # cas où il n'y a pas de liaison C-aromatique/Oxygène : O est aussi aromatique !
        frags = gen_oc_frags(m, react_w)   
    else: # cas où len(bis) > 1 : il y a au moins deux sites de fragmentation donc il faut faire des choix !
        if m.HasSubstructMatch(Chem.MolFromSmiles('Oc1nc(O)nc(O)n1')): # cas chiant de la publi des triazines
            frags = gen_frag_triaz(m, react_w)    
        else:
            frags = gen_frags(m, react_w, bis)         
    if len(unique_elements(frags)) > 2 :
        frags = rm_non_react_frags(frags, smi, prod, react_w) 
        
    can_frags = []
    for frag in frags:
        can_frags.append(rm_atmap(frag))
        
    gc.collect()
    return can_frags


## small helpfull functions ##

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

def can_dummy_frags(smi):
    Du = Chem.MolFromSmiles('*')
    sub_mol = Chem.MolFromSmiles(smi)
    smi_can = AllChem.ReplaceSubstructs(sub_mol, Du, Chem.MolFromSmiles('*'),True)
    smi_can = AllChem.MolToSmiles(smi_can[0])
    return smi_can
#example smi = 'c1ccc(*)cc1OCC*' -> c1ccc(*)cc1OCC*'
#example smi = 'c1[cH:1][cH:2][c:3]([*:4])[cH:5][c:6]1[O:7][CH2:8][CH2:9][*:10]' 
#.             -> '*[c:3]1[cH:2][cH:1]c[c:6]([O:7][CH2:8][CH2:9]*)[cH:5]1'

def rem_dummy_frags(smi):
    Du = Chem.MolFromSmiles('*')
    sub_mol = Chem.MolFromSmiles(smi)
    smi_can = Chem.MolToSmiles(AllChem.DeleteSubstructs(sub_mol, Du)) 
    return smi_can
#example smi = 'c1ccc(*)cc1OCC*' -> CCOc1ccccc1'

def rm_atmap(smi):
    mol = Chem.MolFromSmiles(smi)
    for at in mol.GetAtoms():
        at.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)
#example smi = '*[O:4][CH2:3][CH2:2][CH:1](O)[c:10]1[c:5](*)[cH:6][cH:7][cH:8][cH:9]1' -> '*OCCC(O)c1ccccc1*'

def unique_elements(liste):
    unik = []
    for i in liste:
        if i not in unik:
            unik.append(i)
    return unik

# liste has to be a liste of fragments in smiles
def verify_frag_wt(liste, react_wt):
    tot_wt = 0
    for smi in liste:
        try:
            m = Chem.MolFromSmiles(smi)
            tot_wt += Chem.rdMolDescriptors.CalcExactMolWt(m)
        except TypeError:
            print(smi)

    if abs(tot_wt-react_wt) < 2:
        return True
    else:
        return False
    
    
    

## fragmentation fonctions depending on the molecule substrutcutures 
def gen_frags(mol, react_w, bis):
    bs = [mol.GetBondBetweenAtoms(x,y).GetIdx() for x,y in bis]
    nm = Chem.MolToSmiles(Chem.FragmentOnBonds(mol,bs))
    can_nm = []
    for smi in nm.split('.'):
        can_nm.append(can_dummy_frags(smi))
    if verify_frag_wt(can_nm, react_w):
        frags = can_nm
    else:
        frags = [0, 0, 0]
        print("Pb in two_frags")
    return frags

def gen_oc_frags(mol, react_w):
    bis = mol.GetSubstructMatches(Chem.MolFromSmarts('[c][o]'))
    bis = remove_OH_bad_frag(mol, bis)
    bs = [mol.GetBondBetweenAtoms(x,y).GetIdx() for x,y in bis]
    if len(bs) > 0 and len(bis) == 1:
        nm = Chem.MolToSmiles(Chem.FragmentOnBonds(mol,bs))
        can_nm = []
        for smi in nm.split('.'):
            can_nm.append(can_dummy_frags(smi))
        if verify_frag_wt(can_nm, react_w):
            frags = can_nm
        else:
            frags = [0, 0 , 0]
            print("Pb in oc_frags_1")
    elif len(bs) > 0 and len(bis) > 1 :
        nm = Chem.MolToSmiles(Chem.FragmentOnBonds(mol,bs))
        can_nm = []
        for smi in nm.split('.'):
            can_nm.append(can_dummy_frags(smi.replace('o','O')))
        if verify_frag_wt(can_nm, react_w):
            frags = can_nm
    else:
        print(len(bis))
        print(Chem.MolToSmiles(mol))
        frags = [0, 0, 0]
        print("Pb in oc_frags_2")

    return frags

def gen_frag_triaz(mol, react_w):             
    good_pairs = []
    # recuperer les indices des carbones de l heterocycle aromatique c1ncncn1
    bis = mol.GetSubstructMatches(Chem.MolFromSmarts('[c][n]'))
    bad_ats = []
    for k in bis:
        for j in k:
            bad_ats.append(j)
    # recupere les indices des paires [c][O] et enlever celles faisant intervenir un [c] de l'heterocycle
    bisbis = mol.GetSubstructMatches(Chem.MolFromSmarts('[c][O]'))
    for k in bisbis:
        if k[0] not in bad_ats and k[1] not in bad_ats:
            good_pairs.append(k) 
            
    # retrait des groupes OH:
    good_pairs = remove_OH_bad_frag(mol, good_pairs)
    
    # fragmentation à partir de ces paires
    bs = [mol.GetBondBetweenAtoms(x,y).GetIdx() for x,y in good_pairs]
    nm = Chem.MolToSmiles(Chem.FragmentOnBonds(mol,bs))
    can_nm = []
    for smi in nm.split('.'):
        can_nm.append(can_dummy_frags(smi))
    if verify_frag_wt(can_nm, react_w):
        frags = can_nm
    else:
        print("PROBLEM")  
    if len(frags) > 3:
        print('blabla')
        # to modify
    return frags

# prends en entrée un bis et retire les index qui correspondrait à une liaison c-OH
def remove_OH_bad_frag(mol, bis):
    if mol.HasSubstructMatch(Chem.MolFromSmarts('[OH]')):
        m = Chem.SanitizeMol(mol)
        mol = mol_with_atom_index(mol)
        bs = [mol.GetBondBetweenAtoms(x,y).GetIdx() for x,y in bis]
        nm = Chem.FragmentOnBonds(mol,bs)
        smis = Chem.MolToSmiles(nm)
        smi_list = smis.split('.')
        at_indx = []
        for smi in smi_list:
            smi_wo_dum = rem_dummy_frags(smi)
        
            if len(list(Chem.MolFromSmarts(smi_wo_dum).GetAtoms())) == 1:
                at_indx.append(Chem.MolFromSmarts(smi_wo_dum).GetAtoms()[0].GetAtomMapNum())
    
        if len(at_indx) == 0:
            return bis
    
        else:
            new_bis = []
            for idx in at_indx:
                for pair in bis:
                    if pair[0] != idx and pair[1] != idx:
                        new_bis.append(pair)

            return new_bis
    else:
        return bis


# enelever les fragments qui correspondent à des O-C aromatique mais qui ne reagissent pas
# PISTE d'AMELIORATION : RECONNAITRE LES STRUCTURES AVEC LES H EXPLICITES POUR POUVOIR DIFFERENCIER OC de OCC...
def rm_non_react_frags(frags, smi_react, smi_prod, react_w):
    prod = Chem.MolFromSmiles(smi_prod)
    r = Chem.MolFromSmiles(smi_react)
    reactive_frags = []
    core_frags = []
    new_frags = []
    # idee = voir si les fragments sont ou non dans le produit
    for frag in frags:
        frag_can = can_dummy_frags(frag).replace('(*)','').replace('*','').replace('**', '*')
        f = Chem.MolFromSmiles(frag_can)
        if frag_can == 'OC' or frag_can == 'CO':
            f = Chem.MolFromSmarts('[O][CH3]')
        if prod.HasSubstructMatch(f) is False:
            reactive_frags.append(can_dummy_frags(frag))
    
    new_frags = reactive_frags
    # get core frag
    for react_frag in unique_elements(reactive_frags):
        react_frag = react_frag.replace('(*)','').replace('*','')
        r_frag = Chem.MolFromSmiles(react_frag)
        cores = Chem.ReplaceSubstructs(r, r_frag, Chem.MolFromSmiles('*'))  
        
        for core in cores:
            if core.HasSubstructMatch(Chem.MolFromSmiles('c1ccccc1')):
                if '.' not in Chem.MolToSmiles(core):
                    r = Chem.MolFromSmiles(Chem.MolToSmiles(core))
        #smi_core = Chem.MolToSmiles(core[0])
        #while core[j].HasSubstructMatch(Chem.MolFromSmiles('c1ccccc1')) is False or '.' in smi_core:
        #    print(Chem.MolToSmiles(core[j]))
        #    j +=1
        #r = core[j]
        
    new_frags.append(Chem.MolToSmiles(r).replace('**','*'))

    if verify_frag_wt(new_frags, react_w):
        return new_frags
        
    else:
        print(smi_react, new_frags, Chem.MolToSmiles(prod), 'wrong_weigth')
        return frags
    
    
    
    
    
    
## GENERATION OF A DICTIONNARY OF OZ and aromatic fragments:

def dicts_aromatic_and_OZ(mol_frags):
    frag_aromatique = {}
    frag_oz = {}
    Du = Chem.MolFromSmiles('*')
    wrong_count = 0

    for i, smi_list in enumerate(mol_frags):
        if len(smi_list) > 0:
            for smi in smi_list:
                sub_mol = Chem.MolFromSmiles(smi)
                if sub_mol != None and sub_mol.HasSubstructMatch(Chem.MolFromSmiles('c1ccccc1')):
                    smi_can = AllChem.ReplaceSubstructs(sub_mol, Du, Chem.MolFromSmiles('*'),True)
                    smi_can = AllChem.MolToSmiles(smi_can[0])
                    if Chem.MolFromSmiles(smi_can).HasSubstructMatch(Chem.MolFromSmiles('*O')) == False:
                        if smi_can not in frag_aromatique:
                            frag_aromatique[smi_can] = len(frag_aromatique)+1
                    elif smi_can not in frag_oz:
                        frag_oz[smi_can] = len(frag_oz)+1
                    # cas des 3-pryridines
                elif sub_mol != None and sub_mol.HasSubstructMatch(Chem.MolFromSmiles('c1cnccc1')):
                    smi_can = AllChem.ReplaceSubstructs(sub_mol, Du, Chem.MolFromSmiles('*'),True)
                    smi_can = AllChem.MolToSmiles(smi_can[0])
                    if Chem.MolFromSmiles(smi_can).HasSubstructMatch(Chem.MolFromSmiles('*O')) == False:
                        if smi_can not in frag_aromatique:
                            frag_aromatique[smi_can] = len(frag_aromatique)+1
                    elif smi_can not in frag_oz:
                            frag_oz[smi_can] = len(frag_oz)+1
                elif sub_mol == None:
                    smi_can = Chem.MolToSmiles(Chem.MolFromSmiles('*c1c(c2c(O)cccc2)cccc1'))
                    if smi_can not in frag_aromatique:
                        frag_aromatique[smi_can] = len(frag_aromatique)+1
                    if smi_can not in frag_oz:
                        frag_oz['*Oc1ccccc1-c1ccccc1(*)'] = len(frag_oz)+1
                    #print(i, smi)
                    wrong_count += 1
                else:
                    smi_can = AllChem.ReplaceSubstructs(sub_mol, Du, Chem.MolFromSmiles('*'),True)
                    smi_can = AllChem.MolToSmiles(smi_can[0])
                    if smi_can not in frag_oz:
                        frag_oz[smi_can] = len(frag_oz)+1  
                
    #print('nombre de molecules qui ne sont pas fragmentées correctement : ', wrong_count)
    return frag_aromatique, frag_oz

## Pour une liste de fragments quelconque, la fonction retourne une liste de fragments avec comme deuxième symbole 'O'

def find_OZ(list_frag):
    for frag in list_frag:
        if frag[1]=='O':
            return frag
    return None

