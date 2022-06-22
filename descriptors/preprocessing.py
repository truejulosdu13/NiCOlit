import numpy as np
from rdkit import Chem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)
from descriptors.dictionnaries import dict_solvants, dict_ligand, Lewis_Acids_to_drop

def preprocess(df, remove_small_doi=True):
    """Preprocesses the dataframe as described in the article : reference.
    ### 1.None substrates are removed.
    ### 2.Reaction extracted from Chemical Reviews are removed.
    ### 3.Reaction extracted from https://doi.org/10.1021/acs.orglett.5b03151 are removed.
    ### 4.Double Step Reactions are removed.
    ### 5.Potential Lewis Acid reagent in the reaction are identified and a Lewis Acid category is set up.
    ### 6.All SMILES chain are written as RDKit canonical SMILES.
    ### 7.Unfeaturized molecules are removed.
    ### 8.Remove with less than 20 datapoints after previous preprocessing stages
    
            Parameters:
                    df (dataframe): dataframe obtain from the NiCOLit csv file
            Returns:
                    df (dataframe): preprocessed dataframe       
    """
 
    # 1
    df = df[df["substrate"].isna() == False]
    # 2
    df = df[df["review"] != 'Review'] 
    # 3
    df = df[df["DOI"] != 'https://doi.org/10.1021/acs.orglett.5b03151']
    # 4
    df = df[df["2_steps"] != "Yes"]
    # 5
    df = find_Lewis_Acid(df)
    # 6
        # substrate
    co_can = [Chem.CanonSmiles(smi) for smi in df["substrate"]]
        # coupling partner
    ax_can = [Chem.CanonSmiles(smi) for smi in df["effective_coupling_partner"]]
        # ligand
    lig_can = []
    for lig in df["effective_ligand"]:
        try:
            lig_can.append(Chem.CanonSmiles(dict_ligand[lig]))
        except:
            lig_can.append(dict_ligand[str(lig)])
            
        # base-reagents
    add_can = smiles_additifs(df["effective_reagents"])
        # Lewis acid
    al_can = []
    for al in [additives_mapping(al) for al in df["Lewis Acid"]]:
        try: 
            al_can.append(Chem.CanonSmiles(al))
        except:
            al_can.append(al)
            
        # full dataframe
    df["substrate"] = co_can
    df["effective_coupling_partner"] = ax_can
    df["effective_ligand"] = lig_can
    df["reagents"] = add_can
    df["Lewis Acid"] = al_can
    
    # 7
    df = df[df["effective_ligand"] != '[C]1N(C23CC4CC(CC(C4)C2)C3)C=CN1C12CC3CC(CC(C3)C1)C2']
    df = df[df["effective_coupling_partner"] != "[Li][Zn]([Li])(C)(C)(C)c1ccc(C(=O)N(C(C)C)C(C)C)cc1"]
    df = df[df["effective_coupling_partner"] != "[Na+].c1ccc([B-](c2ccccc2)(c2ccccc2)c2ccccc2)cc1" ]
    df = df[df["substrate"] != "COc1ccc(I)cc1" ] 
    df["Lewis Acid"] = df["Lewis Acid"].fillna('NoLewisAcid')
    df["Lewis Acid"] = df["Lewis Acid"].replace('nan', 'NoLewisAcid')
    for al in Lewis_Acids_to_drop:
        df = df[df["Lewis Acid"] != al]
    
    df = df.reset_index(drop=True)
    
    # 8.
    if remove_small_doi==True:
        vc = df.DOI.value_counts()
        doi_above_20 = np.array(vc[vc > 20].index)
        indexes = []
        for i, row in df.iterrows():
            if row["DOI"] not in doi_above_20:
                indexes.append(i)
        df = df.drop(indexes)
        df = df.reset_index(drop=True)

    return df



def find_Lewis_Acid(df):
    """ Splits additives raw information into Lewis Acids and Bases.
    1. Search for the potential Lewis Acids in the base additive section.
    2. Search for a Lewis Acid in the coupling partner section.
    3. Select best Lewis Acid if more than one candidate appears.
    
            Parameters:
                    df (dataframe): dataframe obtain from the NiCOLit csv file  
            Dict or List User Defined :
                    no_lewis_acid (list)     : list of non Lewis Acid reagents.
                    dict_non_charge_al (dict): dict with a selction rule when multiple Lewis Acids are encountered.
            Returns:
                    df (dataframe): with a new "Lewis Acid" column 
    """
    
    AL = [] 
    # first we find the Lewis Acid for each reaction.
    for i, row in df.iterrows():
        # is there a Lewis Acid in the covalent Lewis Acid column ?
        base = row["effective_reagents_covalent"] 
        al = None
        # is there a Lewis Acid in the ionic Lewis Acid column ?
        if isNaN(base): 
            base = row["effective_reagents"]
            try:
                if Chem.CanonSmiles(base) in no_lewis_acid:
                    base = 'NoLewisAcid'
            except:
                pass
        
        # in case there are no additives, the stronger Lewis Acid may be the coupling partner
        if isNaN(base) or base == 'NoLewisAcid': 
            meca = row["Mechanism"]
            # when there is no LA added in the mechanism, the stronger lewis acid is the coupling partner
            # we assume that only one Nickel center is involved in the mechanism.
            if meca in ['Murahashi', 'Kumada', 'Negishi', 'Al _coupling', 'Suzuki']:
                al = row["effective_coupling_partner"]
                if Chem.CanonSmiles(al) in no_lewis_acid:
                    al = 'NoLewisAcid'
            else:
                al = 'NoLewisAcid'
            AL.append(al) 
            
        else:
            try:
                if Chem.CanonSmiles(base) in no_lewis_acid:
                    AL.append('NoLewisAcid')
                else:
                    AL.append(base)
            except:
                AL.append(base)
    
    # Choosing the good Lewis Acid when more than one candidate are present. 
    new_AL = []
    for al in list(AL) :
        # separates Lewis base from Lewis acid
        als = al.split('.') 
        if len(als) == 1: # in case there is only one Lewis acid
            new_AL.append(al)
        else:
            # when there is no positively charge Lewis Acid : specific rule is applied
            if '+' not in al: 
                new_AL.append(dict_non_charge_al[al])
                
            else: 
                # when there is a cationic specie we take it as the Lewis Acid.
                new_als = []
                for smi in als:
                    if '+' in smi:
                        new_als.append(smi)
                # when there are more than one we prioretize the positively charged one.
                if len(np.unique(new_als)) == 1: 
                    new_AL.append(new_als[0])
                else:
                    # this should not happen
                    print("You have to make a choice between ", new_als)    
                    
    df["Lewis Acid"] = new_AL   
    return df

no_lewis_acid = ['Cc1ccc(Br)cc1',
                'Oc1ccccc1',
                'CCN(CC)CC',
                'CC(C)(C)C(=O)O',
                'O',
                'C1CCC2=NCCCN2CC1',
                'C1=CCCC=CCC1',
                'NoLewisAcid']

# choose the good Lewis Acid when two are available
dict_non_charge_al = {"c1cccc(C)c1[Mg]Br.[Li]Cl" : "[Li]Cl",
                      "c1ccccc1[Mg]Br.Cl[Mg]Cl" : "Cl[Mg]Cl",
                      "c1ccccc1[Mg]Br.[Li]Cl" : "[Li]Cl",
                      "c1ccccc1[Mg]Br.[Cs]Cl" : "[Cs]Cl",
                      "c1ccccc1[Mg]Br.[Sc](OS(=O)(=O)C(F)(F)F)(OS(=O)(=O)C(F)(F)F)OS(=O)(=O)C(F)(F)F" : "[Sc](OS(=O)(=O)C(F)(F)F)(OS(=O)(=O)C(F)(F)F)OS(=O)(=O)C(F)(F)F",
                      "c1ccccc1[Mg]Br.[Ti](OC(C)C)(OC(C)C)OC(C)C" : "[Ti](OC(C)C)(OC(C)C)(OC(C)C)OC(C)C",
                      "c1ccccc1[Mg]Br.Cl[Al](Cl)Cl" : "Cl[Al](Cl)Cl",
                      "F[Cu]F.F[Sr]F" : "F[Sr]F",
                      "F[Cu]F.[Al](C)(C)C" : '[Al](C)(C)C',
                      "F[Cu]F.[Cs]F" : '[Cs]F',
                      "Br[Cu]Br.[Cs]F" : '[Cs]F',
                      'Cl[Cu]Cl.[Cs]F' : '[Cs]F',
                      "[Cu]I.[Cs]F" : '[Cs]F',
                      'CC(=O)O[Cu]OC(=O)C.[Cs]F' : '[Cs]F'
                     }


def find_Lewis_Base(df):
    """ Same idea as find_Lewis_Acid but not available yet
            Parameters:
                    df (dataframe): dataframe obtain from the NiCOLit csv file  
            Returns:
                    df (dataframe): with a new "Lewis Acid" column 
    """
        
    Base = []
    for i, row in df.iterrows():
        base = row["effective_reagents"]
        if isNaN(base): # if there is no base/additives : the base will be the solvent. 
            try:
                # if the solvent is not a mix of solvents:
                base = dict_solvent_to_smiles[row["solvent"]]
            except:
                # in cas of a solvent mix : a choice is made.
                if row["solvent"] == 'tAmOMe + Et2O':
                    base = 'CCOCC'
                elif row["solvent"] == '(EtO)2CH2 + Et2O':
                    base = 'CCOCC'
                elif row["solvent"] == 'THF/DMA' or row["solvent"] == 'THF + DMA':
                    base = dict_solvants['THF']
                
                else:
                    print(row["solvent"])
        Base.append(base)   
        
    # choose good Lewis Base when more than one candidate is present. 
    new_Base = []
    for base in list(Base) :
        bases = base.split('.')
        if len(bases) == 1:
            new_Base.append(base)
        else:
            if '-' not in base:
                print(base)
                
            else: #when there are more than one we prioretize the positively charged one.
                new_bases = []
                for smi in bases:
                    if '-' in smi:
                        new_bases.append(smi)
                if len(np.unique(new_bases)) == 1:
                    new_Base.append(new_bases[0])
                else:
                    new_base = new_bases[0]
                    # needs a ranking between bases.
                    for smi in new_bases:
                        if smi == '[F-]':
                            new_base = smi
                    new_Base.append(new_base)
                    #print("You have to make a choice between ", np.unique(new_bases))    
    
    df["Lewis Base"] = new_Base
    
    # case where df["Lewis Base"] is the same as df["Lewis Acid"]
    for i, row in df.iterrows():
        if row["Lewis Acid"] == row["Lewis Base"]:
            print(row["Lewis Base"])
    
    # numeroter les acides et les bases par atomes :
    # comparer les acides et les bases à nouveau.
    # quand les bases ne possèdent pas de numerotation propre : mettre le solvant à la place.
    
    return df

# Maps an additive to its category
def additives_mapping(add):
    add = str(add)
    add = add.replace('[Sc+++]', '[Sc+3]').replace('[Ti++++]', '[Ti+4]').replace('[Al+++]', '[Al+3]').replace('[Fe+++]', '[Fe+3]').replace('[HO-]', '[O-]')
    if Chem.MolFromSmiles(add):
        return Chem.CanonSmiles(add)
    elif add == 'NoLewisAcid':
        return add
    else:
        return 'nan'

# Maps an additive to its category for the entire list   
def smiles_additifs(liste_additif) :
    base_additif = []
    for i in liste_additif :
        base_additif.append(additives_mapping(i))
    return base_additif

# auxiliary function
def isNaN(num):
    return num != num

            

