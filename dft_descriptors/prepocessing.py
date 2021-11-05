from rdkit import Chem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

# returns a dataframe with out the reviews and the doyble step reactions.
# all the smiles are canonized for ligand, substrate, ax and base_add
def preprocess(df):
    # to be remove soon :
    # df = df.drop(df[df["Ligand effectif"]=='[C]1N(C23CC4CC(CC(C4)C2)C3)C=CN1C12CC3CC(CC(C3)C1)C2'].index)
    # df = df.reset_index(drop=True)
    
    # remove lines with nan substrate
    df = df[df["Reactant Smile (C-O)"].isna() == False]
    
    # remove reviews from dataframe:
    df = df[df["Mechanism"] != 'Review']
    
    # remove the doi : 'https://doi.org/10.1021/acs.orglett.5b03151' because it is a double reaction for dimerisation
    df = df[df["DOI"] != 'https://doi.org/10.1021/acs.orglett.5b03151']
    
    # 'https://doi.org/10.1021/acs.orglett.6b03861' ? intramoleculaire oxydative addition ?
    
    # remove the double steps reactions 
    df = df[df["2 Steps"] != "Yes"]
    
    # check smiles validity
    # Canon CO
    co_can = [Chem.CanonSmiles(smi) for smi in df["Reactant Smile (C-O)"]]
    # Canon AX
    ax_can = [Chem.CanonSmiles(smi) for smi in df["A-X effectif"]]
    # Canon Lig
    lig_can = []
    for lig in df["Ligand effectif"]:
        try:
            lig_can.append(Chem.CanonSmiles(dict_ligand[lig]))
        except:
            lig_can.append(dict_ligand[str(lig)])
    
    #print(lig_can)
            
    # Canon Base
    add_can = smiles_additifs(df["Base/additif après correction effective"])
            
    # Canon_df
    df["Reactant Smile (C-O)"] = co_can
    df["A-X effectif"] = ax_can
    df["Ligand effectif"] = lig_can
    df["Base/additif après correction effective"] = add_can
    
    df = df[df["Ligand effectif"] != '[C]1N(C23CC4CC(CC(C4)C2)C3)C=CN1C12CC3CC(CC(C3)C1)C2']
    df = df[df["A-X effectif"] != "[Li][Zn]([Li])(C)(C)(C)c1ccc(C(=O)N(C(C)C)C(C)C)cc1"]
    df = df[df["A-X effectif"] != "[Na+].c1ccc([B-](c2ccccc2)(c2ccccc2)c2ccccc2)cc1" ] 
    df = df[df["Reactant Smile (C-O)"] != "COc1ccc(I)cc1" ] 
    df = df.reset_index(drop=True)
    return df

# Maps an additive to its category
def additives_mapping(add):
    add = str(add)
    add = add.replace('[Sc+++]', '[Sc+3]').replace('[Ti++++]', '[Ti+4]').replace('[Al+++]', '[Al+3]').replace('[Fe+++]', '[Fe+3]').replace('[HO-]', '[O-]')
    if Chem.MolFromSmiles(add):
        return Chem.CanonSmiles(add)
    else:
        return 'nan'

# Maps an additive to its category for the entire list   
def smiles_additifs(liste_additif) :
    base_additif = []
    for i in liste_additif :
        base_additif.append(additives_mapping(i))
    return base_additif

            
dict_solvants = {'(EtO)2CH2': 'CCOCOCC',
 '(EtO)2CH2 + Et2O': 'CCOCOCC.CCOCC',
 'CH3CN': 'CC#N',
 'CPME': 'COC1CCCC1',
 'DCE': '[Cl]CC[Cl]',
 'DMA': 'CC(=O)N(C)C',
 'DME': 'COCCOC',
 'DMF': 'C(=O)N(C)C',
 'DMSO': 'CS(=O)C',
 'Et2O': 'CCOCC',
 'EtOH': 'CCO',
 'MeOH': 'CO',
 'NMP': 'CN1CCCC1(=O)',
 'THF': 'C1OCCC1',
 'THF + DMA': 'C1OCCC1.CC(=O)N(C)C',
 'THF/DMA': 'C1OCCC1.CC(=O)N(C)C',
 'benzene': 'c1ccccc1',
 'dioxane': 'C1COCCO1',
 'dioxane - H2O': 'C1COCCO1.O',
 'hexane': 'CCCCCC',
 'iPr2O': 'CC(C)OC(C)C',
 'iPrOH': 'OC(C)C',
 'm-xylene': 'Cc1cc(C)ccc1',
 'nBu2O': 'CCCCOCCCC',
 'o-xylene': 'Cc1c(C)cccc1',
 'p-xylene': 'Cc1ccc(C)cc1',
 'sBuOH': 'CC(O)CC',
 't-amyl alcohol': 'CC(O)(C)CC',
 'tAmOMe': 'CCC(C)(C)OC',
 'tAmOMe + Et2O': 'CCC(C)(C)OC.CCOCC',
 'tBuOH': 'C(C)(C)(C)O',
 'tBuOH + H2O': 'C(C)(C)(C)O.O',
 'tBuOMe': 'C(C)(C)(C)OC',
 'toluene': 'c1ccccc1C',
 'toluene - H2O': 'c1ccccc1C.O'}           
            
dict_ligand = {'nan': 'NoLigand',
                   #Phosphines
 'PCy3': 'C1CCC(P(C2CCCCC2)C2CCCCC2)CC1',
 'PCy2(1,2-biPh)': 'c1ccc(-c2ccccc2P(C2CCCCC2)C2CCCCC2)cc1',
 'PCy2(1,2-biPhN)': 'CN(C)c1ccccc1-c1ccccc1P(C1CCCCC1)C1CCCCC1',
 'PPhCy2': 'c1ccc(P(C2CCCCC2)C2CCCCC2)cc1',
 'PhPCy2': 'c1ccc(P(C2CCCCC2)C2CCCCC2)cc1',
 'CC(O)c1ccccc1P(c2ccccc2)c3ccccc3': 'CC(O)c1ccccc1P(c1ccccc1)c1ccccc1',
 't-BuPCy2': 'CC(C)(C)P(C1CCCCC1)C1CCCCC1',
 'PCp3': 'C1=CC(P(C2C=CC=C2)C2C=CC=C2)C=C1',
 'PPh3': 'c1ccc(P(c2ccccc2)c2ccccc2)cc1',
 'P(o-tolyl)3': 'Cc1ccccc1P(c1ccccc1C)c1ccccc1C',
 'P(nBu)3': 'CCCCP(CCCC)CCCC',
 'P(tBu)3': 'CC(C)(C)P(C(C)(C)C)C(C)(C)C',
 'P(OMe)3': 'COP(OC)OC',
 'P(CH2Ph)3': 'c1ccc(CP(Cc2ccccc2)Cc2ccccc2)cc1',
 'P(p-OMePh)3': 'COc1ccc(P(c2ccc(OC)cc2)c2ccc(OC)cc2)cc1',
 'PMe3': 'CP(C)C',
 'PEt3': 'CCP(CC)CC',
 'PiPr3': 'CC(C)P(C(C)C)C(C)C',
 'PiBu3': 'CC(C)CP(CC(C)C)CC(C)C',
 'PBu3': 'CCCCP(CCCC)CCCC',
 'PMetBu': 'CP(C(C)(C)C)C(C)(C)C',
 'JohnPhos': 'CC(C)(C)P(c1ccccc1-c1ccccc1)C(C)(C)C',
 'CyJohnPhos': 'c1ccc(-c2ccccc2P(C2CCCCC2)C2CCCCC2)cc1',
 'CyDPEphos': 'c1cc2c(c(P(C3CCCCC3)C3CCCCC3)c1)Oc1c(cccc1P(C1CCCCC1)C1CCCCC1)C2',
 'Xantphos': 'CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21',
 'CyXantphos': 'CC1(C)c2cccc(P(C3CCCCC3)C3CCCCC3)c2Oc2c(P(C3CCCCC3)C3CCCCC3)cccc21',
 'XPhos': 'CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1',
 'RuPhos': 'CC(C)Oc1cccc(OC(C)C)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1',
 'SPhos': 'COc1cccc(OC)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1',
 'Tris(2-methoxyphenyl)phosphine': 'COc1ccccc1P(c1ccccc1OC)c1ccccc1OC',
 'Tris(4-trifluoromethylphenyl) phosphine': 'FC(F)(F)c1ccc(P(c2ccc(C(F)(F)F)cc2)c2ccc(C(F)(F)F)cc2)cc1',
 'PMetBu2': 'CP(C(C)(C)C)C(C)(C)C',
 'PPh2Cy': 'c1ccc(P(c2ccccc2)C2CCCCC2)cc1',
 'P(p-tolyl)3': 'Cc1ccc(P(c2ccc(C)cc2)c2ccc(C)cc2)cc1',
 'P(C6F5)3': 'Fc1c(F)c(F)c(P(c2c(F)c(F)c(F)c(F)c2F)c2c(F)c(F)c(F)c(F)c2F)c(F)c1F',
 'P(NMe2)3': 'CN(C)P(N(C)C)N(C)C',
 'C1CCCC1P(C2CCCC2)c3cc(c4c(C(C)C)cc(C(C)C)cc4(C(C)C))cc(c4c(C(C)C)cc(C(C)C)cc4(C(C)C))c3': 'CC(C)c1cc(C(C)C)c(-c2cc(-c3c(C(C)C)cc(C(C)C)cc3C(C)C)cc(P(C3CCCC3)C3CCCC3)c2)c(C(C)C)c1',
                     #di-phosphines
 'c6ccc5c(P(C1CCCCC1)C2CCCCC2)c(P(C3CCCCC3)C4CCCCC4)sc5c6': 'c1ccc2c(P(C3CCCCC3)C3CCCCC3)c(P(C3CCCCC3)C3CCCCC3)sc2c1',
 'c5cc(P(C1CCCCC1)C2CCCCC2)c(P(C3CCCCC3)C4CCCCC4)s5': 'c1cc(P(C2CCCCC2)C2CCCCC2)c(P(C2CCCCC2)C2CCCCC2)s1',
 'c7ccc(c6cc(c1ccccc1)n(c2ccccc2NC(c3ccccc3)P(c4ccccc4)c5ccccc5)n6)cc7': 'c1ccc(-c2cc(-c3ccccc3)n(-c3ccccc3NC(c3ccccc3)P(c3ccccc3)c3ccccc3)n2)cc1',
 'CC(C)P(C(C)C)C(Nc1ccccc1n3nc(c2ccccc2)cc3c4ccccc4)c5ccccc5': 'CC(C)P(C(C)C)C(Nc1ccccc1-n1nc(-c2ccccc2)cc1-c1ccccc1)c1ccccc1',
 'c7ccc(c6cc(c1ccccc1)n(c2ccccc2NC(c3ccccc3)P(C4CCCCC4)C5CCCCC5)n6)cc7': 'c1ccc(-c2cc(-c3ccccc3)n(-c3ccccc3NC(c3ccccc3)P(C3CCCCC3)C3CCCCC3)n2)cc1',
 'C3CCC(P(C1CCCCC1)C2CCCCC2)CC3': 'C1CCC(P(C2CCCCC2)C2CCCCC2)CC1',
 'CC(C)c5cc(C(C)C)c(c4cc(c1c(C(C)C)cc(C(C)C)cc1C(C)C)cc(P(C2CCCC2)C3CCCC3)c4)c(C(C)C)c5': 'CC(C)c1cc(C(C)C)c(-c2cc(-c3c(C(C)C)cc(C(C)C)cc3C(C)C)cc(P(C3CCCC3)C3CCCC3)c2)c(C(C)C)c1',
 'CC(C)c5cc(C(C)C)c(c4ccc(c1c(C(C)C)cc(C(C)C)cc1C(C)C)c(P(C2CCCC2)C3CCCC3)c4)c(C(C)C)c5': 'CC(C)c1cc(C(C)C)c(-c2ccc(-c3c(C(C)C)cc(C(C)C)cc3C(C)C)c(P(C3CCCC3)C3CCCC3)c2)c(C(C)C)c1',
 'dppe': 'c1ccc(P(CCP(c2ccccc2)c2ccccc2)c2ccccc2)cc1',
 'depe': 'CCP(CC)CCP(CC)CC',
 'dppp': 'c1ccc(P(CCCP(c2ccccc2)c2ccccc2)c2ccccc2)cc1',
 'dppb': 'c1ccc(P(CCCCP(c2ccccc2)c2ccccc2)c2ccccc2)cc1',
 'dppf': 'c1ccc(P(c2ccccc2)C23C4C5C6C2[Fe]56432789C3C2C7C8(P(c2ccccc2)c2ccccc2)C39)cc1',
 'dippf': 'CC(C)P(C(C)C)C12C3C4C5C1[Fe]45321678C2C1C6C7(P(C(C)C)C(C)C)C28',
 'dppf-Ipr': 'CCCP(CCC)C12C3C4C5C1[Fe]45321678C2C1C6C7(P(CCC)CCC)C28',
 'dppf-tBu': 'CC(C)(C)P(C(C)(C)C)C12C3C4C5C1[Fe]45321678C2C1C6C7(P(C(C)(C)C)C(C)(C)C)C28',
 'dppf-Cy': 'C1CCC(CC1)P(C12C3[Fe]4567892(C1C5C34)C1C6C7C9(C81)P(C1CCCCC1)C1CCCCC1)C1CCCCC1',
 'dcypf': 'C1CCC(CC1)P(C12C3[Fe]4567892(C1C5C34)C1C6C7C9(C81)P(C1CCCCC1)C1CCCCC1)C1CCCCC1',
 'dcype': 'C1CCC(P(CCP(C2CCCCC2)C2CCCCC2)C2CCCCC2)CC1',
 'dcypbz': 'c1ccc(P(C2CCCCC2)C2CCCCC2)c(P(C2CCCCC2)C2CCCCC2)c1',
 'dcypt': 'c1scc(P(C2CCCCC2)C2CCCCC2)c1P(C1CCCCC1)C1CCCCC1',
 'DCYPT': 'c1scc(P(C2CCCCC2)C2CCCCC2)c1P(C1CCCCC1)C1CCCCC1',
 'dcypb': 'C1CCC(P(CCCCP(C2CCCCC2)C2CCCCC2)C2CCCCC2)CC1',
 'dmpe': 'CP(C)CCP(C)C',
 'rac-BINAP': 'c1ccc(P(c2ccccc2)c2ccc3ccccc3c2-c2c(P(c3ccccc3)c3ccccc3)ccc3ccccc23)cc1',
 'L1': 'C1CC(P(CCP(C2CCC2)C2CCC2)C2CCC2)C1',
 'L2': 'C1CCC(P(CCP(C2CCCC2)C2CCCC2)C2CCCC2)C1',
 'L3': 'C1CCCC(P(CCP(C2CCCCCC2)C2CCCCCC2)C2CCCCCC2)CC1',
 'L4': 'CC(C)P(CCP(C(C)C)C(C)C)C(C)C',
 'L5': 'CC(C)(C)P(CCP(C(C)(C)C)C(C)(C)C)C(C)(C)C',
 'L6': 'c1ccc(P(CCP(C2CCCCC2)C2CCCCC2)c2ccccc2)cc1',
                 # NHC
 'ItBu': 'CC(C)(C)N1[C]N(C(C)(C)C)C=C1',
 'ICy': '[C]1N(C2CCCCC2)C=CN1C1CCCCC1',
 'IPr': 'CC(C)c1cccc(C(C)C)c1N1[C]N(c2c(C(C)C)cccc2C(C)C)C=C1',
 'IMes': 'Cc1cc(C)c(N2[C]N(c3c(C)cc(C)cc3C)C=C2)c(C)c1',
 'IAd': '[C]1N(C23CC4CC(CC(C4)C2)C3)C=CN1C12CC3CC(CC(C3)C1)C2',
 'I(2-Ad)': '[C]1N(C2C3CC4CC(C3)CC2C4)C=CN1C1C2CC3CC(C2)CC1C3',
 'SIPr': 'CC(C)c1cccc(C(C)C)c1N1[C]N(c2c(C(C)C)cccc2C(C)C)CC1',
 'SIMes': 'Cc1cc(C)c(N2[C]N(c3c(C)cc(C)cc3C)CC2)c(C)c1',
 'SItBu': 'CC(C)(C)N1[C]N(C(C)(C)C)CC1',
 'CDC': 'CC(C)N1c2ccccc2N(C)C1[C]C1N(C)c2ccccc2N1C(C)C',
 'C1-CDC': 'CC(C)N1c2ccccc2N(C(C)C)C1[C]C1N(C)c2ccccc2N1C',
 'Me2IPr': 'CC1=C(C)N(c2c(C(C)C)cccc2C(C)C)[C]N1c1c(C(C)C)cccc1C(C)C',
 'CCCCN5CN(c3cccc(N2CN(CCCC)c1ccccc12)n3)c4ccccc45': 'CCCCN1[C]N(c2cccc(N3[C]N(CCCC)c4ccccc43)n2)c2ccccc21',
 'c2c[n+](C1CCCCC1)cn2C3CCCCC3.[Cl-]': '[Cl-].c1c[n+](C2CCCCC2)cn1C1CCCCC1',
 'IIpr-HCl': 'CC(C)n1cc[n+](C(C)C)c1.[Cl-]',
 'IIPr': 'CC(C)N1[C]N(C(C)C)C=C1',
 '(IMe)2-2HBr': 'Cn1cc[n+](C[n+]2ccn(C)c2)c1.[Br-].[Br-]',
 '(IMe)2': 'CN1[C]N(CN2[C]N(C)C=C2)C=C1',
 'IPrIMeIIPr-2HBr': 'CC(C)n1cc[n+](C[n+]2ccn(C(C)C)c2)c1.[Br-].[Br-]',
 'IPrIMeIIPr': 'CC(C)N1[C]N(CN2[C]N(C(C)C)C=C2)C=C1',
 'ItBuIMeIItBu-2HBr': 'CC(C)(C)n1cc[n+](C[n+]2ccn(C(C)(C)C)c2)c1.[Br-].[Br-]',
 'ItBuIMeIItBu': 'CC(C)(C)N1[C]N(CN2[C]N(C(C)(C)C)C=C2)C=C1',
 'ICyIMeIICy-2HBr': '[Br-].[Br-].c1c[n+](C[n+]2ccn(C3CCCCC3)c2)cn1C1CCCCC1',
 'ICyIMeIICy': '[C]1N(CN2[C]N(C3CCCCC3)C=C2)C=CN1C1CCCCC1',
 'CC(C)(C)n2cc[n+](n1cc[n+](C(C)(C)C)c1)c2.[Br-].[Br-]': 'CC(C)(C)n1cc[n+](-n2cc[n+](C(C)(C)C)c2)c1.[Br-].[Br-]',
                  # phen/bipy
 'phen': 'c1cnc2c(c1)ccc1cccnc12',
 'bipy': 'c1ccc(-c2ccccn2)nc1',
                  # NHC + Phosphine
 'IPr+PPh3': 'CC(C)c1cccc(C(C)C)c1N1C=CN(c2c(C(C)C)cccc2C(C)C)C1.c1ccc(P(c2ccccc2)c2ccccc2)cc1',
 'PPh3+ItBu': 'CC(C)(C)N1C=CN(C(C)(C)C)C1.c1ccc(P(c2ccccc2)c2ccccc2)cc1',
 'PPh3+IPr': 'CC(C)c1cccc(C(C)C)c1N1C=CN(c2c(C(C)C)cccc2C(C)C)C1.c1ccc(P(c2ccccc2)c2ccccc2)cc1',
 'PCy3+ItBu': 'C1CCC(P(C2CCCCC2)C2CCCCC2)CC1.CC(C)(C)N1C=CN(C(C)(C)C)C1',
 'PCy3+IPr': 'C1CCC(P(C2CCCCC2)C2CCCCC2)CC1.CC(C)c1cccc(C(C)C)c1N1C=CN(c2c(C(C)C)cccc2C(C)C)C1',
 'dppf + PCy3': 'C1CCC(P(C2CCCCC2)C2CCCCC2)CC1.[CH]1[CH][CH][C](P(c2ccccc2)c2ccccc2)[CH]1.[CH]1[CH][CH][C](P(c2ccccc2)c2ccccc2)[CH]1.[Fe]',
                  #autres
 'COD': 'C1=CCCC=CCC1',
 'acac': 'CC(=O)/C=C(/C)[O-]',
}
