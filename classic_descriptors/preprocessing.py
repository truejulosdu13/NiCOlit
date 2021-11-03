from rdkit import Chem

# returns a dataframe with out the reviews and the doyble step reactions.
# all the smiles are canonized for ligand, substrate, ax and base_add
def preprocess(df):
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
            lig_can.append(Chem.CanonSmiles(pp.dict_ligand[lig]))
        except:
            lig_can.append(lig)
            
    # Canon Base
    add_can = smiles_additifs(df["Base/additif après correction effective"])
            
    # Canon_df
    df["Reactant Smile (C-O)"] = co_can
    df["A-X effectif"] = ax_can
    df["Ligand effectif"] = lig_can
    df["Base/additif après correction effective"] = add_can
    
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
            
dict_ligand = {
    #Phosphines
          "PCy3" : "P(C1CCCCC1)(C2CCCCC2)C3CCCCC3",
          "PCy2(1,2-biPh)" : "c1ccccc1c2ccccc2P(C3CCCCC3)C4CCCCC4",
          "PCy2(1,2-biPhN)" : "c1cccc(N(C)C)c1c2ccccc2P(C3CCCCC3)C4CCCCC4",
          "PPhCy2" : "P(c1ccccc1)(C2CCCCC2)C3CCCCC3",
          "PhPCy2" : "P(c1ccccc1)(C2CCCCC2)C3CCCCC3",
          "CC(O)c1ccccc1P(c2ccccc2)c3ccccc3" : "CC(O)c1ccccc1P(c2ccccc2)c3ccccc3",
          "t-BuPCy2" : "P(C(C)(C)C)(C2CCCCC2)C3CCCCC3",
          "PCp3" : "C3=CC(P(C1C=CC=C1)C2C=CC=C2)C=C3",
          "PPh3" : "P(c1ccccc1)(c2ccccc2)c3ccccc3",
          "P(o-tolyl)3" : "Cc1ccccc1P(c2ccccc2C)c3ccccc3C",
          "P(nBu)3" : "P(CCCC)(CCCC)CCCC",
          "P(tBu)3" : "P(C(C)(C)C)(C(C)(C)C)C(C)(C)C",
          "P(OMe)3" : "P(OC)(OC)OC",
          "P(CH2Ph)3" : "P(Cc1ccccc1)(Cc2ccccc2)Cc3ccccc3",
          "P(p-OMePh)3" : "P(c1ccc(OC)cc1)(c2ccc(OC)cc2)c3ccc(OC)cc3",
          "PMe3" : "P(C)(C)C",
          "PEt3" : "P(CC)(CC)CC",
          "PiPr3" : "P(C(C)C)(C(C)C)C(C)C",
          "PiBu3" : "P(CC(C)C)(CC(C)C)CC(C)C",
          "PBu3" : "CCCCP(CCCC)CCCC",
          "PMetBu" : "P(C(C)(C)C)(C(C)(C)C)C",
          "JohnPhos" : "CC(C)(C)P(C1=CC=CC=C1C2=CC=CC=C2)C(C)(C)C",
          "CyJohnPhos" : "C4CCCCC4P(C1=CC=CC=C1C2=CC=CC=C2)C3CCCCC3",
          "CyDPEphos" : "c7cc4Cc3cccc(P(C1CCCCC1)C2CCCCC2)c3Oc4c(P(C5CCCCC5)C6CCCCC6)c7",
          "Xantphos" : "CC1(C2=C(C(=CC=C2)P(C3=CC=CC=C3)C4=CC=CC=C4)OC5=C1C=CC=C5P(C6=CC=CC=C6)C7=CC=CC=C7)C",
          "CyXantphos" : "CC7(C)c3cccc(P(C1CCCCC1)C2CCCCC2)c3Oc6c(P(C4CCCCC4)C5CCCCC5)cccc67",
          "XPhos" : "CC(C)c4cc(C(C)C)c(c1ccccc1P(C2CCCCC2)C3CCCCC3)c(C(C)C)c4",
          "RuPhos" : "CC(C)OC1=C(C(=CC=C1)OC(C)C)C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4",
          "SPhos" : "COC1=C(C(=CC=C1)OC)C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4",
          "Tris(2-methoxyphenyl)phosphine" : "P(c1ccccc1(OC))(c2ccccc2(OC))c3ccccc3OC",
          "Tris(4-trifluoromethylphenyl) phosphine" : "P(c1ccc(C(F)(F)F)cc1)(c2ccc(C(F)(F)F)cc2)c3ccc(C(F)(F)F)cc3",
          "PMetBu2" : "CP(C(C)(C)C)C(C)(C)C",
          "PPh2Cy" : "c3ccc(P(c1ccccc1)C2CCCCC2)cc3",
          "P(p-tolyl)3" : "c1cc(C)ccc1P(c2ccc(C)cc2)c3ccc(C)cc3",
          "P(C6F5)3" : "Fc3c(F)c(F)c(P(c1c(F)c(F)c(F)c(F)c1F)c2c(F)c(F)c(F)c(F)c2F)c(F)c3F",
          "P(NMe2)3" : "CN(C)P(N(C)C)N(C)C",
          "c6ccc5c(P(C1CCCCC1)C2CCCCC2)c(P(C3CCCCC3)C4CCCCC4)sc5c6" : "c6ccc5c(P(C1CCCCC1)C2CCCCC2)c(P(C3CCCCC3)C4CCCCC4)sc5c6",
          "c5cc(P(C1CCCCC1)C2CCCCC2)c(P(C3CCCCC3)C4CCCCC4)s5" : "c5cc(P(C1CCCCC1)C2CCCCC2)c(P(C3CCCCC3)C4CCCCC4)s5",
          "c7ccc(c6cc(c1ccccc1)n(c2ccccc2NC(c3ccccc3)P(c4ccccc4)c5ccccc5)n6)cc7" : "c7ccc(c6cc(c1ccccc1)n(c2ccccc2NC(c3ccccc3)P(c4ccccc4)c5ccccc5)n6)cc7",
          "CC(C)P(C(C)C)C(Nc1ccccc1n3nc(c2ccccc2)cc3c4ccccc4)c5ccccc5" : "CC(C)P(C(C)C)C(Nc1ccccc1n3nc(c2ccccc2)cc3c4ccccc4)c5ccccc5",
          "c7ccc(c6cc(c1ccccc1)n(c2ccccc2NC(c3ccccc3)P(C4CCCCC4)C5CCCCC5)n6)cc7" : "c7ccc(c6cc(c1ccccc1)n(c2ccccc2NC(c3ccccc3)P(C4CCCCC4)C5CCCCC5)n6)cc7",
          "C3CCC(P(C1CCCCC1)C2CCCCC2)CC3" : "C3CCC(P(C1CCCCC1)C2CCCCC2)CC3",
          "CC(C)c5cc(C(C)C)c(c4cc(c1c(C(C)C)cc(C(C)C)cc1C(C)C)cc(P(C2CCCC2)C3CCCC3)c4)c(C(C)C)c5" : "CC(C)c5cc(C(C)C)c(c4cc(c1c(C(C)C)cc(C(C)C)cc1C(C)C)cc(P(C2CCCC2)C3CCCC3)c4)c(C(C)C)c5",
          "CC(C)c5cc(C(C)C)c(c4ccc(c1c(C(C)C)cc(C(C)C)cc1C(C)C)c(P(C2CCCC2)C3CCCC3)c4)c(C(C)C)c5" : "CC(C)c5cc(C(C)C)c(c4ccc(c1c(C(C)C)cc(C(C)C)cc1C(C)C)c(P(C2CCCC2)C3CCCC3)c4)c(C(C)C)c5",
      #di-phosphines
          "dppe" : "c4ccc(P(CCP(c1ccccc1)c2ccccc2)c3ccccc3)cc4",
          "depe" : "CCP(CC)CCP(CC)CC",
          "dppp" : "c1ccc(cc1)P(CCCP(c2ccccc2)c3ccccc3)c4ccccc4",
          "dppb" : "c4ccc(P(CCCCP(c1ccccc1)c2ccccc2)c3ccccc3)cc4",
          "dppf" : "[Fe].[CH]1[CH][CH][C]([CH]1)P(c2ccccc2)c3ccccc3.[CH]4[CH][CH][C]([CH]4)P(c5ccccc5)c6ccccc6",
          "dippf" : "[Fe].CC(C)P([C]1[CH][CH][CH][CH]1)C(C)C.CC(C)P([C]2[CH][CH][CH][CH]2)C(C)C",
          "dppf-Ipr" : "[Fe].CCCP(CCC)C1CCCC1.CCCP(CCC)C1CCCC1",
          "dppf-tBu" : "[Fe].CC(C)(C)P(C1CCCC1)C(C)(C)C.CC(C)(C)P(C1CCCC1)C(C)(C)C",
          "dppf-Cy"  : "[Fe].[CH]1[CH][CH][C]([CH]1)P(C2CCCCC2)C3CCCCC3.[CH]4[CH][CH][C]([CH]4)P(C5CCCCC5)C6CCCCC6",
          "dcypf" : "[Fe].[CH]1[CH][CH][C]([CH]1)P(C2CCCCC2)C3CCCCC3.[CH]4[CH][CH][C]([CH]4)P(C5CCCCC5)C6CCCCC6",
          "dcype": "C1CCCCC1P(C2CCCCC2)CCP(C3CCCCC3)C4CCCCC4",
          "dcypbz" : "P(C1CCCCC1)(C2CCCCC2)c3ccccc3P(C5CCCCC5)C6CCCCC6",
          "dcypt" : "C1CCC(CC1)P(C2CCCCC2)C3=CSC=C3P(C4CCCCC4)C5CCCCC5",
          "DCYPT" : "C1CCC(CC1)P(C2CCCCC2)C3=CSC=C3P(C4CCCCC4)C5CCCCC5",
          "dcypb" : "C1CCC(CC1)P(CCCCP(C2CCCCC2)C3CCCCC3)C4CCCCC4",
          "dmpe" : "CP(C)CCP(C)C",
          "rac-BINAP" : "c8ccc(P(c1ccccc1)c3ccc2ccccc2c3c6c(P(c4ccccc4)c5ccccc5)ccc7ccccc67)cc8",
          "L1"   : "P(CCP(C1CCC1)C2CCC2)(C3CCC3)C4CCC4",
          "L2"   : "P(CCP(C1CCCC1)C2CCCC2)(C3CCCC3)C4CCCC4",
          "L3"   : "P(CCP(C1CCCCCC1)C2CCCCCC2)(C3CCCCCC3)C4CCCCCC4",
          "L4"   : "CC(C)P(C(C)C)CCP(C(C)C)(C(C)C)",
          "L5"   : "CC(C)(C)P(C(C)(C)C)CCP(C(C)(C)C)(C(C)(C)C)",
          "L6"   : "P(c1ccccc1)(c2ccccc2)CCP(C4CCCCC4)C5CCCCC5",
     #NHC
          "ItBu" : "CC(C)(C)N1C=CN([C]1)C(C)(C)C",
          "ICy"  : "C1CCC(CC1)N2C=CN([C]2)C3CCCCC3",
          "IPr"  : "CC(C)c1cccc(C(C)C)c1N2C=CN([C]2)c3c(cccc3C(C)C)C(C)C",
          "IMes" : "Cc1cc(C)cc(C)c1N2C=CN([C]2)c3c(C)cc(C)cc3C",
          "IAd"  : "C1C2CC3CC1CC(C2)(C3)N4C=CN([C]4)C56CC7CC(C5)CC(C7)C6",
          "I(2-Ad)" : "C4=CN(C2C1CC3CC(C1)CC2C3)[C]N4C6C5CC7CC(C5)CC6C7",
          "SIPr" : "CC(C)c1cccc(C(C)C)c1N3[C]N(c2c(C(C)C)cccc2C(C)C)CC3",
          "SIMes": "Cc3cc(C)c(N2[C]N(c1c(C)cc(C)cc1C)CC2)c(C)c3",
          "SItBu": "CC(C)(C)N1CCN(C(C)(C)C)[C]1",
          "CDC"  : "CC(C)N2c1ccccc1N(C)C2[C]C4N(C)c3ccccc3N4C(C)C",
          "C1-CDC" : "CN2c1ccccc1N(C)C2[C]C4N(C(C)(C))c3ccccc3N4C(C)C",
          "Me2IPr" : "CC2=C(C)N(c1c(C(C)C)cccc1C(C)C)[C]N2c3c(C(C)C)cccc3C(C)C",
          "CCCCN5CN(c3cccc(N2CN(CCCC)c1ccccc12)n3)c4ccccc45" : "CCCCN5[C]N(c3cccc(N2[C]N(CCCC)c1ccccc12)n3)c4ccccc45",
     #phen/bipy
          "phen" : "c1ccnc3c1ccc2cccnc23",
          "bipy" : "c2ccc(c1ccccn1)nc2",
     #NHC + phosphine
          "IPr+PPh3" : "CC(C)c1cccc(C(C)C)c1N2C=CN(C2)c3c(cccc3C(C)C)C(C)C.P(c1ccccc1)(c2ccccc2)c3ccccc3",
          "PPh3+ItBu" : "P(c1ccccc1)(c2ccccc2)c3ccccc3.CC(C)(C)N1C=CN(C1)C(C)(C)C",
          "PPh3+IPr" : "P(c1ccccc1)(c2ccccc2)c3ccccc3.CC(C)c1cccc(C(C)C)c1N2C=CN(C2)c3c(cccc3C(C)C)C(C)C",
          "PCy3+ItBu" : "P(C1CCCCC1)(C2CCCCC2)C3CCCCC3.CC(C)(C)N1C=CN(C1)C(C)(C)C",
          "PCy3+IPr" : "P(C1CCCCC1)(C2CCCCC2)C3CCCCC3.CC(C)c1cccc(C(C)C)c1N2C=CN(C2)c3c(cccc3C(C)C)C(C)C",
     #autres
          "COD"  : "C1CC=CCCC=C1",
          "acac" : "C/C(=C/C(=O)C)/[O-]",
          "nan" : "nan",
          "dppf + PCy3" : "[Fe].[CH]1[CH][CH][C]([CH]1)P(c2ccccc2)c3ccccc3.[CH]4[CH][CH][C]([CH]4)P(c5ccccc5)c6ccccc6.P(C1CCCCC1)(C2CCCCC2)C3CCCCC3",
          "c2c[n+](C1CCCCC1)cn2C3CCCCC3.[Cl-]" : "c2c[n+](C1CCCCC1)cn2C3CCCCC3.[Cl-]",
          "IIpr-HCl" : "CC(C)n1cc[n+](C(C)C)c1.[Cl-]",
          "IIPr" : "CC(C)N1C=CN(C(C)C)[C]1",
          "(IMe)2-2HBr" : "[Br-].[Br-].Cn2cc[n+](C[n+]1ccn(C)c1)c2",
          "(IMe)2" : "CN1C=CN(CN2C=CN(C)[C]2)[C]1",
          "IPrIMeIIPr-2HBr" : "[Br-].[Br-].CC(C)n2cc[n+](C[n+]1ccn(C(C)C)c1)c2",
          "IPrIMeIIPr" : "CC(C)N2C=CN(CN1C=CN(C(C)C)[C]1)[C]2",
          "ItBuIMeIItBu-2HBr" : "[Br-].[Br-].CC(C)(C)n2cc[n+](C[n+]1ccn(C(C)(C)C)c1)c2",
          "ItBuIMeIItBu" : "CC(C)(C)N2C=CN(CN1C=CN(C(C)(C)C)[C]1)[C]2",
          "ICyIMeIICy-2HBr" : "[Br-].[Br-].c3c[n+](C[n+]2ccn(C1CCCCC1)c2)cn3C4CCCCC4",
          "ICyIMeIICy" : "C1CCCCC1N2C=CN(CN1C=CN(C3CCCCC3)[C]1)[C]2",
          "CC(C)(C)n2cc[n+](n1cc[n+](C(C)(C)C)c1)c2.[Br-].[Br-]" : "CC(C)(C)n2cc[n+](n1cc[n+](C(C)(C)C)c1)c2.[Br-].[Br-]",
          "C1CCCC1P(C2CCCC2)c3cc(c4c(C(C)C)cc(C(C)C)cc4(C(C)C))cc(c4c(C(C)C)cc(C(C)C)cc4(C(C)C))c3" : "C1CCCC1P(C2CCCC2)c3cc(c4c(C(C)C)cc(C(C)C)cc4(C(C)C))cc(c4c(C(C)C)cc(C(C)C)cc4(C(C)C))c3"
    }

