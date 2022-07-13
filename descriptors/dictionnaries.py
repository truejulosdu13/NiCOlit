## Mappings of "usual organic" names to SMILES ##
### Solvents - Ligands - Precursors ###

# solvents #
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

# solvents #
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

# ligands #
dict_ligand = {
             'nan': 'NoLigand',
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
    #Diphosphines
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
    # others
             'COD': 'C1=CCCC=CCC1',
             'acac': 'CC(=O)/C=C(/C)[O-]'}

# precursors
dict_smiles_catalysts = {
    'Ni(cod)2': 'C1CC=CCCC=C1->[Ni]<-C2=CCCC=CCC2', 
    'NiCl2(PCy3)2': 'C1CCCCC1P(C1CCCCC1)(C1CCCCC1)->[Ni](<-P(C1CCCCC1)(C1CCCCC1)(C1CCCCC1))(Cl)(Cl)', 
    'NiCl2(PPh3)2': 'c1ccccc1P(c1ccccc1)(c1ccccc1)->[Ni](<-P(c1ccccc1)(c1ccccc1)(c1ccccc1))(Cl)(Cl)',
    'NiCl2(Pph3)2': 'c1ccccc1P(c1ccccc1)(c1ccccc1)->[Ni](<-P(c1ccccc1)(c1ccccc1)(c1ccccc1))(Cl)(Cl)', 
    'NiCl2(PMe3)2': 'CP(C)(C)->[Ni](<-P(C)(C)C)(Cl)(Cl)', 
    'NiCl2(PEt3)2': 'CCP(CC)(CC)->[Ni](<-P(CC)(CC)(CC))(Cl)(Cl)', 
    'NiCl2(PiBu3)2': 'CC(C)CP(CC(C)C)(CC(C)C)->[Ni](<-P(CC(C)C)(CC(C)C)CC(C)C)(Cl)(Cl)',
    'NiCl2(PBu3)2': 'CCCCP(CCCC)(CCCC)->[Ni](<-P(CCCC)(CCCC)(CCCC))(Cl)(Cl)', 
    'NiCl2(PiPr3)2': 'CC(C)P(C(C)C)(C(C)C)->[Ni](<-P(C(C)C)(C(C)C)C(C)C)(Cl)(Cl)',
    'NiCl2(PhPCy2)2': 'C1CCCCC1P(c1ccccc1)(C1CCCCC1)->[Ni](<-P(c1ccccc1)(C1CCCCC1)(C1CCCCC1))(Cl)(Cl)',
    'NiCl2(Ph2PCy)2': 'C1CCCCC1P(c1ccccc1)(c1ccccc1)->[Ni](<-P(c1ccccc1)(c1ccccc1)(C1CCCCC1))(Cl)(Cl)',
    'Cl[Ni](Cl)([P+](C1CCCCC1)(C2CCCCC2)C(Nc3ccccc3n5nc(c4ccccc4)cc5c6ccccc6)c7ccccc7)[P+](C8CCCCC8)(C9CCCCC9)C(Nc%10ccccc%10n%12nc(c%11ccccc%11)cc%12c%13ccccc%13)c%14ccccc%14':'Cl[Ni](Cl)(<-P(C1CCCCC1)(C2CCCCC2)C(Nc3ccccc3n5nc(c4ccccc4)cc5c6ccccc6)c7ccccc7)<-P(C8CCCCC8)(C9CCCCC9)C(Nc%10ccccc%10n%12nc(c%11ccccc%11)cc%12c%13ccccc%13)c%14ccccc%14',
    'CC(C)[P+](C(C)C)(C(Nc1ccccc1n3nc(c2ccccc2)cc3c4ccccc4)c5ccccc5)[Ni](Cl)(Cl)[P+](C(C)C)(C(C)C)C(Nc6ccccc6n8nc(c7ccccc7)cc8c9ccccc9)c%10ccccc%10' : 'CC(C)P(C(C)C)(C(Nc1ccccc1n3nc(c2ccccc2)cc3c4ccccc4)c5ccccc5)->[Ni](Cl)(Cl)<-P(C(C)C)(C(C)C)C(Nc6ccccc6n8nc(c7ccccc7)cc8c9ccccc9)c%10ccccc%10',
    'Cl[Ni](Cl)([P+](c1ccccc1)(c2ccccc2)C(Nc3ccccc3n5nc(c4ccccc4)cc5c6ccccc6)c7ccccc7)[P+](c8ccccc8)(c9ccccc9)C(Nc%10ccccc%10n%12nc(c%11ccccc%11)cc%12c%13ccccc%13)c%14ccccc%14' : 'P(c1ccccc1)(c2ccccc2)(C(Nc3ccccc3n5nc(c4ccccc4)cc5c6ccccc6)c7ccccc7)->[Ni](Cl)(Cl)<-P(c8ccccc8)(c9ccccc9)C(Nc%10ccccc%10n%12nc(c%11ccccc%11)cc%12c%13ccccc%13)c%14ccccc%14',
    'Ni(PCy3)2(C2H4)': 'C1CCCCC1P(C1CCCCC1)(C1CCCCC1)->[Ni]<-2(<-P(C1CCCCC1)(C1CCCCC1)C1CCCCC1)(<-C=C2)',
    'NiBr2(PPh3)2': 'c1ccccc1P(c1ccccc1)(c1ccccc1)->[Ni](<-P(c1ccccc1)(c1ccccc1)(c1ccccc1))(Br)(Br)', 
    'NiBr2(PCy3)2': 'C1CCCCC1P(C1CCCCC1)(C1CCCCC1)->[Ni](<-P(C1CCCCC1)(C1CCCCC1)(C1CCCCC1))(Br)(Br)', 
    
    'NiCl2(glyme)': 'CO1CCO(->[Ni]<-1(Cl)(Cl))C', 
    'NiCl2(dme)': 'CO1CCO(->[Ni]<-1(Cl)(Cl))C',
    'NiCl2(Py)2': 'c1ccccn1->[Ni](<-n1ccccc1)(Cl)(Cl)', 
    'NiCl2(phen)': 'c1cc2ccc3cccn4->[Ni](Cl)(Cl)<-n(c1)c2c34', 
    'NiBr2(bipy)2': 'c1cccc3n1->[Ni](Br)(Br)<-n2c3cccc2',
    'NiBr2(glyme)': 'CO1CCO(->[Ni]<-1(Br)(Br))C',
    'NiBr2(diglyme)': 'CO2CCO1CCO(->[Ni]<-1<-2(Br)(Br))C',
    'CCCCN4c1ccccc1N5c6cccc7N3c2ccccc2N(CCCC)C3[Ni](Br)(C45)[n+]67.[Br-]': 'CCCCN4c1ccccc1N5c6cccc7N3c2ccccc2N(CCCC)C3[Ni](Br)(Br)(C45)<-n67',
    'CN2C=CN3c4cccc5N1C=CN(C)C1[Ni](Br)(C23)[n+]45.[Br-]': 'CN2C=CN3c4cccc5N1C=CN(C)C1[Ni](Br)(Br)(C23)<-n45', 
    'NiF2': 'F[Ni]F', 
    'NiCl2': 'Cl[Ni]Cl',
    'NiBr2': 'Br[Ni]Br', 
    'NiI2': 'I[Ni]I',
    'Ni(acac)2': 'CC1=O->[Ni]<-2(<-O=C(C)C1)<-O=C(C)CC(C)=O2',
    'Ni(OAc)2': 'CC(=O)O[Ni]OC(=O)C',
    'Ni(OTf)2': 'FC(F)(F)S(=O)(=O)O[Ni]OS(=O)(=O)C(F)(F)F', 

    'NiO'                   : '[Ni]=O', 
    
    'Ni(o-tol)Cl(dppf)'     : 'c5ccc(P4(c2ccccc2)C6(=CC=C[C-]76)->[Fe++]<-7<-9<-C8(=CC=C[C-]89)P(c2ccccc2)(c3ccccc3)->[Ni](c1c(C)cccc1)(Cl)<-4)cc5', 
    'Ni(o-tol)Cl(dippf)'    : 'CC(C)P4(C(C)C)C6(=CC=C[C-]76)->[Fe++]<-7<-9<-C8(=CC=C[C-]89)P(C(C)C)(C(C)C)->[Ni](c1c(C)cccc1)(Cl)<-4', 
    'Ni(o-tol)Cl(dcypf)'    : 'C1CCCCC1P4(C1CCCCC1)C6(=CC=C[C-]76)->[Fe++]<-7<-9<-C8(=CC=C[C-]89)P(C1CCCCC1)(C1CCCCC1)->[Ni](c1c(C)cccc1)(Cl)<-4',
    'Ni(2-OMePh)Br(dcypf)'  : 'C1CCCCC1P4(C1CCCCC1)C6(=CC=C[C-]76)->[Fe++]<-7<-9<-C8(=CC=C[C-]89)P(C1CCCCC1)(C1CCCCC1)->[Ni](c1c(OC)cccc1)(Br)<-4', 
    'Ni(2,4-xylyl)Br(dcypf)': 'C1CCCCC1P4(C1CCCCC1)C6(=CC=C[C-]76)->[Fe++]<-7<-9<-C8(=CC=C[C-]89)P(C1CCCCC1)(C1CCCCC1)->[Ni](c1c(C)cc(C)cc1)(Br)<-4',
    'Ni(naphtyl)Br(dcypf)'  : 'C1CCCCC1P4(C1CCCCC1)C6(=CC=C[C-]76)->[Fe++]<-7<-9<-C8(=CC=C[C-]89)P(C1CCCCC1)(C1CCCCC1)->[Ni](c1c(cccc2)c2ccc1)(Cl)<-4', 
    'Ni(2-CF3Ph)Br(dcypf)'  : 'C1CCCCC1P4(C1CCCCC1)C6(=CC=C[C-]76)->[Fe++]<-7<-9<-C8(=CC=C[C-]89)P(C1CCCCC1)(C1CCCCC1)->[Ni](c1c(C(F)(F)F)cccc1)(Br)<-4', 
    'Ni(2,6-xylyl)Br(dcypf)': 'C1CCCCC1P4(C1CCCCC1)C6(=CC=C[C-]76)->[Fe++]<-7<-9<-C8(=CC=C[C-]89)P(C1CCCCC1)(C1CCCCC1)->[Ni](c1c(C)cccc1C)(Br)<-4', 
    'Ni(o-tol)Br(dcypf)'    : 'C1CCCCC1P4(C1CCCCC1)C6(=CC=C[C-]76)->[Fe++]<-7<-9<-C8(=CC=C[C-]89)P(C1CCCCC1)(C1CCCCC1)->[Ni](c1c(C)cccc1)(Br)<-4',
    'Ni(2-ethylPh)Br(dcypf)': 'C1CCCCC1P4(C1CCCCC1)C6(=CC=C[C-]76)->[Fe++]<-7<-9<-C8(=CC=C[C-]89)P(C1CCCCC1)(C1CCCCC1)->[Ni](c1c(CC)cc(C)cc1)(Br)<-4',
    'Ni(dcype)(CO)2': 'C1CCCCC1P4(C1CCCCC1)CCP(C1CCCCC1)(C1CCCCC1)->[Ni](<-[C-]#[O+])(<-[C-]#[O+])<-4', 
    'Ni(L1)(CO)2': 'C1CCC1P4(C1CCC1)CCP(C1CCC1)(C1CCC1)->[Ni](<-[C-]#[O+])(<-[C-]#[O+])<-4', 
    'Ni(L2)(CO)2': 'C1CCCC1P4(C1CCCC1)CCP(C1CCCC1)(C1CCCC1)->[Ni](<-[C-]#[O+])(<-[C-]#[O+])<-4', 
    'Ni(L3)(CO)2': 'C1CCCCC1P4(C1CCCCC1)CCP(C1CCCCC1)(C1CCCCC1)->[Ni](<-[C-]#[O+])(<-[C-]#[O+])<-4',
    'Ni(dcypbz)(CO)2': 'C1CCCCC1P4(C1CCCCC1)c(cccc3)c3P(C1CCCCC1)(C1CCCCC1)->[Ni](<-[C-]#[O+])(<-[C-]#[O+])<-4', 
    'Ni(dcypt)(CO)2': 'C1CCCCC1P4(C1CCCCC1)c(csc3)c3P(C1CCCCC1)(C1CCCCC1)->[Ni](<-[C-]#[O+])(<-[C-]#[O+])<-4', 
    'Ni(dppe)(CO)2': 'c5ccc(P4(c1ccccc1)CCP(c2ccccc2)(c3ccccc3)->[Ni](<-[C-]#[O+])(<-[C-]#[O+])<-4)cc5',
    'Ni(L4)(CO)2': 'CC(C)P4(C(C)C)CCP(C(C)C)(C(C)C)->[Ni](<-[C-]#[O+])(<-[C-]#[O+])<-4', 
    'Ni(L5)(CO)2': 'CC(C)CP4(CC(C)(C))CCP(CC(C)(C))(CC(C)(C))->[Ni](<-[C-]#[O+])(<-[C-]#[O+])<-4',
    'Ni(L6)(CO)2': 'C1CCCCC1P4(C1CCCCC1)CCP(c1ccccc1)(c1ccccc1)->[Ni](<-[C-]#[O+])(<-[C-]#[O+])<-4', 
    'NiCl2(dppe)': 'c5ccc(P4(c1ccccc1)CCP(c2ccccc2)(c3ccccc3)->[Ni](Cl)(Cl)<-4)cc5',
    'NiCl2(dppp)': 'c5ccc(P4(c1ccccc1)CCCP(c2ccccc2)(c3ccccc3)->[Ni](Cl)(Cl)<-4)cc5', 
    'NiCl2(dppb)': 'c5ccc(P4(c1ccccc1)CCCCP(c2ccccc2)(c3ccccc3)->[Ni](Cl)(Cl)<-4)cc5', 
    'NiCl2(dppf)': 'c5ccc(P4(c1ccccc1)C6(=CC=C[C-]76)->[Fe++]<-7<-9<-C8(=CC=C[C-]89)P(c2ccccc2)(c3ccccc3)->[Ni](Cl)(Cl)<-4)cc5',
    
    'NiCl2(IPr) (PPh3)' : 'c2(C(C)C)cccc(C(C)C)c2N1C=CN(c2c(C(C)C)cccc2C(C)C)[C]1->[Ni](Cl)(Cl)<-P(c1ccccc1)(c1ccccc1)c1ccccc1',
    'NiBr2(PPh3)(IPr)'  : 'c2(C(C)C)cccc(C(C)C)c2N1C=CN(c2c(C(C)C)cccc2C(C)C)[C]1->[Ni](Br)(Br)<-P(c1ccccc1)(c1ccccc1)c1ccccc1',
    'NiBr2(PCy3)(IPr)'  : 'c2(C(C)C)cccc(C(C)C)c2N1C=CN(c2c(C(C)C)cccc2C(C)C)[C]1->[Ni](Br)(Br)<-P(C1CCCCC1)(C1CCCCC1)C1CCCCC1',
    'NiCl2(PPh3)(ItBu)' : 'CC(C)(C)N1C=CN(C(C)(C)C)[C]1->[Ni](Cl)(Cl)<-P(c1ccccc1)(c1ccccc1)c1ccccc1',
    'NiBr2(PPh3)(ItBu)' : 'CC(C)(C)N1C=CN(C(C)(C)C)[C]1->[Ni](Br)(Br)<-P(c1ccccc1)(c1ccccc1)c1ccccc1',
    'NiBr2(PCy3)(ItBu)' : 'CC(C)(C)N1C=CN(C(C)(C)C)[C]1->[Ni](Br)(Br)<-P(C1CCCCC1)(C1CCCCC1)C1CCCCC1',
    'NiBr2(IPr)2' :  'c2(C(C)C)cccc(C(C)C)c2N1C=CN(c2c(C(C)C)cccc2C(C)C)[C]1->[Ni](Br)(Br)<-[C]1N(c2c(C(C)C)cccc2C(C)C)C=CN1c2c(C(C)C)cccc2C(C)C'
}


### Lists of molecules needed in the rest of the code ###

# DFT Featurizartion failed on these Lewis acids
Lewis_Acids_to_drop = ['O=C(O[Cs])O[Cs]', 'Cl[Cs]', 
                       'O=S(=O)(O[Sc](OS(=O)(=O)C(F)(F)F)OS(=O)(=O)C(F)(F)F)C(F)(F)F', 
                       'F[Cs]', 'O=P(O[Na])(O[Na])O[Na]', '[Rb+]',
                       'CC(C)(C)C(=O)O[Cs]', '[Cs+]', 'CC(=O)O[Cu]OC(C)=O', 'F[Sr]F']


# List of additives
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


# Mapping to go from precursor to simplified category (oxidation state of the nickel) 
Ni0 = ['Ni(cod)2', 
       'Ni(dcypbz)(CO)2', 
       'Ni(dcype)(CO)2', 
       'Ni(dcypt)(CO)2', 
       'Ni(dppe)(CO)2', 
       'Ni(L1)(CO)2',
       'Ni(L2)(CO)2', 
       'Ni(L3)(CO)2', 
       'Ni(L4)(CO)2', 
       'Ni(L5)(CO)2', 
       'Ni(L6)(CO)2', 
       'Ni(PCy3)2', 
       'Ni(PPh3)4']

Ni2 = ['Ni(acac)2', 
       'NiBr2', 
       'NiBr2(glyme)', 
       'NiBr2(diglyme)', 
       'NiBr2(IPr)2', 
       'NiBr2(PCy3)2', 
       'NiBr2(PCy3)(IPr)', 
       'NiBr2(PCy3)', 
       '(ItBu)', 
       'NiBr2(PPh3)2', 
       'NiBr2(PPh3)(IPr)', 
       'NiBr2(PPh3)(ItBu)', 
       'Ni(2-CF3Ph)Br(dcypf)', 
       'NiCl2', 
       'NiCl2(dme)', 
       'NiCl2(dme)2', 
       'NiCl2(dppb)', 
       'NiCl2(dppe)', 
       'NiCl2(dppf)', 
       'NiCl2(dppp)', 
       'NiCl2(glyme)', 
       'NiCl2(IPr) (PPh3)',
       'NiCl2(PBu3)2', 
       'NiCl2(PCy3)2', 
       'NiCl2(phen)', 
       'NiCl2(PEt3)2', 
       'NiCl2(Ph2PCy)2', 
       'NiCl2(PhPCy2)2', 
       'NiCl2(PiBu3)2',
       'NiCl2(PiPr3)2', 
       'NiCl2(PMe3)2', 
       'NiCl2(PPh3)2', 
       'NiCl2(PPh3)(ItBu)', 
       'NiCl2(Py)2', 
       'Ni(2-ethylPh)Br', 
       'NiF2',
       'NiI2', 
       'NiO', 
       'Ni(OAc)2-4H2O', 
       'Ni(2-OMePh)Br ', 
       'Ni(OTf)2', 
       'Ni(o-tol)Br', 
       'Ni(o-tol)Cl', 
       'Ni(naphtyl)Br ',
       'Ni(PCy3)2(C2H4)', 
       'Ni(2,4-xylyl)Br', 
       'Ni(2,6-xylyl)Br', 
       'NiCl2(dme)2', 
       'Ni(OAc)2', 
       'NiBr2(bipy)2',
       'CCCCN4c1ccccc1N5c6cccc7N3c2ccccc2N(CCCC)C3[Ni](Br)(C45)[n+]67.[Br-]', 
       'CN2C=CN3c4cccc5N1C=CN(C)C1[Ni](Br)(C23)[n+]45.[Br-]',
       'Ni(o-tol)Cl(dppf)', 
       'Ni(o-tol)Cl(dippf)', 
       'Ni(o-tol)Cl(dcypf)', 
       'Ni(2-OMePh)Br(dcypf)', 
       'Ni(2,4-xylyl)Br(dcypf)',
       'Ni(naphtyl)Br(dcypf)', 
       'Ni(2,6-xylyl)Br(dcypf)', 
       'Ni(o-tol)Br(dcypf)', 
       'Ni(2-ethylPh)Br(dcypf)', 
       'NiCl2(Pph3)2',
       'Cl[Ni](Cl)([P+](C1CCCCC1)(C2CCCCC2)C(Nc3ccccc3n5nc(c4ccccc4)cc5c6ccccc6)c7ccccc7)[P+](C8CCCCC8)(C9CCCCC9)C(Nc%10ccccc%10n%12nc(c%11ccccc%11)cc%12c%13ccccc%13)c%14ccccc%14',
       'CC(C)[P+](C(C)C)(C(Nc1ccccc1n3nc(c2ccccc2)cc3c4ccccc4)c5ccccc5)[Ni](Cl)(Cl)[P+](C(C)C)(C(C)C)C(Nc6ccccc6n8nc(c7ccccc7)cc8c9ccccc9)c%10ccccc%10',
       'Cl[Ni](Cl)([P+](c1ccccc1)(c2ccccc2)C(Nc3ccccc3n5nc(c4ccccc4)cc5c6ccccc6)c7ccccc7)[P+](c8ccccc8)(c9ccccc9)C(Nc%10ccccc%10n%12nc(c%11ccccc%11)cc%12c%13ccccc%13)c%14ccccc%14', 
       'NiBr2(PCy3)(ItBu)', 
       'NiCl(PCy3)2(para-trifluorophenyl)', 
       'NiCl2(dppp)']


# lists of unnecessary DFT-descriptors:
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

descritpors_to_remove_al = ["converged", "stoichiometry", "ES_root_molar_volume", "X_0", "Y_0", "Z_0", "at_0", "ES_transition_7", "ES_transition_8", "ES_transition_9", 'ES_osc_strength_7', 'ES_osc_strength_8', 'ES_osc_strength_9', 'ES_<S**2>_7', 'ES_<S**2>_8', 'ES_<S**2>_9']
