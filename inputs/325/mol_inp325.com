%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      4.407090   -0.804031   -2.672687
C      4.515210   -0.635851   -1.165739
N      3.924947    0.623119   -0.719351
C      4.789939    1.792433   -0.618518
C      5.437520    1.896476    0.752973
C      2.578769    0.749487   -0.390315
O      2.081833    1.795281    0.016584
O      1.874984   -0.437239   -0.584729
C      0.515554   -0.409614   -0.257703
C      0.093638   -0.250011    1.055165
C     -1.275706   -0.289520    1.344769
C     -2.244963   -0.497018    0.346715
N     -3.598189   -0.538893    0.635450
C     -4.103130   -0.685127    2.009132
C     -5.550899   -1.203850    1.980841
O     -6.466770   -0.349857    1.280352
C     -5.836726    0.441991    0.263828
C     -4.613429   -0.228038   -0.385696
C     -1.767572   -0.680320   -0.964407
C     -0.402375   -0.644007   -1.271079
H      3.361658   -0.797993   -2.997801
H      4.856376   -1.752426   -2.982667
H      4.923260    0.006738   -3.196981
H      4.013348   -1.472818   -0.668659
H      5.565314   -0.664018   -0.854638
H      4.202328    2.697957   -0.806949
H      5.554728    1.735825   -1.400879
H      4.680942    1.973008    1.540802
H      6.077454    2.782571    0.805255
H      6.053145    1.016799    0.966698
H      0.811634   -0.083090    1.851893
H     -1.569897   -0.129699    2.377907
H     -4.065623    0.287828    2.513096
H     -3.498229   -1.402459    2.575477
H     -5.589888   -2.189633    1.502259
H     -5.928208   -1.322243    3.001599
H     -5.548452    1.396943    0.719316
H     -6.592814    0.660632   -0.497058
H     -4.915877   -1.161425   -0.874313
H     -4.223498    0.457602   -1.146645
H     -2.458678   -0.884291   -1.777479
H     -0.068746   -0.801220   -2.291697
