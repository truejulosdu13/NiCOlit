%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      6.318711    0.596189    0.177286
O      5.334609   -0.429240    0.218634
C      4.033934   -0.046656    0.034812
C      3.588252    1.250670   -0.208092
C      2.222792    1.504241   -0.378991
C      1.282169    0.465512   -0.309388
C     -0.089343    0.707532   -0.479128
C     -0.995562   -0.347399   -0.390895
O     -2.348359   -0.074316   -0.623098
C     -3.179274   -0.181860    0.489989
O     -2.790184   -0.462963    1.619506
N     -4.511804    0.075385    0.182784
C     -4.936020    0.567812   -1.115867
C     -5.473611    0.163061    1.264494
C     -0.568016   -1.652446   -0.168476
C      0.793859   -1.901176    0.002312
C      1.729729   -0.853683   -0.064330
C      3.104026   -1.088586    0.104520
H      6.347833    1.075609   -0.806759
H      6.156401    1.332024    0.971843
H      7.293953    0.130623    0.349323
H      4.272998    2.089477   -0.270673
H      1.898711    2.525806   -0.565796
H     -0.455838    1.710863   -0.675441
H     -4.246981    0.272923   -1.910662
H     -4.985760    1.659811   -1.074192
H     -5.925747    0.163988   -1.348427
H     -6.408060   -0.312773    0.953151
H     -5.655663    1.218924    1.485859
H     -5.113115   -0.331515    2.170360
H     -1.282528   -2.468149   -0.118399
H      1.122964   -2.921024    0.188871
H      3.464923   -2.097243    0.294870
