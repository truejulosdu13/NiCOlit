%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      3.353717   -0.255719    1.364914
N      3.072744    0.038728   -0.029078
C      4.072603   -0.362344   -0.999708
C      1.802424    0.383765   -0.481108
O      1.517448    0.493319   -1.669763
O      0.905470    0.581163    0.564953
C     -0.398756    0.908493    0.160843
C     -0.666119    2.233984   -0.188583
C     -1.959567    2.594398   -0.547853
C     -2.976970    1.637074   -0.540631
C     -2.719548    0.304691   -0.169630
C     -3.747859   -0.652248   -0.152862
C     -3.493092   -1.972045    0.222260
C     -2.206702   -2.354211    0.586289
C     -1.172832   -1.416677    0.573837
C     -1.407717   -0.077556    0.195507
H      3.251115   -1.333753    1.519896
H      4.376336    0.053624    1.599888
H      2.670958    0.267227    2.038649
H      3.834492    0.002202   -2.002511
H      4.119830   -1.455174   -1.021941
H      5.044661    0.040977   -0.701459
H      0.128678    2.973458   -0.190710
H     -2.179256    3.619041   -0.835527
H     -3.981869    1.940959   -0.826868
H     -4.761258   -0.373865   -0.433798
H     -4.300044   -2.699970    0.228561
H     -2.004879   -3.381529    0.877388
H     -0.174006   -1.738012    0.859046
