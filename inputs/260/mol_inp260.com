%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      4.060004    0.319675    0.693724
O      3.168737   -0.786398    0.725587
C      2.781208   -1.192945   -0.586196
O      2.058684   -0.202008   -1.315560
C      0.746826   -0.003542   -0.983711
C      0.023399    0.760190   -1.899840
C     -1.327100    1.041663   -1.682919
C     -1.977903    0.567862   -0.535418
C     -3.335094    0.839888   -0.298177
C     -3.966582    0.365724    0.852529
C     -3.249646   -0.384348    1.779814
C     -1.899947   -0.663104    1.557768
C     -1.248778   -0.196174    0.403937
C      0.111305   -0.471055    0.167789
H      4.351783    0.553117    1.721329
H      3.572360    1.200789    0.266356
H      4.961682    0.073145    0.124761
H      3.680866   -1.439258   -1.162590
H      2.203421   -2.122728   -0.515337
H      0.513161    1.137813   -2.793922
H     -1.865227    1.635917   -2.417489
H     -3.910952    1.426456   -1.010398
H     -5.017282    0.583964    1.024403
H     -3.739126   -0.752904    2.677470
H     -1.356557   -1.249849    2.295343
H      0.660757   -1.041891    0.910747
