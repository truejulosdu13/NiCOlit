%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      4.196164    0.972348    1.234092
N      3.358581    0.156130    0.376571
C      4.032966   -0.651554   -0.624491
C      1.991565    0.410378    0.338226
O      1.430526    1.203879    1.087607
O      1.340455   -0.329923   -0.648000
C     -0.046994   -0.165944   -0.755022
C     -0.552354    0.241899   -1.982963
C     -1.935251    0.361682   -2.144736
C     -2.819926    0.070907   -1.112971
N     -4.141775    0.201116   -1.321773
C     -4.976223   -0.089458   -0.299214
C     -4.553864   -0.515475    0.945786
C     -3.187940   -0.653003    1.158598
C     -2.289446   -0.359827    0.124541
C     -0.900921   -0.486780    0.296877
H      4.963752    0.341638    1.691843
H      4.672347    1.745917    0.624305
H      3.618051    1.453099    2.027749
H      4.411336    0.010875   -1.408519
H      3.363366   -1.388704   -1.073667
H      4.867887   -1.181799   -0.157115
H      0.114596    0.468697   -2.808837
H     -2.336663    0.688551   -3.100929
H     -6.032930    0.035328   -0.518304
H     -5.268911   -0.733172    1.730618
H     -2.828168   -0.983548    2.129516
H     -0.490227   -0.823257    1.244620
