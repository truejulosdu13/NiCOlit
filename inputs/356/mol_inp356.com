%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      2.060750    3.026209   -0.709395
O      2.246207    1.634270   -0.499419
C      1.130915    0.869186   -0.258956
C     -0.175599    1.369231   -0.207604
C     -1.266585    0.520284    0.044206
C     -2.581059    1.014131    0.097385
C     -3.656335    0.160591    0.348386
C     -3.432234   -1.197396    0.549508
C     -2.132579   -1.703540    0.499860
C     -1.039224   -0.857486    0.248224
C      0.273910   -1.354770    0.195696
C      1.357683   -0.505092   -0.055581
O      2.666952   -0.915520   -0.122032
C      2.938496   -2.295080    0.075606
H      3.044763    3.472194   -0.883520
H      1.449632    3.213952   -1.598321
H      1.631338    3.505900    0.176272
H     -0.371962    2.425113   -0.361693
H     -2.777940    2.072779   -0.057126
H     -4.667513    0.557313    0.386524
H     -4.267894   -1.864263    0.745169
H     -1.979009   -2.768524    0.659847
H      0.427084   -2.416960    0.355323
H      4.019731   -2.439039   -0.012185
H      2.641613   -2.616597    1.079290
H      2.458861   -2.906887   -0.695465

