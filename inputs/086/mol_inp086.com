%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      5.702156    0.852936   -0.329107
O      4.936416   -0.251658    0.135043
C      3.575979   -0.123708    0.087040
C      2.868007   -1.231781    0.554266
C      1.469540   -1.235583    0.560935
C      0.744737   -0.125739    0.099300
N     -0.698425   -0.122978    0.103945
C     -1.416513    1.070891    0.483922
C     -0.981750    1.865823    1.559445
C     -1.676635    3.021193    1.926963
C     -2.817518    3.402068    1.224720
C     -3.265233    2.629146    0.156084
C     -2.571079    1.473147   -0.210815
C     -1.425023   -1.313728   -0.270152
C     -2.582572   -1.704412    0.426422
C     -3.285537   -2.856870    0.065011
C     -2.843798   -3.638189   -0.999897
C     -1.700069   -3.269247   -1.703750
C     -0.996563   -2.117260   -1.341831
C      1.470125    0.983618   -0.367932
C      2.871290    0.985993   -0.375296
H      5.515745    1.745428    0.277372
H      6.760131    0.594113   -0.223539
H      5.511419    1.047757   -1.389652
H      3.408006   -2.101306    0.920840
H      0.949777   -2.108638    0.948140
H     -0.104222    1.576444    2.132948
H     -1.327254    3.619132    2.764555
H     -3.358060    4.300603    1.510653
H     -4.153229    2.925001   -0.396264
H     -2.928070    0.893307   -1.058808
H     -2.934912   -1.118888    1.272425
H     -4.175140   -3.143914    0.619391
H     -3.390423   -4.534564   -1.280933
H     -1.354247   -3.874575   -2.537510
H     -0.116139   -1.838714   -1.916178
H      0.944239    1.855596   -0.750701
H      3.374843    1.869558   -0.752220

