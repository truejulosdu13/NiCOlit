%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      6.711223   -0.703872   -1.388264
C      6.166889    0.201981   -0.299801
O      4.732345    0.214618   -0.338684
C      4.114644   -0.766233    0.365641
O      4.667481   -1.645680    1.007144
C      2.635179   -0.640054    0.253118
C      2.021601    0.361448   -0.516175
C      0.628672    0.435693   -0.590571
C     -0.142949   -0.483476    0.118689
O     -1.537818   -0.463177    0.022979
C     -2.160693    0.602648    0.668896
O     -1.568504    1.475518    1.296452
N     -3.544265    0.567952    0.529634
C     -4.236159   -0.455254   -0.249734
C     -4.557026   -1.680581    0.590792
C     -4.347008    1.589953    1.191049
C     -4.582704    2.790796    0.289074
C      0.446840   -1.495876    0.871140
C      1.839085   -1.568994    0.942656
H      7.804336   -0.675279   -1.406017
H      6.333535   -0.396225   -2.368901
H      6.391123   -1.739220   -1.234348
H      6.536921   -0.091680    0.689274
H      6.500451    1.229952   -0.475534
H      2.615078    1.090701   -1.061787
H      0.154658    1.210090   -1.186608
H     -3.613996   -0.750363   -1.101372
H     -5.155896   -0.024891   -0.661015
H     -5.200452   -1.419478    1.437245
H     -3.645568   -2.134369    0.993274
H     -5.074949   -2.431384   -0.013730
H     -5.300928    1.147425    1.498277
H     -3.836702    1.921078    2.102483
H     -3.635092    3.252012   -0.008005
H     -5.182723    3.544463    0.808086
H     -5.112699    2.500865   -0.623915
H     -0.167155   -2.217851    1.401161
H      2.303222   -2.353256    1.538034
