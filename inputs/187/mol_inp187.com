%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      2.977948    0.378500    2.655527
C      2.575439    1.575289    1.783722
C      2.464542    2.824366    2.672870
C      3.629999    1.827341    0.698212
C      1.219661    1.299132    1.122754
O      0.220371    1.991465    1.270370
O      1.308179    0.154006    0.332629
C      0.112549   -0.189397   -0.320925
C     -0.227825    0.522911   -1.468049
C     -1.382102    0.195184   -2.177980
C     -1.717637    0.937217   -3.323497
C     -2.861109    0.643768   -4.060465
C     -3.683003   -0.397182   -3.658815
C     -3.358954   -1.142733   -2.520766
C     -2.203531   -0.877676   -1.741552
C     -1.829743   -1.625337   -0.563524
C     -2.577098   -2.719345   -0.055444
C     -2.187798   -3.432255    1.082049
C     -1.031740   -3.076892    1.755452
C     -0.271169   -2.007886    1.290398
C     -0.651235   -1.272751    0.144776
H      3.131927   -0.525476    2.055629
H      2.200218    0.149798    3.393270
H      3.909903    0.577304    3.196351
H      1.710701    2.687902    3.456997
H      3.418604    3.050876    3.161956
H      2.171908    3.704064    2.087854
H      3.325066    2.645562    0.035640
H      4.597788    2.091230    1.138985
H      3.783393    0.942294    0.070527
H      0.409737    1.340135   -1.794587
H     -1.085736    1.760511   -3.651210
H     -3.107790    1.228495   -4.942273
H     -4.579658   -0.634409   -4.225409
H     -4.039792   -1.945298   -2.252470
H     -3.491452   -3.043335   -0.544292
H     -2.792500   -4.263518    1.434974
H     -0.720837   -3.625610    2.640022
H      0.632775   -1.748248    1.836299

