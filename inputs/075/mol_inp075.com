%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -0.780437    3.648441   -1.091163
O     -0.911805    2.316049   -0.617149
C      0.220096    1.538818   -0.559522
C      1.487798    1.981004   -0.938329
C      2.587641    1.127954   -0.849289
C      2.449671   -0.183521   -0.381356
C      3.574353   -1.022571   -0.301541
C      3.462518   -2.332212    0.161044
C      2.223134   -2.819701    0.550047
C      1.097419   -1.995962    0.475590
C      1.174656   -0.660000    0.010368
C      0.046439    0.215666   -0.081190
C     -1.300025   -0.245443    0.318227
C     -1.747693   -0.091796    1.639784
C     -3.014926   -0.538907    2.020030
C     -3.850976   -1.143714    1.083268
C     -3.422777   -1.300166   -0.233720
C     -2.155574   -0.853116   -0.614137
H     -1.772055    4.110611   -1.067782
H     -0.430540    3.666098   -2.128502
H     -0.124908    4.236479   -0.440542
H      1.658626    2.985951   -1.308665
H      3.563185    1.503135   -1.152581
H      4.554796   -0.658686   -0.601917
H      4.341816   -2.968077    0.216639
H      2.124875   -3.839565    0.912304
H      0.142714   -2.414404    0.789746
H     -1.104216    0.382104    2.377747
H     -3.348638   -0.411482    3.046199
H     -4.838053   -1.489200    1.378592
H     -4.074862   -1.766984   -0.966854
H     -1.832252   -0.976802   -1.645346

