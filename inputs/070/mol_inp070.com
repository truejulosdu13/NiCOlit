%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      5.793431    0.689008    0.927023
O      4.449630    1.011969    1.260920
C      3.466601    0.403143    0.529868
C      3.660929   -0.503778   -0.509457
C      2.560533   -1.053712   -1.176371
C      1.250145   -0.705662   -0.814085
C      0.135249   -1.245570   -1.472960
C     -1.165214   -0.887841   -1.098683
C     -1.371307    0.012722   -0.057428
O     -2.577527    0.454075    0.408722
C     -3.780241   -0.151250   -0.075619
C     -4.812309   -0.031840    1.043955
C     -4.270757    0.588793   -1.316621
C     -0.264006    0.559576    0.599161
C      1.048256    0.211457    0.237027
C      2.163931    0.754751    0.896113
H      6.449218    1.253609    1.596808
H      6.028274    0.989071   -0.099567
H      5.994344   -0.375878    1.084261
H      4.652535   -0.806068   -0.828354
H      2.736634   -1.759454   -1.985249
H      0.269896   -1.950955   -2.290114
H     -1.988185   -1.326615   -1.651348
H     -3.645938   -1.219724   -0.278848
H     -4.456819   -0.544445    1.944569
H     -4.973834    1.015731    1.321936
H     -5.771206   -0.469612    0.749976
H     -4.410283    1.655067   -1.105956
H     -3.541652    0.526115   -2.130224
H     -5.219290    0.175837   -1.673686
H     -0.434444    1.267375    1.408035
H      2.023405    1.464102    1.709182
