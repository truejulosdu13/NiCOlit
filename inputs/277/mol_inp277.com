%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      4.499984   -0.158322    1.220137
O      5.009223   -0.846892    0.085236
C      4.408224   -0.434773   -1.148117
C      2.951387   -0.885569   -1.289553
O      2.105022    0.065048   -0.634246
C      0.759991   -0.181705   -0.652817
C      0.133619   -1.273226   -1.248847
C     -1.257248   -1.403512   -1.186330
C     -2.040955   -0.448619   -0.523620
C     -1.416057    0.665170    0.060507
C     -0.017934    0.778220   -0.000403
C     -2.204821    1.712905    0.806546
C     -3.697631    1.686249    0.483426
C     -4.229787    0.260073    0.541879
C     -3.541174   -0.610281   -0.507855
H      4.510447    0.924302    1.062103
H      3.490414   -0.502491    1.458722
H      5.143592   -0.389713    2.073502
H      4.997943   -0.912686   -1.937339
H      4.511270    0.650530   -1.261372
H      2.826709   -1.876797   -0.838496
H      2.696527   -0.909352   -2.355431
H      0.692938   -2.042692   -1.769510
H     -1.729484   -2.261496   -1.660533
H      0.476616    1.627329    0.466907
H     -2.063908    1.542293    1.881266
H     -1.812573    2.710435    0.575220
H     -3.870725    2.101815   -0.517440
H     -4.239002    2.321712    1.193172
H     -5.312404    0.252781    0.372995
H     -4.057463   -0.155233    1.542985
H     -3.801485   -1.658999   -0.321109
H     -3.921257   -0.346503   -1.502819
