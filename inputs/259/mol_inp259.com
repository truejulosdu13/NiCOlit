%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -3.850425   -0.065135    0.983007
C     -2.548504   -0.013747    0.164547
C     -2.723061    1.033598   -0.944898
C     -2.307679   -1.406050   -0.437218
O     -1.536103    0.360683    1.127835
C     -0.257059    0.471543    0.662491
C      0.217166    1.707738    0.236970
C      1.542742    1.836819   -0.181247
C      2.414306    0.736206   -0.148535
C      3.750120    0.847344   -0.566680
C      4.607538   -0.252974   -0.515318
C      4.141936   -1.476778   -0.043333
C      2.817021   -1.604335    0.376829
C      1.940075   -0.508656    0.327670
C      0.604423   -0.621176    0.745260
H     -3.766820   -0.789221    1.802003
H     -4.711320   -0.340966    0.364833
H     -4.056109    0.905998    1.448449
H     -2.839760    2.037607   -0.522088
H     -3.600331    0.816098   -1.564093
H     -1.854027    1.062616   -1.610669
H     -3.164640   -1.735548   -1.035024
H     -2.130249   -2.148260    0.348939
H     -1.430765   -1.414458   -1.093213
H     -0.438166    2.573433    0.233340
H      1.894048    2.807409   -0.523467
H      4.133967    1.796281   -0.934307
H      5.639426   -0.152494   -0.841466
H      4.809579   -2.333324   -0.000544
H      2.471847   -2.568445    0.743511
H      0.230822   -1.561806    1.140264

