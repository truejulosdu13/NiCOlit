%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      3.147497   -0.131490    1.716214
C      3.479414    0.035298    0.241372
N      2.357968   -0.277680   -0.637694
C      2.222643   -1.638343   -1.154404
C      1.369851   -2.535567   -0.272396
C      1.368346    0.687734   -0.816324
O      1.437070    1.814899   -0.332824
O      0.312248    0.226482   -1.596031
C     -0.735537    1.146923   -1.736233
C     -0.690868    2.023320   -2.823831
C     -1.741636    2.914218   -3.027860
C     -2.831607    2.914652   -2.157751
C     -2.875257    2.020924   -1.082259
C     -1.827941    1.114280   -0.852818
C     -1.901558    0.177522    0.273821
C     -2.667125   -0.994743    0.182647
C     -2.735149   -1.888183    1.254390
C     -2.039441   -1.619618    2.431633
C     -1.278118   -0.457814    2.539637
C     -1.212500    0.435928    1.468697
H      2.282600    0.474929    2.003033
H      3.998769    0.177709    2.330851
H      2.921455   -1.172747    1.959517
H      3.776753    1.075839    0.064956
H      4.336277   -0.589744   -0.031960
H      3.221227   -2.070065   -1.282677
H      1.773044   -1.590650   -2.152646
H      0.371852   -2.115975   -0.121065
H      1.256963   -3.520897   -0.735445
H      1.825317   -2.679740    0.710528
H      0.158541    2.014879   -3.499845
H     -1.711565    3.608514   -3.863381
H     -3.651061    3.611681   -2.314533
H     -3.734982    2.036126   -0.415018
H     -3.211269   -1.218581   -0.732490
H     -3.330533   -2.793199    1.168437
H     -2.091980   -2.314529    3.265498
H     -0.735195   -0.243465    3.456266
H     -0.614511    1.341173    1.565967
