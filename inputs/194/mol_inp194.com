%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -4.048749    1.783081   -0.891992
C     -3.409518    0.405062   -0.953826
N     -2.336402    0.262754    0.023045
C     -2.679312   -0.214184    1.360211
C     -2.578016   -1.727383    1.461357
C     -1.038264    0.598577   -0.349641
O     -0.752242    1.046277   -1.455959
O     -0.120518    0.364674    0.670178
C      1.203745    0.694358    0.341786
C      1.595649    2.028908    0.469013
C      2.912774    2.379541    0.196855
C      3.830525    1.397484   -0.183576
C      3.448221    0.048140   -0.294135
C      4.377480   -0.937479   -0.666319
C      3.999571   -2.276843   -0.769446
C      2.687208   -2.650115   -0.502027
C      1.750717   -1.683529   -0.132304
C      2.110540   -0.323383   -0.023707
H     -3.311976    2.568356   -1.090524
H     -4.481019    1.973947    0.095525
H     -4.846641    1.865928   -1.636254
H     -3.009409    0.240391   -1.960617
H     -4.158493   -0.373755   -0.773219
H     -2.012278    0.245565    2.097125
H     -3.695828    0.115060    1.602284
H     -3.252982   -2.215717    0.751056
H     -2.843880   -2.059625    2.469496
H     -1.561803   -2.071335    1.245009
H      0.878096    2.786921    0.767266
H      3.228513    3.416131    0.278543
H      4.856205    1.694426   -0.393137
H      5.409106   -0.666468   -0.880025
H      4.730769   -3.026469   -1.060045
H      2.388911   -3.691850   -0.583670
H      0.729299   -1.997449    0.067491
