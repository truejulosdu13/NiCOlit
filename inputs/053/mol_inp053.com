%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -3.770178   -2.358691    2.470423
O     -4.052171   -1.678090    1.254548
C     -2.957373   -1.027618    0.725102
N     -3.206967   -0.369711   -0.398179
C     -2.153917    0.257513   -0.906299
O     -2.282518    0.978919   -2.074409
C     -3.609094    0.990510   -2.585738
N     -0.932761    0.263452   -0.393676
C     -0.808823   -0.439557    0.727546
O      0.417223   -0.491129    1.383367
C      1.511388   -0.052815    0.680714
C      2.138089    1.121002    1.094687
C      3.282148    1.565507    0.429304
C      3.814518    0.832060   -0.644737
C      5.057332    1.320844   -1.348567
F      6.166660    1.229195   -0.563315
F      4.970701    2.628497   -1.720611
F      5.347986    0.629082   -2.485984
C      3.186237   -0.363733   -1.029785
C      2.041701   -0.809250   -0.365121
N     -1.784479   -1.100706    1.341017
H     -3.031151   -3.152961    2.319101
H     -3.444522   -1.661906    3.250700
H     -4.697473   -2.828866    2.812123
H     -4.300845    1.468590   -1.883857
H     -3.946025   -0.019669   -2.843441
H     -3.607163    1.583676   -3.505402
H      1.735369    1.691489    1.927236
H      3.761279    2.489281    0.750875
H      3.586773   -0.955583   -1.851540
H      1.568057   -1.739334   -0.666981

