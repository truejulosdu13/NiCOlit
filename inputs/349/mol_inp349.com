%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      5.318983    1.661184   -1.014959
C      4.733873    0.266813   -1.170299
N      4.467582   -0.353726    0.124419
C      5.511722   -1.177836    0.721503
C      5.404902   -2.628713    0.280234
C      3.264462   -0.192856    0.804569
O      3.020416   -0.736235    1.877214
O      2.384656    0.651322    0.128701
C      1.152229    0.918998    0.742748
C      1.083476    1.543695    1.989429
C     -0.163984    1.858100    2.531965
C     -1.329833    1.561456    1.823206
C     -1.236384    0.936469    0.578286
O     -2.375457    0.684315   -0.201234
C     -3.367504   -0.124861    0.350919
O     -3.300582   -0.649023    1.458207
N     -4.454019   -0.283335   -0.503620
C     -4.540936    0.375895   -1.803647
C     -3.897060   -0.456891   -2.900358
C     -5.555463   -1.145974   -0.093082
C     -6.630360   -0.369983    0.650950
C      0.000226    0.631005    0.015676
H      6.263723    1.631971   -0.462437
H      5.510470    2.104802   -1.996751
H      4.634718    2.319108   -0.469463
H      3.805161    0.323590   -1.747995
H      5.421301   -0.372964   -1.734584
H      5.434986   -1.128857    1.813509
H      6.488741   -0.764400    0.448186
H      5.500576   -2.719004   -0.806586
H      4.438711   -3.057118    0.566349
H      6.195507   -3.226511    0.743906
H      1.988970    1.780116    2.540575
H     -0.227685    2.334169    3.506864
H     -2.298545    1.811973    2.245685
H     -4.049738    1.353439   -1.752509
H     -5.595355    0.561027   -2.036511
H     -2.836844   -0.633960   -2.692356
H     -4.385664   -1.432379   -2.991686
H     -3.975686    0.056568   -3.863464
H     -5.173160   -1.941332    0.556547
H     -5.976563   -1.629717   -0.981260
H     -7.442765   -1.038514    0.951764
H     -7.052868    0.420724    0.022640
H     -6.223343    0.101735    1.551372
H      0.064408    0.175714   -0.966140
