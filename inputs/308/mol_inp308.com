%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -1.124728    2.100297    2.316681
O     -0.973532    1.124025    1.295636
C     -1.521393    1.395325    0.072570
C     -2.263930    2.546754   -0.233304
C     -2.789057    2.761399   -1.506270
C     -2.588418    1.828115   -2.522616
C     -1.850666    0.660529   -2.252881
C     -1.478625   -0.453239   -3.052415
C     -0.752138   -1.309195   -2.262059
N     -0.667230   -0.758756   -1.005595
C      0.026986   -1.407736    0.101170
C      1.459999   -0.880790    0.266513
N      2.197965   -1.498848    1.390835
C      3.428835   -0.745905    1.649575
C      2.525501   -2.900994    1.124669
C     -1.331969    0.456804   -0.966878
H     -0.658476    3.048677    2.030025
H     -0.608033    1.732872    3.208595
H     -2.179177    2.238535    2.577020
H     -2.453649    3.308359    0.516632
H     -3.360028    3.663548   -1.710926
H     -2.998937    2.001211   -3.513053
H     -1.720162   -0.608721   -4.095382
H     -0.292675   -2.262146   -2.489597
H      0.024828   -2.481583   -0.117563
H     -0.569967   -1.272711    1.008312
H      1.402438    0.202034    0.436058
H      2.016713   -1.009630   -0.671966
H      3.967188   -1.169216    2.505054
H      4.101587   -0.743461    0.784222
H      3.197904    0.292685    1.911809
H      1.620640   -3.509832    1.036185
H      3.117342   -3.021274    0.210112
H      3.094865   -3.327133    1.958830

