%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      4.032661   -0.531558    0.418905
O      3.165509    0.461069   -0.113461
C      1.820046    0.214337   -0.039676
C      1.227026   -0.921044    0.509805
C     -0.166565   -1.072740    0.533929
C     -0.953345   -0.047750   -0.011709
C     -0.385359    1.100153   -0.568435
C     -1.452495    1.922160   -1.022373
C     -2.632308    1.272437   -0.739450
N     -2.321488    0.085536   -0.130134
C     -3.291177   -0.886289    0.322861
C      1.017528    1.227651   -0.580027
H      3.866727   -0.666286    1.492858
H      5.061463   -0.184442    0.282697
H      3.927235   -1.478679   -0.120455
H      1.823251   -1.721747    0.935359
H     -0.614216   -1.962256    0.965056
H     -1.362929    2.886496   -1.504603
H     -3.658547    1.561852   -0.923668
H     -3.111030   -1.826378   -0.204764
H     -4.300146   -0.527307    0.104267
H     -3.171250   -1.017676    1.401268
H      1.479412    2.112460   -1.008250

