%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      5.689698    0.056787    0.611721
C      4.249422    0.450392    0.468512
O      3.866714    1.588085    0.233061
O      3.459332   -0.681128    0.644249
C      2.079583   -0.468113    0.544639
C      1.405130    0.316406    1.474883
C      0.021372    0.461390    1.368766
C     -0.694695   -0.188212    0.347810
C     -2.089375   -0.047676    0.231899
C     -2.793991   -0.703320   -0.785837
C     -4.274511   -0.577616   -0.939494
C     -5.033830    0.322215    0.000107
O     -4.854552   -1.206075   -1.822878
C     -2.097686   -1.508076   -1.694068
C     -0.712636   -1.655721   -1.591742
C      0.002345   -1.001744   -0.578473
C      1.393646   -1.140050   -0.465321
H      5.951703   -0.675993   -0.155543
H      5.865091   -0.350549    1.610557
H      6.321089    0.940209    0.480299
H      1.944369    0.821647    2.270563
H     -0.494265    1.088879    2.092699
H     -2.610451    0.579713    0.949742
H     -4.955228   -0.048658    1.025243
H     -4.664520    1.348283   -0.074699
H     -6.091736    0.323976   -0.280901
H     -2.630282   -2.025511   -2.490527
H     -0.198009   -2.286258   -2.314026
H      1.946273   -1.765420   -1.161243

