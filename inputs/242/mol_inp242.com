%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      5.241544   -0.810579    2.308726
N      4.773202   -0.709963    0.940061
C      5.725447   -1.037466   -0.106347
C      3.578484   -0.042831    0.687567
O      2.833979    0.365434    1.573777
O      3.335484    0.111041   -0.675089
C      2.134216    0.754548   -0.988742
C      2.189556    2.059431   -1.465468
C      1.006373    2.698756   -1.835498
C     -0.228953    2.034080   -1.741592
C     -1.425811    2.665911   -2.115013
C     -2.646171    1.995409   -2.016733
C     -2.684702    0.691170   -1.530698
O     -3.916522    0.077964   -1.487926
C     -4.239359   -0.620710   -0.346436
C     -4.858096   -1.859396   -0.519667
C     -5.240083   -2.608687    0.593488
C     -5.016627   -2.111532    1.877935
C     -4.417968   -0.863057    2.050585
C     -4.035013   -0.111668    0.938578
C     -1.502185    0.034865   -1.178615
C     -0.268308    0.698191   -1.268804
C      0.926933    0.062526   -0.902241
H      4.430592   -0.664658    3.027712
H      6.004813   -0.044400    2.474821
H      5.676338   -1.801826    2.466861
H      6.471976   -0.239463   -0.159253
H      6.216952   -1.983809    0.137667
H      5.244047   -1.140340   -1.081405
H      3.140236    2.576934   -1.546092
H      1.053924    3.722106   -2.200571
H     -1.416463    3.687200   -2.488492
H     -3.565793    2.492691   -2.312723
H     -5.037096   -2.241979   -1.520643
H     -5.713769   -3.577680    0.459803
H     -5.314342   -2.694928    2.745427
H     -4.252010   -0.472575    3.051211
H     -3.584810    0.866432    1.082862
H     -1.541570   -0.995228   -0.832906
H      0.921557   -0.961912   -0.540884

