%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -5.503421   -0.851967   -0.782425
C     -5.139550    0.284484    0.182133
C     -5.483791   -0.105487    1.625659
C     -5.940645    1.539563   -0.200233
C     -3.639645    0.585937    0.076639
O     -3.169344    1.664269   -0.264093
O     -2.908434   -0.550205    0.422858
C     -1.518155   -0.405855    0.380774
C     -0.812120   -1.184985   -0.530937
C      0.582203   -1.107480   -0.562320
C      1.269971   -0.256882    0.316427
C      2.737238   -0.170657    0.333721
C      3.499173   -0.182160   -0.772189
C      4.966306   -0.095327   -0.753942
C      5.700751   -0.849139   -1.678621
C      7.095535   -0.773293   -1.711196
C      7.769246    0.066393   -0.826950
C      7.049219    0.833836    0.086152
C      5.654291    0.757391    0.119314
C      0.538158    0.492163    1.250053
C     -0.856205    0.418656    1.286880
H     -6.581749   -1.046257   -0.777625
H     -4.998660   -1.787022   -0.514638
H     -5.207977   -0.606570   -1.809039
H     -6.561230   -0.262865    1.747827
H     -5.173647    0.675894    2.329001
H     -4.979272   -1.030723    1.925964
H     -5.702032    2.380169    0.461734
H     -5.716398    1.857053   -1.225254
H     -7.019400    1.358513   -0.134931
H     -1.336478   -1.849872   -1.210190
H      1.125904   -1.727944   -1.270365
H      3.188154   -0.086428    1.319503
H      3.049811   -0.264483   -1.758813
H      5.192681   -1.509892   -2.377260
H      7.654363   -1.368882   -2.428217
H      8.854041    0.127036   -0.853103
H      7.571717    1.497750    0.769848
H      5.112136    1.380637    0.826026
H      1.049071    1.148018    1.951428
H     -1.411818    1.006614    2.010772

