%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      6.955979    0.945908   -0.300934
O      6.122728   -0.187222   -0.508266
C      4.779797   -0.007713   -0.322892
C      4.156272    1.169352    0.076863
C      2.764974    1.213098    0.221710
C      1.964586    0.089254   -0.046177
C      2.602316   -1.114109   -0.401855
C      3.999520   -1.140856   -0.556891
C      1.829170   -2.383634   -0.661542
C      0.425940   -2.360351   -0.059317
C     -0.280941   -1.061403   -0.459907
C      0.453264    0.150985    0.152402
C     -0.202746    1.472204   -0.310439
C     -1.487312    1.298282   -1.122898
C     -2.498793    0.245158   -0.598563
C     -3.450414   -0.085869   -1.769001
C     -3.283969    0.778802    0.638600
O     -4.614849    0.248176    0.652931
C     -5.519955    1.033417    1.417903
C     -2.525091    0.231481    1.841773
C     -2.104175   -1.161125    1.396160
C     -1.786511   -1.055199   -0.106954
H      6.906615    1.286854    0.738474
H      6.704393    1.754410   -0.995352
H      7.988032    0.643304   -0.502823
H      4.718521    2.072000    0.289246
H      2.312630    2.144521    0.555427
H      4.492722   -2.063222   -0.857869
H      1.760988   -2.527548   -1.747126
H      2.371822   -3.242213   -0.247825
H     -0.134435   -3.232511   -0.415447
H      0.497741   -2.445340    1.031401
H     -0.208405   -0.999193   -1.556723
H      0.337588    0.096870    1.243776
H     -0.410662    2.094633    0.568670
H      0.484176    2.062397   -0.929646
H     -1.980621    2.273796   -1.224284
H     -1.184192    1.026694   -2.144110
H     -2.892244   -0.449572   -2.639484
H     -4.168831   -0.869596   -1.504503
H     -4.014850    0.801066   -2.077832
H     -3.324103    1.874365    0.646729
H     -5.192876    1.114999    2.457912
H     -5.623158    2.029774    0.977483
H     -6.497493    0.544130    1.398701
H     -1.651294    0.847898    2.066926
H     -3.131834    0.172598    2.750023
H     -2.947376   -1.850629    1.532814
H     -1.282315   -1.544086    2.005659
H     -2.230329   -1.935035   -0.592917
