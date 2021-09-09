%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -5.242294   -1.822981   -1.189201
C     -4.757019   -0.394038   -1.005411
N     -3.817661   -0.281249    0.103805
C     -4.352746   -0.022922    1.437895
C     -4.410556    1.465533    1.741033
C     -2.456914   -0.428890   -0.146407
O     -2.000973   -0.678856   -1.257988
O     -1.687004   -0.257039    1.003590
C     -0.286693   -0.339215    0.891592
C      0.360542   -1.186981    1.782807
C      1.755631   -1.273776    1.771463
C      2.499570   -0.492022    0.859352
C      3.899079   -0.341284    0.605446
C      5.057051   -0.917990    1.170574
C      6.319198   -0.544292    0.699090
C      6.441079    0.392443   -0.321361
C      5.310294    0.977303   -0.899351
C      4.045245    0.592797   -0.422333
N      2.773440    0.982138   -0.794132
C      2.474525    1.921363   -1.854081
C      1.818429    0.357447   -0.015680
C      0.416813    0.466133   -0.008442
H     -5.941651   -1.882197   -2.028742
H     -4.405766   -2.499897   -1.391510
H     -5.753979   -2.184831   -0.291594
H     -5.602618    0.278230   -0.823069
H     -4.271993   -0.058859   -1.929089
H     -5.352960   -0.464195    1.509488
H     -3.729678   -0.523553    2.186636
H     -5.049295    1.988766    1.022065
H     -3.414936    1.918361    1.692749
H     -4.814885    1.633971    2.743781
H     -0.212243   -1.781650    2.487876
H      2.264928   -1.937007    2.464195
H      4.969882   -1.647572    1.970095
H      7.209473   -0.989162    1.136012
H      7.429467    0.674698   -0.676576
H      5.421628    1.707390   -1.694242
H      2.393345    2.916985   -1.411258
H      1.536743    1.638546   -2.338313
H      3.269382    1.899613   -2.602843
H     -0.103880    1.138741   -0.680446
