%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      4.696487    0.975394    0.727991
O      3.323650    1.324070    0.816814
C      2.481179    0.245505    0.406664
O      1.119732    0.645205    0.514128
C      0.151539   -0.264659    0.161273
C      0.419263   -1.559813   -0.297193
C     -0.626428   -2.418175   -0.633595
C     -1.947297   -1.992443   -0.515706
C     -2.245387   -0.701205   -0.059457
C     -3.579581   -0.275734    0.057983
C     -3.885817    1.007234    0.511044
C     -2.862030    1.881871    0.852715
C     -1.531680    1.472675    0.741022
C     -1.190925    0.179858    0.285340
H      5.291871    1.833158    1.052193
H      4.965885    0.734703   -0.304973
H      4.920980    0.127917    1.382900
H      2.669703   -0.611398    1.064432
H      2.714951   -0.000039   -0.636190
H      1.430091   -1.936925   -0.405855
H     -0.412316   -3.423000   -0.989118
H     -2.746898   -2.679654   -0.784032
H     -4.395574   -0.945520   -0.204516
H     -4.922422    1.321623    0.596473
H     -3.092175    2.883193    1.206552
H     -0.746801    2.176161    1.014807

