%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -4.284717    0.641871    0.626472
C     -2.801752    0.399647    0.547835
C     -1.942862    1.122498    1.394427
C     -0.563392    0.927394    1.346684
C     -0.039364    0.015337    0.438981
O      1.343443   -0.202281    0.452200
C      2.031406    0.192976   -0.693036
O      1.509215    0.729912   -1.665362
N      3.393687   -0.081171   -0.612757
C      4.025316   -0.557072    0.605148
C      4.269091    0.422214   -1.653305
C     -0.869143   -0.730222   -0.396659
C     -2.258845   -0.540084   -0.355704
C     -3.136776   -1.337988   -1.280145
H     -4.527715    1.412496    1.365806
H     -4.804096   -0.274952    0.922587
H     -4.666549    0.983036   -0.340923
H     -2.347243    1.845461    2.099460
H      0.091837    1.484936    2.008045
H      4.858949   -1.215966    0.345106
H      4.399873    0.306293    1.162806
H      3.328839   -1.113842    1.236367
H      3.720766    0.647401   -2.571905
H      4.749371    1.337227   -1.294027
H      5.031936   -0.329604   -1.875501
H     -0.430646   -1.449249   -1.083256
H     -3.851013   -1.935923   -0.705367
H     -3.679224   -0.671536   -1.958081
H     -2.550393   -2.028807   -1.895373
