%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      8.011800   -1.394660    0.878824
C      7.429755   -0.130013    0.268084
N      6.164214    0.237886    0.891716
C      6.199667    1.106334    2.065124
C      6.090532    2.573177    1.681731
C      4.978613   -0.253326    0.352580
O      4.944878   -1.013593   -0.610650
O      3.856794    0.220161    1.027754
C      2.644561   -0.252817    0.522652
C      2.049396   -1.354121    1.120034
C      0.814329   -1.797080    0.646612
C      0.145172   -1.128055   -0.399187
C      0.787151   -0.031859   -1.014412
C      2.024346    0.419831   -0.524809
C      0.185435    0.694343   -2.189384
C     -0.910403   -0.106115   -2.883080
C     -1.921925   -0.693099   -1.884990
C     -1.190407   -1.659286   -0.919084
C     -2.133819   -2.204622    0.177872
C     -3.560796   -1.667354    0.099966
C     -3.610594   -0.123768    0.036736
C     -3.062911    0.461369    1.356525
C     -2.769319    0.400230   -1.174489
C     -3.794647    1.018363   -2.152583
C     -5.137479    0.421838   -1.761058
C     -5.059115    0.386817   -0.238565
O     -6.067048   -0.469670    0.302696
C     -7.095508    0.416035    0.737795
C     -6.330262    1.630946    1.182066
O     -5.269314    1.730776    0.233014
H      8.959260   -1.650091    0.394654
H      8.198804   -1.266416    1.949735
H      7.327087   -2.240574    0.759119
H      7.274445   -0.288886   -0.804905
H      8.131740    0.704645    0.371949
H      5.378949    0.843895    2.741232
H      7.132090    0.924716    2.611112
H      6.915044    2.870857    1.025806
H      6.120644    3.203143    2.575960
H      5.154141    2.775182    1.151594
H      2.542340   -1.873466    1.935275
H      0.382496   -2.685576    1.101702
H      2.513580    1.279160   -0.975047
H     -0.203305    1.659837   -1.846244
H      0.969134    0.909910   -2.926613
H     -0.450583   -0.932443   -3.442375
H     -1.404270    0.525124   -3.629237
H     -2.617517   -1.308026   -2.475718
H     -0.899661   -2.527695   -1.531379
H     -2.173515   -3.298964    0.096947
H     -1.752601   -2.006674    1.185098
H     -4.057484   -2.105955   -0.774627
H     -4.116537   -2.029847    0.974320
H     -2.018891    0.182243    1.527531
H     -3.088153    1.556580    1.357251
H     -3.649784    0.111446    2.212778
H     -2.108682    1.205154   -0.833914
H     -3.816885    2.106646   -2.013135
H     -3.576827    0.834604   -3.208215
H     -5.964552    1.039528   -2.126798
H     -5.261533   -0.583381   -2.179607
H     -7.745504    0.636042   -0.116151
H     -7.688132   -0.038699    1.535258
H     -6.931181    2.543692    1.187713
H     -5.887253    1.485621    2.173015

