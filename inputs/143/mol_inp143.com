%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      7.615388    0.724904   -0.066691
O      6.205903    0.817593    0.145297
C      5.500936   -0.215058   -0.381124
O      5.971807   -1.164236   -0.988256
C      4.045185   -0.046788   -0.119481
C      3.530099    1.048062    0.583114
C      2.154048    1.156796    0.803165
C      1.270777    0.176226    0.325964
C     -0.111667    0.274786    0.542643
C     -0.964926   -0.707601    0.040725
O     -2.342038   -0.664272    0.289211
C     -3.028728    0.414633   -0.267886
O     -2.532395    1.302081   -0.950711
C     -4.514015    0.327109    0.102896
C     -4.665083    0.391505    1.628357
C     -5.102781   -0.981929   -0.440045
C     -5.267540    1.511485   -0.524243
C     -0.469992   -1.811354   -0.645146
C      0.903490   -1.919149   -0.863596
C      1.785577   -0.934134   -0.386689
C      3.170095   -1.031359   -0.600470
H      8.016125   -0.177527    0.405925
H      7.843560    0.738067   -1.137192
H      8.085456    1.596057    0.398352
H      4.184943    1.827085    0.965758
H      1.775568    2.018905    1.349380
H     -0.524779    1.116539    1.091732
H     -4.188408   -0.465158    2.118551
H     -4.198043    1.296962    2.032455
H     -5.720593    0.394743    1.922169
H     -4.637739   -1.858378    0.025800
H     -4.943951   -1.066628   -1.521582
H     -6.180349   -1.040776   -0.250212
H     -6.334781    1.479631   -0.277940
H     -4.870794    2.469058   -0.167226
H     -5.176036    1.506954   -1.616837
H     -1.142682   -2.581635   -1.009976
H      1.283377   -2.781122   -1.408532
H      3.574983   -1.882063   -1.147659
