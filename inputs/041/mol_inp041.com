%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      3.577209   -1.951321   -1.377210
O      2.297317   -1.338533   -1.450039
C      2.107357   -0.348833   -0.509106
N      3.096086   -0.025387    0.316818
C      2.791073    0.947403    1.168459
O      3.731674    1.378316    2.080113
C      4.972816    0.693375    1.980300
N      1.630059    1.579338    1.245131
C      0.723622    1.160692    0.370202
O     -0.496056    1.827403    0.446867
C     -1.531381    1.287639   -0.284046
C     -1.704794    1.727887   -1.599734
C     -2.757898    1.230426   -2.358719
C     -3.644293    0.309416   -1.798094
C     -3.498416   -0.120987   -0.467696
C     -4.403546   -1.035784    0.095676
C     -4.269395   -1.454582    1.419799
C     -3.229593   -0.962986    2.200384
C     -2.319750   -0.055323    1.655680
C     -2.430911    0.377575    0.316187
N      0.906376    0.206008   -0.536898
H      3.731015   -2.438824   -0.408092
H      3.621159   -2.726137   -2.148795
H      4.375587   -1.230365   -1.585060
H      5.438335    0.853708    1.001508
H      5.644651    1.107044    2.738675
H      4.855187   -0.375802    2.188781
H     -1.014064    2.446301   -2.031549
H     -2.890708    1.557250   -3.386551
H     -4.461020   -0.068652   -2.409737
H     -5.226922   -1.429095   -0.496639
H     -4.979197   -2.161860    1.840483
H     -3.123921   -1.283180    3.233533
H     -1.517656    0.317869    2.289011

