%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      4.636601   -0.021283   -0.537362
C      3.227519   -0.534981   -0.512667
O      2.873770   -1.596569   -1.006548
O      2.428653    0.391894    0.149343
C      1.074601    0.054040    0.268182
C      0.670567   -1.061832    0.994819
C     -0.692176   -1.327788    1.133392
C     -1.652991   -0.473966    0.564238
C     -3.027026   -0.729920    0.696863
C     -3.971034    0.128708    0.131007
C     -3.554433    1.254072   -0.574098
C     -2.192419    1.523958   -0.716471
C     -1.229378    0.670036   -0.154955
C      0.143162    0.927929   -0.290487
H      4.676480    0.925378   -1.081923
H      5.002981    0.105368    0.484452
H      5.276908   -0.745411   -1.049238
H      1.400700   -1.728014    1.444000
H     -1.001218   -2.209177    1.690827
H     -3.373003   -1.604080    1.243930
H     -5.031444   -0.082752    0.241192
H     -4.289225    1.922484   -1.015242
H     -1.887277    2.407543   -1.272527
H      0.489679    1.804366   -0.830727

