%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      2.861169   -1.816603    0.459488
O      1.481630   -1.689002    0.151679
C      0.928927   -0.425960    0.168171
C      1.656281    0.734476    0.472259
C      1.030574    1.977743    0.469407
C     -0.322215    2.068247    0.163407
C     -1.060038    0.914172   -0.142402
C     -2.424742    1.002070   -0.451524
C     -3.144762   -0.145426   -0.752415
C     -2.472159   -1.348731   -0.733930
N     -1.160418   -1.471029   -0.441184
C     -0.443767   -0.359062   -0.145201
H      3.121019   -2.877818    0.397731
H      3.070885   -1.483057    1.481048
H      3.479836   -1.281124   -0.268063
H      2.712732    0.699875    0.715402
H      1.597780    2.874571    0.705614
H     -0.799698    3.045658    0.164531
H     -2.931920    1.963489   -0.459225
H     -4.200065   -0.101442   -0.994231
H     -2.981052   -2.281047   -0.960561
