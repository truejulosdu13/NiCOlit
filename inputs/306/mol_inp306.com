%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
O     -1.329657   -0.445257    1.973360
C     -1.183265   -0.754096    0.589108
C     -2.185642    0.050815   -0.227444
C     -1.659066    1.466149   -0.396344
O     -0.442109    1.481507   -1.148218
C      0.516052    0.630375   -0.662293
C      1.824164    0.875596   -1.081021
C      2.865625    0.062255   -0.635676
C      2.597784   -0.999868    0.225684
C      1.286841   -1.256076    0.631350
C      0.229942   -0.451020    0.179232
H     -2.196113   -0.783286    2.253674
H     -1.384731   -1.824265    0.460058
H     -2.296330   -0.407688   -1.218141
H     -3.166732    0.068163    0.258940
H     -1.501418    1.968312    0.566508
H     -2.388413    2.062279   -0.955166
H      2.031324    1.708703   -1.746841
H      3.884706    0.261739   -0.955362
H      3.409159   -1.628968    0.582203
H      1.087879   -2.085367    1.306391
