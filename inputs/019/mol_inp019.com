%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -4.747416    0.171873   -0.402521
C     -3.593975   -0.500010    0.349235
C     -3.767149   -0.324881    1.862600
Si    -1.816982    0.122827   -0.264335
C     -1.747351    1.983174   -0.019664
C     -1.685458   -0.280177   -2.094941
O     -0.646882   -0.666204    0.625267
C      0.719516   -0.640406    0.654218
C      1.316347   -1.479605    1.593278
C      2.706370   -1.533559    1.710156
C      3.522093   -0.745301    0.885696
C      4.921999   -0.785480    0.988027
C      5.720379    0.005254    0.160376
C      5.129942    0.845098   -0.779031
C      3.739609    0.896178   -0.893216
C      2.920538    0.107380   -0.068448
C      1.518222    0.149196   -0.173024
H     -4.668497    0.011459   -1.482928
H     -5.708848   -0.241396   -0.078436
H     -4.771095    1.252535   -0.224298
H     -3.605809   -1.572859    0.120442
H     -4.726275   -0.740212    2.190466
H     -2.976302   -0.842457    2.416767
H     -3.742506    0.731757    2.151685
H     -2.548666    2.477676   -0.576179
H     -1.856885    2.237524    1.038544
H     -0.793957    2.388310   -0.370347
H     -1.796253   -1.356067   -2.259011
H     -2.464343    0.236197   -2.663431
H     -0.715304    0.028616   -2.494195
H      0.696129   -2.095858    2.238926
H      3.146329   -2.196773    2.451058
H      5.401506   -1.435767    1.716151
H      6.802722   -0.035602    0.250209
H      5.750565    1.461476   -1.424129
H      3.298348    1.559272   -1.633925
H      1.069340    0.806811   -0.909542

