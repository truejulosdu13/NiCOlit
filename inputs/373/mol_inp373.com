%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      7.022845    0.582099    1.306239
O      6.959346    1.089722   -0.019178
C      5.877725    0.514306   -0.756330
O      4.618161    0.905742   -0.219253
C      3.499693    0.403317   -0.824783
C      3.471055   -0.401726   -1.961533
C      2.248129   -0.847193   -2.476168
C      1.035625   -0.495063   -1.864888
C     -0.198064   -0.933927   -2.376811
C     -1.395332   -0.573818   -1.757775
C     -1.356211    0.214651   -0.612146
O     -2.535600    0.642551    0.007818
C     -3.313683   -0.361878    0.583601
O     -3.045072   -1.556790    0.595998
C     -4.569749    0.255368    1.210290
C     -5.393377    0.957228    0.122481
C     -4.160922    1.253110    2.301936
C     -5.422954   -0.855610    1.843442
C     -0.148545    0.679421   -0.094919
C      1.060222    0.323020   -0.711851
C      2.296147    0.762687   -0.210065
H      6.126705    0.854889    1.871418
H      7.147972   -0.505029    1.297919
H      7.890339    1.026743    1.801602
H      5.988709   -0.577371   -0.755099
H      5.972265    0.888650   -1.781606
H      4.375597   -0.701984   -2.479727
H      2.250718   -1.474487   -3.365348
H     -0.234215   -1.561414   -3.264551
H     -2.341881   -0.916010   -2.164640
H     -5.640555    0.265947   -0.691453
H     -6.331526    1.352096    0.528094
H     -4.846272    1.797041   -0.320436
H     -3.523271    0.774241    3.053949
H     -3.596773    2.096594    1.888318
H     -5.038855    1.662861    2.813699
H     -6.331138   -0.447961    2.301446
H     -5.729178   -1.596357    1.095600
H     -4.866059   -1.388993    2.622615
H     -0.153103    1.314272    0.786765
H      2.331081    1.394575    0.675332
