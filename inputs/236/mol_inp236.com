%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -2.668775    1.665615   -0.377433
C     -2.196299    0.540931   -1.285438
N     -1.691960   -0.617235   -0.550025
C     -2.601256   -1.721676   -0.261736
C     -3.331291   -1.561752    1.062032
C     -0.416153   -0.643895    0.011300
O     -0.004219   -1.574699    0.697314
O      0.319186    0.496964   -0.298256
C      1.605073    0.524227    0.254067
C      1.759111    0.984406    1.563028
C      3.036178    1.061755    2.113264
C      4.145044    0.691608    1.351618
C      3.981614    0.248754    0.035289
C      2.705668    0.171781   -0.537294
C      2.529425   -0.311810   -1.945303
H     -2.980929    2.526859   -0.976490
H     -1.876452    1.993583    0.302555
H     -3.522273    1.355155    0.230785
H     -1.396623    0.917883   -1.933135
H     -3.008932    0.218983   -1.945613
H     -3.316398   -1.821571   -1.085336
H     -2.021538   -2.651805   -0.232012
H     -3.939650   -2.448385    1.266542
H     -3.998933   -0.696323    1.046126
H     -2.629424   -1.430169    1.891661
H      0.892398    1.269971    2.150763
H      3.168133    1.405970    3.135370
H      5.141154    0.746128    1.783742
H      4.860204   -0.035956   -0.539136
H      2.065784   -1.303275   -1.947882
H      1.902166    0.382281   -2.514143
H      3.489965   -0.384304   -2.466223
