%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
O     -0.902770   -0.247082    1.748306
C     -1.343464   -0.286293    0.609707
O     -0.643087   -0.102835   -0.567631
C      0.727805    0.153284   -0.363516
C      1.153447    1.499411   -0.251700
C      0.277692    2.607163   -0.309243
C      0.751964    3.914123   -0.186142
C      2.110343    4.142150   -0.005069
C      2.993688    3.064033    0.049283
C      2.531536    1.742580   -0.075347
C      3.427757    0.660426   -0.031867
C      2.996805   -0.670115   -0.173652
C      3.918344   -1.730949   -0.146062
C      3.497674   -3.052114   -0.298223
C      2.149160   -3.331232   -0.481446
C      1.221776   -2.288435   -0.508793
C      1.624442   -0.942916   -0.351239
C     -2.776747   -0.551704    0.301400
C     -3.651695   -0.765273    1.376117
C     -5.005144   -1.018555    1.145052
C     -5.493127   -1.060129   -0.159886
C     -4.630907   -0.848979   -1.235208
C     -3.275580   -0.595163   -1.008612
H     -0.790080    2.457162   -0.448847
H      0.058875    4.750051   -0.228729
H      2.484690    5.157510    0.094305
H      4.053830    3.264313    0.189874
H      4.489615    0.859540    0.106960
H      4.979066   -1.533703   -0.005560
H      4.223666   -3.860369   -0.273149
H      1.815383   -4.358604   -0.600022
H      0.172662   -2.535128   -0.652432
H     -3.281567   -0.735152    2.399792
H     -5.677202   -1.183148    1.983899
H     -6.547470   -1.257492   -0.339087
H     -5.014652   -0.882033   -2.252303
H     -2.626727   -0.434342   -1.866028

