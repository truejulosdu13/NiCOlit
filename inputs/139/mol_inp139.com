%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -3.266892    0.988966   -1.077541
C     -2.751009    0.078198    0.044233
C     -2.605976    0.867118    1.352321
C     -3.754938   -1.065610    0.260914
C     -1.391530   -0.506775   -0.357563
O     -1.164267   -1.701972   -0.498850
O     -0.473233    0.525935   -0.538067
C      0.811286    0.093017   -0.900587
C      1.035061   -0.270061   -2.230685
C      2.309375   -0.662688   -2.623056
C      3.351689   -0.675277   -1.693245
C      3.139196   -0.289873   -0.357245
C      4.193385   -0.292970    0.571158
C      3.984515    0.097394    1.894468
C      2.718934    0.497349    2.308454
C      1.659705    0.504765    1.399554
C      1.847545    0.110370    0.057502
H     -4.267452    1.372407   -0.848642
H     -2.609852    1.852310   -1.231627
H     -3.320753    0.448685   -2.029721
H     -1.945898    1.733393    1.231314
H     -3.575976    1.238468    1.701245
H     -2.178011    0.241296    2.143928
H     -4.739619   -0.681599    0.550191
H     -3.416476   -1.746847    1.050283
H     -3.882950   -1.660219   -0.651114
H      0.222623   -0.257566   -2.950542
H      2.495029   -0.961316   -3.651383
H      4.340882   -0.987906   -2.021926
H      5.191561   -0.601089    0.267463
H      4.811148    0.088551    2.600015
H      2.553411    0.802235    3.338358
H      0.679486    0.821311    1.746782
