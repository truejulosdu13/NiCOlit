%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      7.350256    0.362246   -1.027585
O      6.499880    0.853397    0.000703
C      5.156602    0.657984   -0.162875
C      4.362796    1.160542    0.868655
C      2.971086    1.027847    0.828817
C      2.339393    0.388291   -0.247553
C      0.875440    0.245385   -0.293584
C      0.139357    0.705974   -1.394149
C     -1.251718    0.572303   -1.438432
C     -1.947269   -0.028678   -0.380468
C     -3.343019   -0.170995   -0.412076
C     -4.005636   -0.760166    0.664424
O     -5.390021   -0.967759    0.637276
C     -6.168735    0.184246    0.597606
C     -7.614058   -0.215517    0.563470
O     -5.775043    1.342235    0.593159
C     -3.304648   -1.246174    1.762638
C     -1.916807   -1.109538    1.801209
C     -1.224485   -0.501212    0.739441
C      0.174098   -0.356490    0.765616
C      3.149378   -0.111916   -1.279418
C      4.544043    0.019664   -1.239356
H      7.270956   -0.726055   -1.119245
H      8.381918    0.598259   -0.749341
H      7.141686    0.856374   -1.982284
H      4.830172    1.663615    1.711642
H      2.381025    1.441609    1.644053
H      0.646325    1.190165   -2.226598
H     -1.787854    0.949451   -2.306570
H     -3.912105    0.182419   -1.267335
H     -8.237221    0.682632    0.526489
H     -7.812332   -0.813284   -0.329688
H     -7.864860   -0.776643    1.467100
H     -3.828569   -1.725547    2.583784
H     -1.375193   -1.484149    2.667041
H      0.731983   -0.729405    1.623471
H      2.696245   -0.624469   -2.126078
H      5.116934   -0.390465   -2.063962
