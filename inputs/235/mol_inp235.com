%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -5.141851    0.816753    1.970355
C     -5.195117    0.234240    0.571527
O     -3.859783   -0.102726    0.177505
C     -3.763918   -0.631635   -1.067711
O     -4.696222   -0.839694   -1.828369
C     -2.347339   -0.942257   -1.399453
C     -2.070153   -1.507454   -2.655155
C     -0.758069   -1.826632   -3.017392
C      0.286872   -1.579397   -2.130141
C      0.007049   -1.002092   -0.893289
O      1.063502   -0.804556    0.000608
C      1.488532    0.517974    0.105298
O      1.000354    1.454221   -0.520665
N      2.541304    0.666133    1.005686
C      3.169006   -0.475819    1.668296
C      4.153071   -1.216933    0.776637
C      3.131011    1.990821    1.173504
C      4.173978    2.316558    0.116636
C     -1.297627   -0.695157   -0.507487
H     -4.503173    1.705958    1.993461
H     -6.142005    1.090851    2.317345
H     -4.709097    0.095809    2.671843
H     -5.819700   -0.666095    0.569092
H     -5.610248    0.974198   -0.121934
H     -2.879480   -1.702509   -3.357151
H     -0.555167   -2.266476   -3.990674
H      1.308209   -1.827905   -2.402885
H      3.668178   -0.130244    2.580025
H      2.382573   -1.170287    1.984903
H      4.999085   -0.581610    0.502090
H      3.678923   -1.559056   -0.148496
H      4.548974   -2.092952    1.299996
H      2.330636    2.738089    1.117769
H      3.563946    2.066039    2.176815
H      3.763997    2.215568   -0.893276
H      5.040410    1.654716    0.195451
H      4.527744    3.344933    0.240508
H     -1.478404   -0.261375    0.471103
