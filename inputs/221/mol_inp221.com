%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      5.061691   -0.343763   -0.371510
C      4.261250    0.215209    0.775608
O      4.813200    0.765906    1.726306
C      2.773360    0.087282    0.749941
C      2.103759   -0.634067   -0.248890
C      0.709256   -0.726968   -0.231879
C     -0.009106   -0.082762    0.774301
O     -1.403081   -0.192996    0.851522
C     -2.126265    0.363064   -0.202722
O     -1.616880    0.934651   -1.162069
N     -3.498223    0.194107   -0.052261
C     -4.081989   -0.612258    1.005471
C     -4.377725    0.592419   -1.134455
C      0.640302    0.618381    1.786278
C      2.033283    0.707050    1.769438
H      4.983698   -1.433761   -0.389762
H      6.114850   -0.077130   -0.237872
H      4.725500    0.088331   -1.317654
H      2.640998   -1.134567   -1.048767
H      0.191610   -1.287841   -1.005055
H     -4.284462   -1.611384    0.608689
H     -3.419430   -0.700355    1.869416
H     -5.018612   -0.151509    1.332921
H     -4.686854   -0.303291   -1.681372
H     -3.885354    1.278564   -1.828576
H     -5.258442    1.088268   -0.716058
H      0.072204    1.096225    2.578890
H      2.541464    1.263196    2.555457

