%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -6.233410   -0.252242    0.037131
C     -4.750134   -0.056962   -0.087011
C     -4.051846   -0.616504   -1.164221
C     -2.673136   -0.408668   -1.319972
C     -1.965574   -1.028525   -2.493202
C     -1.992095    0.361371   -0.355018
C     -0.535700    0.585855   -0.481064
O     -0.031793    1.668785   -0.724294
O      0.082376   -0.624214   -0.230206
C      1.481201   -0.596678   -0.273476
C      2.110233   -1.418638   -1.202450
C      3.503993   -1.464426   -1.240642
C      4.272118   -0.702182   -0.343403
C      5.675439   -0.738244   -0.370378
C      6.426513    0.021291    0.528335
C      5.785326    0.826342    1.465315
C      4.391028    0.874260    1.506801
C      3.620059    0.116979    0.610373
C      2.217794    0.154962    0.641893
C     -2.680788    0.948864    0.725071
C     -1.978012    1.809408    1.738773
C     -4.062831    0.738101    0.838333
H     -6.761196    0.540257   -0.502209
H     -6.535416   -1.221852   -0.372581
H     -6.543352   -0.233610    1.087277
H     -4.592837   -1.215531   -1.895553
H     -1.275217   -0.315731   -2.955570
H     -2.676178   -1.330377   -3.270298
H     -1.413325   -1.917845   -2.174555
H      1.525300   -2.018525   -1.892463
H      3.990005   -2.102668   -1.974974
H      6.195703   -1.360134   -1.095299
H      7.512215   -0.015344    0.494821
H      6.369370    1.419762    2.163847
H      3.908180    1.512153    2.243781
H      1.700832    0.773141    1.370298
H     -2.632474    2.045063    2.584873
H     -1.666838    2.754340    1.282660
H     -1.102760    1.291912    2.144735
H     -4.612773    1.196055    1.659336

