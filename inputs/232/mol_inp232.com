%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      5.077816   -0.738238    0.521452
O      4.034660   -1.098493   -0.381921
C      2.860849   -0.448963   -0.153154
O      2.673577    0.374479    0.731618
O      1.944160   -0.889965   -1.091899
C      0.691396   -0.289987   -0.957836
C      0.440066    0.922085   -1.592518
C     -0.829872    1.490287   -1.492142
C     -1.851609    0.844338   -0.773729
C     -3.134027    1.405293   -0.665396
C     -4.141412    0.751805    0.046509
C     -3.880763   -0.470611    0.658841
C     -2.611644   -1.043368    0.561958
C     -1.586408   -0.399180   -0.149029
C     -0.306264   -0.963046   -0.254464
H      5.312568    0.327202    0.432710
H      5.970241   -1.311284    0.254838
H      4.800261   -0.990393    1.549815
H      1.219783    1.424495   -2.156436
H     -1.020184    2.443011   -1.981204
H     -3.358365    2.359598   -1.136638
H     -5.129294    1.198632    0.122449
H     -4.664783   -0.979378    1.213623
H     -2.427667   -1.998491    1.048797
H     -0.083086   -1.919830    0.208961
