%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      3.477014   -0.086676   -0.007358
C      2.626529    1.171367   -0.036185
O      1.591212    1.106226    0.942605
C      0.389717    0.554716    0.597234
C      0.083874   -0.082372   -0.604028
C     -1.196330   -0.607578   -0.818522
C     -2.182312   -0.514365    0.169763
C     -3.561715   -1.048182   -0.076879
C     -1.875519    0.133297    1.369548
C     -0.597130    0.655887    1.580690
H      2.891153   -0.972812   -0.268695
H      4.313320   -0.003650   -0.707353
H      3.875806   -0.256093    0.998388
H      2.245771    1.397151   -1.038279
H      3.260148    2.021693    0.240994
H      0.815061   -0.196162   -1.396234
H     -1.416218   -1.094855   -1.766010
H     -4.185484   -0.275228   -0.536193
H     -3.530662   -1.918906   -0.740272
H     -4.029280   -1.370419    0.859485
H     -2.626025    0.235402    2.149703
H     -0.368928    1.151559    2.521128

