%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      1.082061    3.131079    0.622269
O      2.411867    2.640938    0.502815
C      2.471160    1.302629    0.175171
N      3.698012    0.817418    0.063303
C      3.748396   -0.468423   -0.252180
O      5.026037   -0.974111   -0.370699
C      5.058477   -2.352635   -0.715638
N      2.704029   -1.264254   -0.451002
C      1.529841   -0.661639   -0.305316
O      0.429682   -1.486149   -0.522459
C     -0.801414   -0.935540   -0.266278
C     -1.299753   -0.913898    1.037082
C     -2.570741   -0.386366    1.288374
C     -3.129463   -0.383031    2.680419
C     -3.341233    0.080896    0.212997
C     -2.845035    0.052344   -1.099083
C     -3.695255    0.521642   -2.242292
C     -1.572065   -0.477754   -1.334626
N      1.348149    0.619224    0.001994
H      0.539913    2.620438    1.426003
H      1.140104    4.192095    0.883996
H      0.541903    3.052954   -0.327845
H      6.108071   -2.654233   -0.786792
H      4.595675   -2.529563   -1.692837
H      4.588486   -2.968058    0.059445
H     -0.696770   -1.307450    1.851055
H     -3.814401    0.459240    2.823991
H     -2.330127   -0.280856    3.421949
H     -3.670062   -1.315612    2.868884
H     -4.342085    0.467546    0.399002
H     -4.303064   -0.305835   -2.620881
H     -4.358659    1.333985   -1.927377
H     -3.074863    0.907514   -3.058044
H     -1.176874   -0.534534   -2.345405

