%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -2.385768   -0.611573   -1.911167
C     -2.513961   -0.095859   -0.471792
C     -3.093391   -1.185781    0.439565
C     -3.455918    1.118862   -0.458586
C     -1.133456    0.324931    0.045798
O     -0.840240    1.457827    0.407539
O     -0.274362   -0.773972    0.040127
C      1.024287   -0.510830    0.486612
C      1.444405   -1.123959    1.663892
C      2.746889   -0.917098    2.112792
C      3.621207   -0.114065    1.375066
C      3.206376    0.483059    0.171522
C      4.151263    1.360097   -0.610511
F      4.186636    2.637414   -0.142169
F      5.438772    0.913757   -0.574491
F      3.831258    1.452718   -1.931910
C      1.891850    0.275633   -0.271191
H     -1.918709    0.139790   -2.558291
H     -1.766825   -1.514315   -1.963678
H     -3.366408   -0.859063   -2.332656
H     -2.484345   -2.096513    0.418686
H     -3.134410   -0.847012    1.481195
H     -4.108625   -1.460957    0.132734
H     -3.072149    1.926165   -1.093200
H     -4.454140    0.852531   -0.823776
H     -3.567397    1.524156    0.553862
H      0.763861   -1.751882    2.231153
H      3.083046   -1.379773    3.037663
H      4.635424    0.043600    1.740788
H      1.544829    0.732113   -1.195577

