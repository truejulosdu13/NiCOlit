%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      3.386486    2.536379   -2.536908
O      2.270606    1.948290   -1.880935
C      2.624405    0.986744   -0.958182
N      3.905837    0.728801   -0.731225
C      4.117454   -0.210723    0.179477
O      5.449113   -0.479603    0.415509
C      5.655663   -1.506035    1.377515
N      3.181871   -0.872663    0.848107
C      1.940114   -0.521170    0.533104
O      0.955749   -1.221181    1.225143
C     -0.342728   -0.851112    0.985182
C     -0.892222    0.255756    1.633058
C     -2.231680    0.598919    1.413791
C     -3.036704   -0.168806    0.560809
C     -4.447192    0.136444    0.283638
C     -5.101214    1.328320    0.917610
C     -5.122509   -0.672843   -0.545607
C     -4.528823   -1.852768   -1.192957
O     -5.196938   -2.558425   -1.939887
O     -3.199860   -2.116771   -0.914266
C     -2.458794   -1.285321   -0.051946
C     -1.126537   -1.642806    0.153520
N      1.597697    0.400491   -0.361237
H      4.029814    3.068439   -1.827504
H      3.955901    1.789557   -3.101271
H      3.003811    3.271188   -3.252034
H      5.228071   -2.457557    1.042505
H      5.257720   -1.219080    2.357113
H      6.735233   -1.648085    1.487180
H     -0.281879    0.857839    2.301031
H     -2.624672    1.474735    1.923263
H     -6.149760    1.426471    0.614661
H     -5.086861    1.245105    2.009522
H     -4.592314    2.252877    0.624850
H     -6.164510   -0.502336   -0.792048
H     -0.710347   -2.519074   -0.333086
