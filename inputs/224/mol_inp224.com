%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      3.399242    0.602647    0.779782
N      2.801508    0.353345   -0.520216
C      3.693519    0.355273   -1.663803
C      1.523657   -0.175747   -0.669955
O      1.061674   -0.529547   -1.750293
O      0.839891   -0.258245    0.540085
C     -0.458221   -0.779016    0.431656
C     -0.611934   -2.164482    0.451521
C     -1.892323   -2.697086    0.391671
C     -2.997569   -1.841443    0.329172
C     -2.862857   -0.457201    0.328252
N     -3.971225    0.304230    0.280988
C     -3.822318    1.646865    0.284951
C     -2.599993    2.286367    0.340600
C     -1.457347    1.498286    0.392991
C     -1.566220    0.094598    0.385190
H      3.981526   -0.277048    1.069150
H      4.058048    1.473347    0.712456
H      2.645138    0.798192    1.545688
H      4.373861   -0.497244   -1.578144
H      3.145366    0.277585   -2.606517
H      4.269309    1.285308   -1.668353
H      0.253723   -2.817430    0.502771
H     -2.039685   -3.773727    0.392461
H     -3.997885   -2.266698    0.282359
H     -4.748506    2.212773    0.240599
H     -2.537161    3.368196    0.340382
H     -0.483220    1.977901    0.434553

