%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      5.814257   -0.094836   -1.316313
O      5.076483    0.278528   -0.159483
C      3.742638   -0.025540   -0.151284
C      3.041306   -0.662739   -1.172338
C      1.671835   -0.913957   -1.033293
C      0.982322   -0.531257    0.127180
C     -0.391815   -0.777510    0.277702
C     -1.059263   -0.372508    1.436937
O     -2.397286   -0.618433    1.647092
C     -3.266370   -0.299155    0.628268
C     -4.258105   -1.233380    0.327667
C     -5.191640   -0.959479   -0.672370
C     -5.141398    0.253914   -1.359421
C     -4.163356    1.197511   -1.042567
C     -3.228560    0.926158   -0.042402
C     -0.359517    0.242297    2.471664
C      1.006851    0.491498    2.334152
C      1.689941    0.113756    1.166354
C      3.064180    0.357678    1.009714
H      6.852958    0.214132   -1.164389
H      5.807087   -1.180922   -1.455768
H      5.440307    0.421420   -2.206654
H      3.525589   -0.978414   -2.090118
H      1.145095   -1.412785   -1.844264
H     -0.942263   -1.289361   -0.507621
H     -4.300174   -2.174306    0.869149
H     -5.960238   -1.689425   -0.912376
H     -5.870493    0.468200   -2.136832
H     -4.132794    2.147494   -1.569806
H     -2.482046    1.673730    0.210717
H     -0.876857    0.532363    3.381938
H      1.537559    0.980526    3.148085
H      3.623764    0.852782    1.800681

