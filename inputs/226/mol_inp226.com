%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -3.011483    1.058552   -1.811316
C     -2.866713    0.886773   -0.307853
N     -2.259763   -0.397039    0.032799
C     -3.140867   -1.531578    0.281999
C     -3.547766   -1.617226    1.744282
C     -0.883187   -0.578642    0.126612
O     -0.366654   -1.645803    0.443899
O     -0.176375    0.580161   -0.186609
C      1.217017    0.486152   -0.098401
C      1.845419    0.353793    1.137593
C      3.239296    0.319970    1.193578
C      4.001214    0.444730    0.022594
C      5.498597    0.370286    0.082406
C      3.347724    0.583065   -1.210658
C      1.954134    0.616470   -1.272102
H     -3.641916    0.271793   -2.238196
H     -2.038232    1.012606   -2.310651
H     -3.468754    2.025727   -2.041260
H     -3.846084    0.955030    0.178428
H     -2.254036    1.700460    0.094643
H     -4.025411   -1.440369   -0.357891
H     -2.630999   -2.458717   -0.002972
H     -4.081950   -0.714943    2.058984
H     -2.671185   -1.728043    2.391015
H     -4.204311   -2.477451    1.906688
H      1.260942    0.264232    2.047901
H      3.728656    0.194668    2.156712
H      5.872021    0.771505    1.030442
H      5.825472   -0.669454   -0.015476
H      5.951420    0.961246   -0.720657
H      3.922628    0.666631   -2.130018
H      1.451145    0.735415   -2.226514
