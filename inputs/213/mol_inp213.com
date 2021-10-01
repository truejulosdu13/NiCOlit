%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -5.197267    0.312542    0.378754
C     -4.103927   -0.727418    0.193511
N     -2.797549   -0.212608    0.586537
C     -2.381076   -0.377013    1.976782
C     -1.622727   -1.677633    2.186632
C     -2.003439    0.410675   -0.371074
O     -2.357780    0.576069   -1.534374
O     -0.771688    0.806936    0.143194
C      0.063564    1.451810   -0.777634
C     -0.132025    2.816517   -1.004059
C      0.721832    3.497424   -1.868267
C      1.771601    2.818387   -2.485664
C      1.970769    1.456768   -2.239916
C      1.118100    0.749719   -1.377209
C      1.362802   -0.717203   -1.114893
C      2.525529   -0.988461   -0.186421
C      2.448126   -0.651225    1.172902
C      3.520773   -0.904387    2.029241
C      4.681410   -1.497727    1.537138
C      4.771011   -1.837263    0.188631
C      3.699341   -1.584183   -0.669596
H     -5.271838    0.624835    1.425374
H     -6.165798   -0.095312    0.073771
H     -4.996962    1.205809   -0.221978
H     -4.073925   -1.032460   -0.858547
H     -4.323815   -1.624198    0.783029
H     -3.269540   -0.347553    2.617125
H     -1.747227    0.465363    2.273329
H     -1.316261   -1.774742    3.232576
H     -2.244629   -2.541544    1.930806
H     -0.725253   -1.718147    1.561793
H     -0.946141    3.343177   -0.515766
H      0.571709    4.556556   -2.058788
H      2.440806    3.350145   -3.157347
H      2.802769    0.948597   -2.722734
H      1.526123   -1.216654   -2.079564
H      0.468952   -1.202719   -0.705154
H      1.551075   -0.182067    1.570278
H      3.451294   -0.634832    3.079616
H      5.516983   -1.693017    2.203806
H      5.676837   -2.298392   -0.196008
H      3.787458   -1.854573   -1.719302
