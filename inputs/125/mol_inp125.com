%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -3.965959    1.093789    1.263756
O     -2.665862    0.948737    0.711159
C     -2.599552   -0.158438   -0.190318
C     -1.210397   -0.235723   -0.795135
C     -1.037055    0.021066   -2.174706
O     -2.169208    0.328622   -2.886252
C     -2.007193    0.772172   -4.227698
C      0.233965   -0.066869   -2.743157
C      1.338387   -0.396900   -1.959018
C      1.200644   -0.645807   -0.590211
C      2.331632   -0.970209    0.174149
C      2.202320   -1.198292    1.539102
O      3.340449   -1.538882    2.287380
C      4.101349   -0.566108    2.929684
C      3.661948    0.861273    2.758730
O      5.085594   -0.872762    3.597261
C      0.967561   -1.157678    2.164870
C     -0.159235   -0.836144    1.406996
C     -0.080531   -0.569070    0.016285
H     -3.956182    1.948641    1.945473
H     -4.244071    0.198826    1.828909
H     -4.699762    1.285900    0.475022
H     -3.377598   -0.045963   -0.952084
H     -2.816882   -1.092156    0.343558
H     -1.637659   -0.037261   -4.865532
H     -1.355638    1.650831   -4.281770
H     -2.992358    1.066085   -4.602734
H      0.403500    0.109610   -3.800020
H      2.314896   -0.456451   -2.435300
H      3.312880   -1.043000   -0.287155
H      3.711773    1.157457    1.708805
H      2.662667    1.012058    3.172631
H      4.347516    1.507750    3.316198
H      0.872626   -1.365464    3.226239
H     -1.114563   -0.786739    1.924883

