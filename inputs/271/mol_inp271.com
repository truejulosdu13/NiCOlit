%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -2.807073    2.915093   -1.422057
C     -3.701876    1.710758   -1.176915
N     -3.056443    0.727807   -0.313787
C     -3.242713    0.851302    1.129812
C     -4.437388    0.045295    1.613255
C     -2.308619   -0.289968   -0.895978
O     -2.131787   -0.388721   -2.105714
O     -1.784745   -1.164270    0.051919
C     -1.025066   -2.210173   -0.493079
C     -1.691543   -3.347673   -0.953241
C     -0.948006   -4.412004   -1.454329
C      0.445204   -4.335169   -1.477592
C      1.104803   -3.196742   -0.996093
C      0.375793   -2.119880   -0.490151
N      0.963403   -0.950885    0.025985
C      2.284790   -0.633354    0.137288
O      3.245581   -1.314418   -0.179854
O      2.361834    0.608037    0.683189
C      3.646737    1.238374    0.924266
C      4.474232    0.447513    1.943500
C      4.411437    1.466216   -0.384358
C      3.299153    2.606222    1.532159
H     -3.309501    3.635956   -2.074214
H     -2.557632    3.419195   -0.482802
H     -1.867912    2.616983   -1.899651
H     -3.949242    1.246259   -2.138049
H     -4.646155    2.021305   -0.716396
H     -2.339331    0.509016    1.645856
H     -3.371176    1.910010    1.380612
H     -5.360726    0.383163    1.131653
H     -4.314483   -1.018915    1.387373
H     -4.556530    0.153894    2.695567
H     -2.775463   -3.399571   -0.930353
H     -1.450138   -5.300215   -1.828737
H      1.027342   -5.165152   -1.871611
H      2.189857   -3.184843   -1.031096
H      0.337949   -0.230810    0.358166
H      3.878705    0.219751    2.834718
H      5.364538    1.005758    2.252661
H      4.814111   -0.510983    1.540066
H      4.749791    0.526142   -0.830226
H      5.297928    2.089398   -0.224254
H      3.771886    1.954418   -1.128298
H      2.673997    3.189921    0.846274
H      4.196871    3.189355    1.762945
H      2.717605    2.486605    2.453809
