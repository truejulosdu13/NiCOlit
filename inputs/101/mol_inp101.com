%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      2.745565    0.070823    0.213560
O      1.709638   -0.899567    0.128054
C      0.428876   -0.430600    0.031052
C     -0.551319   -1.423994   -0.050238
C     -1.903208   -1.087183   -0.153453
C     -2.276995    0.254250   -0.175478
I     -4.284362    0.767502   -0.328659
C     -1.308588    1.254272   -0.095068
C      0.044639    0.911127    0.008214
H      2.778904    0.693047   -0.686882
H      2.639013    0.683627    1.114775
H      3.698301   -0.462711    0.284744
H     -0.257835   -2.471041   -0.032912
H     -2.644410   -1.879302   -0.215138
H     -1.588614    2.304545   -0.111331
H      0.770395    1.715204    0.068761

