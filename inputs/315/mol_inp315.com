%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      4.519210    1.334770   -2.432203
O      3.142946    0.979919   -2.485613
C      2.509418    0.765052   -1.292859
C      3.080720    0.873162   -0.033596
C      2.280721    0.611042    1.064171
N      0.978716    0.260250    0.987470
C      0.418907    0.155726   -0.249751
C     -1.011398   -0.238099   -0.304251
C     -1.754167   -0.493354    0.852247
C     -3.091485   -0.863557    0.740162
O     -3.939482   -1.144889    1.776435
C     -3.373766   -1.152264    3.082601
C     -3.659981   -0.964456   -0.522848
C     -2.868208   -0.699823   -1.622118
N     -1.569385   -0.342154   -1.542033
C      1.167440    0.401977   -1.400451
H      4.876389    1.438396   -3.461238
H      4.658792    2.296982   -1.928394
H      5.113994    0.547767   -1.955327
H      4.114587    1.150125    0.123075
H      2.680152    0.681045    2.072459
H     -1.269152   -0.394712    1.816353
H     -4.163111   -1.427526    3.788777
H     -2.578697   -1.901144    3.163336
H     -3.009498   -0.156641    3.357656
H     -4.698662   -1.249897   -0.645191
H     -3.268639   -0.774712   -2.629272
H      0.713638    0.307013   -2.382869

