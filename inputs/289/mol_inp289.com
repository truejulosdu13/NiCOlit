%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      5.569215    0.316612    0.009372
O      4.578615    0.331096   -1.010553
C      3.294205    0.059065   -0.629740
C      2.862303   -0.218559    0.664925
C      1.509558   -0.479709    0.914624
C      0.560374   -0.473536   -0.124447
C     -0.906643   -0.741098    0.216951
C     -1.649026    0.425665    0.866047
C     -1.337625    1.767850    0.587731
C     -2.053976    2.805968    1.187541
C     -3.095005    2.521108    2.067445
C     -3.424389    1.197902    2.348250
C     -2.709618    0.157805    1.751563
N     -1.658784   -1.322194   -0.907429
C     -1.660403   -2.650726   -1.257579
C     -2.483118   -2.732393   -2.355412
N     -2.984411   -1.499922   -2.678667
C     -2.478689   -0.673257   -1.790276
C      1.008844   -0.182959   -1.421924
C      2.361119    0.075846   -1.666952
H      6.532575    0.544471   -0.456940
H      5.647999   -0.673935    0.469507
H      5.372565    1.087976    0.761290
H      3.548553   -0.236524    1.505030
H      1.198886   -0.682542    1.938713
H     -0.890625   -1.551881    0.962651
H     -0.534869    2.026443   -0.100172
H     -1.800001    3.839378    0.964066
H     -3.653177    3.331027    2.530093
H     -4.242908    0.975085    3.028292
H     -2.994358   -0.868631    1.977300
H     -1.087971   -3.395798   -0.723131
H     -2.739258   -3.612315   -2.931190
H     -2.672471    0.391192   -1.749903
H      0.322936   -0.152447   -2.266144
H      2.689579    0.293936   -2.680932

