%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -3.871996   -0.317024   -0.479090
C     -2.436042   -0.336405   -0.047342
C     -1.965253   -1.335294    0.809118
C     -0.633879   -1.329885    1.232528
C      0.244552   -0.324821    0.820767
O      1.514952   -0.449504    1.308873
C      2.467437    0.564278    0.999179
C      3.170354    0.286443   -0.327152
O      4.165673    1.269816   -0.589204
C     -0.225194    0.679958   -0.023545
C     -1.558934    0.676551   -0.449834
H     -4.274698   -1.333401   -0.542642
H     -3.975446    0.135399   -1.470972
H     -4.469976    0.256465    0.235786
H     -2.628567   -2.125477    1.152756
H     -0.279308   -2.116415    1.894239
H      3.219900    0.542315    1.797216
H      2.047179    1.575914    1.048490
H      3.638891   -0.703769   -0.321228
H      2.459811    0.293673   -1.158799
H      4.884712    1.134332    0.052177
H      0.412918    1.483034   -0.376159
H     -1.907085    1.473819   -1.103321
