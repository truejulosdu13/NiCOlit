%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -4.406232   -0.683486   -1.413336
C     -4.154490   -0.104384   -0.014897
C     -4.493567    1.391740    0.015775
C     -5.047421   -0.839533    0.997726
C     -2.682260   -0.304254    0.365055
O     -2.294965   -0.949760    1.331260
O     -1.868030    0.360717   -0.551590
C     -0.493021    0.246156   -0.325459
C      0.158785   -0.973413   -0.483420
C      1.542314   -1.036288   -0.300162
C      2.285525    0.113819    0.023124
C      3.743041    0.041639    0.217360
C      4.612890    0.871901   -0.509094
C      5.996605    0.802790   -0.325015
C      6.533977   -0.098115    0.590818
C      5.689523   -0.929707    1.321843
C      4.306188   -0.860165    1.135684
C      1.598391    1.334572    0.153578
C      0.214767    1.405454   -0.025605
H     -3.832915   -0.151339   -2.180741
H     -5.465197   -0.615063   -1.686244
H     -4.111427   -1.738108   -1.461445
H     -5.556652    1.561747   -0.187984
H     -3.920797    1.951737   -0.731983
H     -4.262286    1.827687    0.994496
H     -6.108922   -0.717304    0.754777
H     -4.829495   -1.913786    1.013123
H     -4.891265   -0.459899    2.014207
H     -0.396005   -1.870914   -0.738470
H      2.045100   -1.993742   -0.422182
H      4.214501    1.575073   -1.237670
H      6.652815    1.451359   -0.899367
H      7.609773   -0.152008    0.735005
H      6.104673   -1.631774    2.040145
H      3.663120   -1.510655    1.725548
H      2.142944    2.241471    0.409112
H     -0.299986    2.355836    0.074958
