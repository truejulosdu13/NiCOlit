%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -2.077515    1.418190    1.530587
O     -1.087375    1.613449    0.528226
C     -0.347104    0.522325    0.166334
C     -0.473095   -0.765598    0.689540
C      0.351623   -1.796147    0.228399
C      1.305639   -1.543723   -0.757541
C      1.437225   -0.256719   -1.285775
C      2.415349    0.009207   -2.298858
N      3.209481    0.218140   -3.119187
C      0.610205    0.771156   -0.821610
H     -2.573845    2.378294    1.701624
H     -2.838573    0.703945    1.199422
H     -1.624456    1.104054    2.476673
H     -1.202905   -1.002031    1.457346
H      0.248571   -2.797693    0.640092
H      1.941904   -2.354065   -1.108523
H      0.704871    1.777215   -1.226749

