%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
F      3.114719   -2.191697   -0.924094
C      2.113191   -1.660560   -0.187396
F      2.662917   -1.040965    0.879128
F      1.324262   -2.671841    0.235063
O      1.390704   -0.748498   -0.995239
C      0.329771   -0.100810   -0.430176
C     -0.332379    0.782250   -1.289943
C     -1.436233    1.514509   -0.849818
C     -1.887679    1.368320    0.458679
C     -1.234589    0.491205    1.322241
C     -0.128380   -0.242483    0.880832
H      0.013398    0.902937   -2.314203
H     -1.941285    2.197021   -1.528309
H     -2.746884    1.936482    0.805834
H     -1.586684    0.376408    2.344754
H      0.345151   -0.912276    1.592648

