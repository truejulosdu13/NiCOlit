%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -3.144271    0.450261    0.233713
C     -1.661167    0.228060    0.254881
C     -0.949360    0.300308    1.455129
C      0.436352    0.123607    1.464041
C      1.135565   -0.116999    0.277829
O      2.482783   -0.269776    0.437806
C      3.260104   -0.520494   -0.719447
F      3.171784    0.500202   -1.599237
F      4.555384   -0.644330   -0.352110
F      2.885604   -1.669563   -1.321920
C      0.429272   -0.174759   -0.923465
C     -0.959479    0.002001   -0.934003
H     -3.612029   -0.113986   -0.579948
H     -3.361309    1.514129    0.097358
H     -3.603391    0.112128    1.168649
H     -1.466770    0.493447    2.392157
H      0.973604    0.177509    2.408333
H      0.918183   -0.352260   -1.876875
H     -1.490859   -0.039485   -1.882889
