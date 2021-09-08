%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -5.577292    1.479502   -0.192964
C     -5.086665    0.360076   -1.096882
N     -4.051779   -0.462361   -0.477975
C     -4.450949   -1.636021    0.296740
C     -4.582672   -1.352581    1.785022
C     -2.732215   -0.017201   -0.528124
O     -2.390397    1.010491   -1.105657
O     -1.861072   -0.868355    0.148084
C     -0.525564   -0.454369    0.161859
C     -0.123906    0.588051    0.990690
C      1.223412    0.947334    1.026115
C      2.171892    0.258938    0.249162
C      3.531528    0.613238    0.277417
C      4.470957   -0.081394   -0.497744
C      5.919445    0.261948   -0.495549
O      6.754695   -0.335935   -1.155889
O      6.164392    1.310637    0.329442
C      7.537723    1.699533    0.380328
C      4.046366   -1.140104   -1.308018
C      2.697919   -1.502559   -1.346367
C      1.750077   -0.813285   -0.575527
C      0.393305   -1.166830   -0.605689
H     -4.749902    2.111751    0.144536
H     -6.080903    1.083713    0.692861
H     -6.292948    2.110697   -0.729129
H     -4.678283    0.803057   -2.012904
H     -5.920496   -0.281745   -1.401042
H     -3.702146   -2.422414    0.148834
H     -5.395198   -2.023047   -0.101323
H     -5.377922   -0.629696    1.984458
H     -3.654101   -0.950198    2.201647
H     -4.825841   -2.274000    2.323451
H     -0.848813    1.123839    1.595713
H      1.528652    1.772325    1.666282
H      3.852627    1.437811    0.910179
H      8.154457    0.879922    0.762271
H      7.624464    2.546589    1.066416
H      7.883180    2.017845   -0.608284
H      4.763808   -1.688504   -1.916078
H      2.393830   -2.328003   -1.986898
H      0.046335   -1.988696   -1.225662

