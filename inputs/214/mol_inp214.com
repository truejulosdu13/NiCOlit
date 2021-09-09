%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -3.977126    0.051298   -1.333906
C     -2.649324    0.792377   -1.323327
N     -2.192401    1.131155    0.022892
C     -2.510048    2.447607    0.566683
C     -3.829501    2.476656    1.320918
C     -1.581041    0.197938    0.858951
O     -1.262021    0.443354    2.018618
O     -1.390467   -1.028021    0.227980
C     -0.694042   -1.960751    1.007500
C     -1.437454   -2.820832    1.819267
C     -0.776649   -3.784151    2.577358
C      0.612605   -3.892376    2.506549
C      1.347089   -3.037742    1.676490
C      0.703553   -2.052874    0.915024
C      1.455817   -1.179712    0.001256
C      2.601015   -0.565184    0.337686
C      3.335873    0.324651   -0.570492
C      4.734161    0.257142   -0.605351
C      5.465731    1.107401   -1.438283
C      4.806117    2.041203   -2.234930
C      3.416000    2.129300   -2.195635
C      2.684155    1.278464   -1.363531
H     -4.244774   -0.226723   -2.358163
H     -3.932342   -0.862972   -0.734041
H     -4.783384    0.672225   -0.934856
H     -2.721521    1.709507   -1.917881
H     -1.887778    0.166871   -1.802839
H     -1.707657    2.745067    1.252090
H     -2.516842    3.179831   -0.247888
H     -4.673001    2.278219    0.654457
H     -3.984830    3.462140    1.771163
H     -3.848890    1.728911    2.120122
H     -2.518405   -2.733397    1.866496
H     -1.342507   -4.451703    3.221788
H      1.127788   -4.648238    3.094250
H      2.426394   -3.158743    1.622922
H      1.010763   -1.038288   -0.980349
H      3.041821   -0.702080    1.322116
H      5.265350   -0.464638    0.010741
H      6.550228    1.041118   -1.462479
H      5.375514    2.704126   -2.881016
H      2.900211    2.864787   -2.807094
H      1.601824    1.377076   -1.330034
