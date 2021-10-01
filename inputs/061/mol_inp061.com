%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      6.855821   -0.321252   -0.741133
O      6.195365    0.700804   -0.005822
C      4.830137    0.638820    0.053649
C      4.233507    1.664163    0.786736
C      2.845790    1.723004    0.929419
C      2.026767    0.751285    0.338170
C      0.627799    0.794343    0.471263
C     -0.199155   -0.177001   -0.120640
N     -1.634559   -0.120947    0.027031
C     -2.487764   -0.425454   -1.097518
C     -2.168609    0.024585   -2.391095
C     -2.994068   -0.270345   -3.479424
C     -4.152184   -1.021253   -3.293212
C     -4.486676   -1.477208   -2.020419
C     -3.661873   -1.181429   -0.931824
C     -2.216266    0.238771    1.299054
C     -3.343291    1.076989    1.368732
C     -3.904803    1.427056    2.599615
C     -3.349588    0.944374    3.782158
C     -2.233602    0.112474    3.735987
C     -1.671562   -0.237052    2.505206
C      0.411090   -1.202708   -0.858218
C      1.800964   -1.259014   -0.999173
C      2.622880   -0.289351   -0.406786
C      4.023761   -0.333925   -0.540457
H      6.563433   -0.300385   -1.796185
H      7.931621   -0.127542   -0.690368
H      6.673899   -1.306304   -0.298818
H      4.854539    2.425668    1.252158
H      2.410779    2.535036    1.507663
H      0.171588    1.608616    1.031040
H     -1.278901    0.627869   -2.555970
H     -2.732332    0.091082   -4.470409
H     -4.794160   -1.250652   -4.139708
H     -5.387730   -2.066450   -1.871898
H     -3.932646   -1.561888    0.050355
H     -3.781875    1.479831    0.458735
H     -4.773170    2.079787    2.631895
H     -3.786366    1.216544    4.739472
H     -1.799755   -0.269123    4.656526
H     -0.810956   -0.901805    2.492195
H     -0.194430   -1.981641   -1.316069
H      2.234422   -2.073451   -1.575239
H      4.462161   -1.142240   -1.116675

