%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -6.350570    0.532095   -0.044109
O     -5.585959   -0.664670   -0.113613
C     -4.225148   -0.530043   -0.147863
C     -3.523978   -1.733405   -0.215911
C     -2.128092   -1.743367   -0.256900
C     -1.404093   -0.542847   -0.230322
C      0.003241   -0.534914   -0.270883
C      0.712313    0.666959   -0.243396
O      2.072583    0.802772   -0.277941
C      2.849550   -0.394834   -0.319641
C      4.338985   -0.019056   -0.333071
C      4.737446    0.727348    0.942192
C      5.202231   -1.266654   -0.509154
C      0.011239    1.870369   -0.176480
C     -1.384591    1.880540   -0.135524
C     -2.108619    0.680060   -0.161553
C     -3.515983    0.672064   -0.120942
H     -6.187983    1.156444   -0.928877
H     -6.135562    1.085093    0.876231
H     -7.408276    0.252246   -0.024457
H     -4.069463   -2.673758   -0.237309
H     -1.613000   -2.699644   -0.309772
H      0.523283   -1.485774   -0.325097
H      2.611813   -0.953245   -1.233537
H      2.633755   -1.016004    0.558853
H      4.520312    0.656401   -1.178948
H      4.557182    0.115318    1.832514
H      5.800792    0.989027    0.921540
H      4.170700    1.657830    1.050032
H      4.951834   -1.784397   -1.441072
H      6.264288   -1.002611   -0.549324
H      5.058871   -1.969701    0.318465
H      0.556941    2.810619   -0.155525
H     -1.899585    2.836886   -0.083154
H     -4.036457    1.622852   -0.068274
