%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
O     -3.836885   -0.035991    1.649698
C     -3.026277   -0.564202    0.606291
C     -1.592117   -0.055030    0.715252
O     -0.846323   -0.559286   -0.389846
C      0.470186   -0.197241   -0.468485
C      1.161756    0.594866    0.449207
C      2.513801    0.888903    0.244362
C      3.179154    0.393240   -0.875644
C      2.494151   -0.398744   -1.793501
C      1.144894   -0.691770   -1.588095
H     -3.969004    0.909910    1.464808
H     -3.467756   -0.296345   -0.359897
H     -3.045073   -1.657134    0.672462
H     -1.163509   -0.398075    1.665042
H     -1.599928    1.042054    0.707564
H      0.682989    0.997635    1.335168
H      3.048552    1.505402    0.962596
H      4.229964    0.622827   -1.031539
H      3.007295   -0.789587   -2.667965
H      0.614129   -1.311432   -2.306624
