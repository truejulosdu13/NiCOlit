%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -5.136752    0.730357    1.334781
O     -3.723921    0.752978    1.494089
C     -2.985283    0.084816    0.556105
C     -3.493558   -0.648031   -0.520182
C     -2.625640   -1.297442   -1.406512
C     -1.246607   -1.215099   -1.225820
C     -0.747635   -0.465646   -0.163859
O      0.627015   -0.395481    0.091159
C      1.402206    0.214755   -0.895103
O      0.983846    0.687496   -1.944741
C      2.875162    0.209004   -0.468463
C      3.032582    0.989138    0.843405
C      3.724421    0.885741   -1.556558
C      3.352085   -1.238739   -0.292794
C     -1.603044    0.167327    0.735800
H     -5.435641    1.194641    0.389131
H     -5.526327   -0.289759    1.415964
H     -5.574086    1.318875    2.147085
H     -4.560202   -0.737701   -0.697328
H     -3.028833   -1.867699   -2.239624
H     -0.575786   -1.721881   -1.912653
H      2.489342    0.510895    1.666194
H      4.085182    1.058610    1.139677
H      2.640441    2.007938    0.744886
H      3.632780    0.361940   -2.515193
H      3.408540    1.922774   -1.719410
H      4.785404    0.898612   -1.282883
H      3.189693   -1.820309   -1.207594
H      4.420207   -1.277995   -0.051670
H      2.812872   -1.747074    0.514443
H     -1.198462    0.726958    1.573719
