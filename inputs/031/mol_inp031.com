%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      4.541744   -0.814748    0.280580
O      3.373662   -1.571784   -0.010348
C      2.181170   -0.902381    0.019074
C      2.005709    0.446554    0.306440
C      0.720745    1.001913    0.300579
C     -0.422853    0.228079    0.008333
C     -1.737228    0.764414   -0.004843
O     -1.845845    2.104058    0.287262
C     -3.146997    2.675783    0.283454
C     -2.826020   -0.062311   -0.303446
C     -2.628272   -1.412246   -0.588756
C     -1.342817   -1.948144   -0.578444
C     -0.233709   -1.143348   -0.282723
C      1.062699   -1.686782   -0.272993
H      4.688561   -0.016818   -0.454848
H      5.401402   -1.488747    0.216025
H      4.509016   -0.413808    1.299048
H      2.839847    1.099122    0.539557
H      0.613004    2.060863    0.529241
H     -3.783676    2.214343    1.045405
H     -3.604006    2.611552   -0.709458
H     -3.046104    3.736272    0.533822
H     -3.843422    0.312845   -0.322696
H     -3.478378   -2.049095   -0.819943
H     -1.211562   -3.004433   -0.804096
H      1.213328   -2.741152   -0.496227
