%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -5.415775   -0.040271    0.231888
O     -4.340017    0.060256   -0.692471
C     -3.074161    0.009787   -0.176286
C     -2.742611   -0.130786    1.169135
C     -1.399570   -0.167911    1.560767
C     -0.367355   -0.065222    0.614811
C      0.989032   -0.100780    0.991988
C      2.039790    0.000545    0.059598
C      3.480236   -0.045432    0.538260
C      4.557766   -0.710922   -0.283371
C      4.553228    0.782944   -0.126201
C      1.671747    0.141248   -1.288087
C      0.330843    0.179214   -1.687716
C     -0.699628    0.076928   -0.745605
C     -2.052251    0.112680   -1.124507
H     -6.350004    0.016548   -0.335063
H     -5.405889    0.793660    0.941495
H     -5.400434   -1.003560    0.752389
H     -3.500184   -0.214290    1.940765
H     -1.164654   -0.278372    2.617227
H      1.239821   -0.210941    2.046058
H      3.598713   -0.158503    1.616446
H      5.317349   -1.275463    0.246341
H      4.307275   -1.126754   -1.251841
H      4.299631    1.389712   -0.987084
H      5.309739    1.229436    0.509882
H      2.438287    0.224289   -2.054953
H      0.100930    0.289786   -2.745080
H     -2.321853    0.222173   -2.172795

