%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -2.592081    1.154022   -4.017975
O     -2.549059    0.970409   -2.602779
C     -1.493906    0.226940   -2.159507
O     -0.648452   -0.250719   -2.909208
C     -1.437238    0.171670   -0.615870
C     -0.501962   -0.955518   -0.122525
C      0.971002   -0.600815   -0.166599
C      1.517210    0.265803    0.795032
C      2.872566    0.594593    0.763612
C      3.682342    0.042225   -0.224320
O      5.027377    0.406107   -0.287358
C      5.865977   -0.440087    0.432548
O      5.512838   -1.407270    1.094209
C      7.318906    0.029237    0.288568
C      8.247822   -0.958881    1.012177
C      7.707018    0.075276   -1.195701
C      7.467027    1.416990    0.925326
C      3.161069   -0.803669   -1.198882
C      1.805200   -1.130182   -1.162776
N     -2.780827    0.147624    0.062236
C     -3.334068    1.392711    0.393757
O     -2.799429    2.437490    0.001226
C     -4.528303    1.474448    1.399046
C     -4.747965    2.969904    1.747016
C     -4.158285    0.752856    2.705992
C     -5.863219    0.950276    0.861192
C     -3.583232   -0.998229   -0.211032
C     -4.531962   -0.984400   -1.240313
C     -5.274706   -2.129884   -1.535116
C     -5.075413   -3.297887   -0.802596
C     -4.132551   -3.321214    0.222597
C     -3.386116   -2.177203    0.513979
H     -3.467983    1.765367   -4.253579
H     -2.696436    0.190093   -4.526660
H     -1.698084    1.681877   -4.363164
H     -0.963065    1.121379   -0.334852
H     -0.676548   -1.881361   -0.684885
H     -0.720161   -1.191203    0.926892
H      0.888775    0.689300    1.575779
H      3.289123    1.268988    1.504966
H      8.004407   -1.025855    2.078633
H      8.156603   -1.967748    0.593591
H      9.297032   -0.654863    0.927809
H      8.764350    0.334036   -1.321719
H      7.119344    0.819102   -1.745301
H      7.535972   -0.894704   -1.676861
H      8.506539    1.761547    0.889359
H      7.151135    1.404569    1.974690
H      6.852612    2.164542    0.411141
H      3.798263   -1.207254   -1.979340
H      1.399911   -1.791137   -1.927019
H     -5.543228    3.098952    2.489711
H     -3.839060    3.422317    2.160561
H     -5.029394    3.544319    0.856454
H     -4.897844    0.950357    3.490625
H     -4.109713   -0.331816    2.577381
H     -3.179417    1.080508    3.075261
H     -6.704481    1.313649    1.464294
H     -6.033869    1.279767   -0.168985
H     -5.925436   -0.139752    0.898353
H     -4.703823   -0.085101   -1.825923
H     -6.008371   -2.107818   -2.337138
H     -5.655536   -4.189171   -1.029193
H     -3.977550   -4.229816    0.799075
H     -2.671644   -2.215694    1.330950
