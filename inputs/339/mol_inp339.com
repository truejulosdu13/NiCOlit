%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      5.461070   -0.748421    0.324446
O      4.592122    0.230033    0.880584
C      3.264355    0.123247    0.572270
C      2.451279    1.104104    1.140988
C      1.072130    1.112276    0.908189
C      0.472432    0.137485    0.097624
C     -0.978439    0.141449   -0.150497
C     -1.489738    0.129927   -1.457726
C     -2.868238    0.133901   -1.695139
C     -3.771373    0.137645   -0.628131
C     -5.248995    0.175760   -0.881222
C     -3.274097    0.163736    0.678096
C     -1.895070    0.161023    0.912382
C      1.302241   -0.843132   -0.469099
C      2.684031   -0.852778   -0.235161
H      5.205366   -1.751831    0.680658
H      5.457466   -0.703657   -0.769670
H      6.477190   -0.524223    0.663103
H      2.894013    1.872348    1.770142
H      0.468852    1.898064    1.358006
H     -0.811529    0.133605   -2.308691
H     -3.229346    0.136248   -2.721078
H     -5.499387   -0.343685   -1.812200
H     -5.796517   -0.322745   -0.074410
H     -5.589899    1.213207   -0.952031
H     -3.955481    0.183185    1.525637
H     -1.538242    0.164978    1.940455
H      0.871859   -1.623517   -1.094112
H      3.271946   -1.638231   -0.697479

