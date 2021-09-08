%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -4.987573    2.341861    1.036000
C     -4.938444    0.948790    0.403884
C     -5.763669   -0.051202    1.218117
C     -3.505395    0.478659    0.214478
C     -3.002118    0.290856   -1.080440
C     -1.680355   -0.133651   -1.307856
C     -1.180672   -0.316932   -2.737074
C     -1.136729    1.021154   -3.482681
C     -2.017043   -1.348297   -3.504000
C     -0.855734   -0.363290   -0.183440
C      0.545234   -0.775387   -0.392765
O      0.907782   -1.935490   -0.490214
O      1.287862    0.386857   -0.447816
C      2.647803    0.139312   -0.632555
C      3.169691    0.163840   -1.921307
C      4.538387   -0.035448   -2.102658
C      5.387130   -0.245129   -1.000359
C      6.766003   -0.448927   -1.168906
C      7.598003   -0.653005   -0.066134
C      7.063373   -0.655656    1.219093
C      5.694894   -0.454558    1.405931
C      4.843551   -0.248945    0.307928
C      3.466405   -0.046842    0.480943
C     -1.335274   -0.200850    1.136579
C     -0.457179   -0.454585    2.357239
C     -1.098897   -1.435912    3.344606
C     -0.099165    0.861301    3.052067
C     -2.662607    0.232115    1.307620
H     -4.422127    3.063024    0.436276
H     -4.567504    2.347918    2.047528
H     -6.020964    2.698505    1.102392
H     -5.414646    1.024908   -0.583154
H     -5.407365   -0.134655    2.250619
H     -6.813688    0.258498    1.253176
H     -5.725571   -1.049433    0.767392
H     -3.654624    0.485041   -1.930564
H     -0.155923   -0.705475   -2.730582
H     -0.517351    1.748552   -2.943565
H     -0.704792    0.889455   -4.480857
H     -2.136429    1.455930   -3.606497
H     -1.598758   -1.516093   -4.502885
H     -3.058679   -1.030423   -3.631407
H     -2.023835   -2.310827   -2.977763
H      2.522396    0.330426   -2.776980
H      4.941553   -0.026533   -3.112777
H      7.203055   -0.451141   -2.164883
H      8.663439   -0.810788   -0.212125
H      7.710889   -0.815778    2.077517
H      5.295710   -0.462621    2.417669
H      3.028994   -0.039489    1.474661
H      0.485013   -0.919813    2.049153
H     -1.370602   -2.371979    2.844526
H     -0.397887   -1.678186    4.150864
H     -2.002291   -1.026573    3.808756
H      0.394801    1.549115    2.356215
H     -0.985311    1.368704    3.449242
H      0.585686    0.681976    3.888035
H     -3.048451    0.387115    2.313739

