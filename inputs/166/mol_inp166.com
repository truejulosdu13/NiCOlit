%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -5.768529    0.413732   -0.992427
C     -5.196918    0.036605    0.380642
C     -5.403323   -1.458693    0.655974
C     -5.924311    0.848681    1.464412
C     -3.699004    0.362463    0.419557
O     -3.178600    1.150981    1.198878
O     -3.034460   -0.371207   -0.562950
C     -1.650745   -0.174510   -0.626047
C     -1.130006    0.394336   -1.782746
C      0.250833    0.552734   -1.901537
C      1.113962    0.131871   -0.874339
C      2.507672    0.286031   -0.980972
C      3.378718   -0.136618    0.038151
C      4.834863    0.034595   -0.091098
C      5.706668   -1.059180    0.039779
C      7.089307   -0.897320   -0.083762
C      7.623639    0.363018   -0.339176
C      6.777163    1.460802   -0.471472
C      5.394696    1.297047   -0.348885
C      2.819935   -0.724708    1.181418
C      1.436769   -0.885663    1.306972
C      0.571622   -0.462857    0.288648
C     -0.818231   -0.619373    0.400262
H     -5.319642   -0.182199   -1.795029
H     -6.851780    0.253075   -1.028806
H     -5.574387    1.467441   -1.223597
H     -6.468782   -1.710492    0.700226
H     -4.949766   -2.080042   -0.124410
H     -4.946460   -1.750118    1.608762
H     -6.999820    0.638855    1.464451
H     -5.793850    1.925884    1.308364
H     -5.537869    0.614091    2.463100
H     -1.787561    0.714415   -2.584947
H      0.652571    1.006130   -2.805031
H      2.926832    0.734536   -1.880540
H      5.308808   -2.054231    0.228830
H      7.746690   -1.756823    0.017550
H      8.698774    0.489723   -0.434502
H      7.190705    2.446672   -0.667013
H      4.751555    2.169832   -0.443795
H      3.462587   -1.051922    1.996525
H      1.041443   -1.339784    2.212908
H     -1.251767   -1.077811    1.284490
