%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -3.638073   -0.502124   -1.112350
C     -2.715222    0.200091   -0.103214
C     -3.085949   -0.257877    1.313521
C     -2.890837    1.718233   -0.240054
C     -1.258070   -0.175719   -0.398364
O     -0.901621   -0.921812   -1.302233
O     -0.416343    0.458219    0.514057
C      0.934436    0.172039    0.315100
C      1.531124   -0.825438    1.081533
C      2.892213   -1.086710    0.918456
C      3.659168   -0.339793    0.006542
C      5.126143   -0.640936   -0.185821
F      5.342578   -1.582350   -1.144727
F      5.851028    0.450172   -0.561623
F      5.726842   -1.111651    0.943309
C      3.033537    0.668701   -0.747799
C      1.672781    0.933388   -0.587536
H     -4.689705   -0.254451   -0.929390
H     -3.399279   -0.207563   -2.140925
H     -3.535982   -1.591965   -1.052564
H     -2.491436    0.259971    2.074568
H     -2.908255   -1.332187    1.438953
H     -4.141952   -0.061720    1.530174
H     -2.293773    2.262817    0.500041
H     -2.572859    2.063457   -1.230594
H     -3.937344    2.010747   -0.099287
H      0.944606   -1.399605    1.792542
H      3.357481   -1.876372    1.506967
H      3.608946    1.252991   -1.464647
H      1.195814    1.717448   -1.167992
