%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      4.701681   -0.070525   -0.018271
O      3.416690   -0.658252    0.134207
C      2.318667    0.148122   -0.053456
C      2.393098    1.504040   -0.383437
C      1.229162    2.254951   -0.557113
C     -0.031622    1.667943   -0.405015
C     -1.202006    2.418612   -0.578559
C     -2.458044    1.828981   -0.425846
C     -2.575062    0.473582   -0.095685
C     -3.831482   -0.130347    0.060516
C     -3.934308   -1.480919    0.389422
C     -2.782218   -2.245109    0.566402
C     -1.512257   -1.668428    0.416719
C     -0.348023   -2.422506    0.591176
C      0.909849   -1.833893    0.438700
C      1.053813   -0.471841    0.106679
C     -0.123477    0.297231   -0.071317
C     -1.403058   -0.299223    0.083231
H      4.849564    0.296850   -1.039062
H      4.862799    0.723755    0.717940
H      5.448509   -0.849021    0.165260
H      3.342598    2.011759   -0.513927
H      1.318517    3.308449   -0.813723
H     -1.147287    3.474262   -0.835409
H     -3.346239    2.440565   -0.567712
H     -4.742130    0.449159   -0.073405
H     -4.912994   -1.938425    0.507961
H     -2.882406   -3.297405    0.822826
H     -0.404779   -3.477726    0.848001
H      1.792445   -2.454639    0.582898

