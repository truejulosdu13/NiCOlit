%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -4.245413    0.741615    0.913586
O     -3.955909   -0.569473    0.449153
C     -2.728284   -0.614030   -0.282689
O     -1.613451   -0.319726    0.552677
C     -0.379701   -0.307678   -0.036840
C     -0.097802   -0.633408   -1.362730
C      1.217921   -0.576594   -1.830157
C      2.257908   -0.196561   -0.977921
C      1.986366    0.127359    0.350045
O      2.903918    0.512738    1.288283
C      4.262675    0.571844    0.873908
C      0.667608    0.068004    0.810003
H     -5.213042    0.718313    1.422307
H     -4.311080    1.442607    0.075802
H     -3.488934    1.077588    1.628957
H     -2.791588    0.088783   -1.123015
H     -2.639040   -1.635441   -0.668703
H     -0.870717   -0.939890   -2.059427
H      1.435073   -0.832080   -2.864695
H      3.264343   -0.165681   -1.381296
H      4.861363    0.885329    1.734570
H      4.401875    1.316065    0.082832
H      4.622882   -0.413086    0.558834
H      0.453028    0.318526    1.846516
