%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      3.787675    1.188349   -0.540483
O      3.073188   -0.024083   -0.743296
C      1.794916   -0.076580   -0.260175
C      1.140976   -1.292273   -0.470132
C     -0.170104   -1.483531   -0.028470
C     -0.849181   -0.454790    0.633622
C     -2.275474   -0.648432    1.093939
C     -3.288549   -0.019425    0.137686
O     -3.351856   -0.737601   -1.090783
C     -0.191803    0.764319    0.841869
C      1.122445    0.952269    0.398554
H      4.784705    1.065384   -0.974371
H      3.301506    2.023702   -1.055151
H      3.912839    1.399476    0.526700
H      1.659031   -2.099695   -0.982203
H     -0.654937   -2.441288   -0.206508
H     -2.480352   -1.720948    1.199584
H     -2.385481   -0.206299    2.091633
H     -3.057245    1.028306   -0.081258
H     -4.286638   -0.057261    0.585351
H     -2.477706   -0.653221   -1.512844
H     -0.697715    1.580222    1.353971
H      1.589759    1.913401    0.584322
