%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C     -3.908872    0.479110   -0.289637
O     -2.671755    1.136056   -0.533995
C     -1.532447    0.446246   -0.220940
C     -1.473516   -0.838506    0.314334
C     -0.234855   -1.429126    0.588469
C      0.963298   -0.745179    0.331987
C      2.213454   -1.325520    0.602114
C      3.394382   -0.629030    0.340021
C      3.340526    0.654778   -0.194740
C      2.105629    1.244912   -0.468583
C      0.908368    0.557328   -0.210571
C     -0.343554    1.135087   -0.479667
H     -3.998639   -0.430701   -0.892352
H     -4.037276    0.262643    0.776071
H     -4.712825    1.157856   -0.590321
H     -2.368947   -1.410705    0.531385
H     -0.213553   -2.433293    1.006268
H      2.275780   -2.327664    1.020020
H      4.355290   -1.089647    0.553693
H      4.258969    1.198589   -0.399462
H      2.082690    2.248977   -0.886376
H     -0.402147    2.137786   -0.897719
