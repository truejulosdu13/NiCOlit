%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      3.407420   -0.545897    1.734953
C      2.550013   -1.441523    0.854474
N      2.297651   -0.873684   -0.468544
C      3.172767   -1.245083   -1.575646
C      4.371172   -0.321507   -1.722560
C      1.340261    0.117154   -0.683072
O      1.173115    0.663841   -1.769237
O      0.618385    0.407213    0.469478
C     -0.392353    1.366100    0.282562
C     -0.047597    2.718963    0.390080
C     -1.028061    3.698525    0.259942
C     -2.366226    3.354123    0.038281
C     -2.688475    1.992056   -0.041060
N     -3.902983    1.379508   -0.239509
C     -3.773872    0.010888   -0.233542
C     -4.744113   -0.986517   -0.390117
C     -4.318993   -2.317175   -0.343122
C     -2.977411   -2.634637   -0.148015
C     -2.021275   -1.624245    0.006626
C     -2.422462   -0.272021   -0.035664
C     -1.734742    0.980041    0.081359
H      3.518295   -0.990428    2.729006
H      4.407835   -0.413599    1.314886
H      2.961927    0.446610    1.854088
H      1.587655   -1.612566    1.350081
H      3.021555   -2.423320    0.738052
H      3.500582   -2.281509   -1.441145
H      2.591554   -1.215681   -2.504849
H      4.059958    0.722748   -1.826921
H      4.951697   -0.595295   -2.609165
H      5.034377   -0.389653   -0.856234
H      0.985833    3.003884    0.563686
H     -0.750289    4.747749    0.329854
H     -3.125583    4.122966   -0.064165
H     -4.775519    1.869530   -0.368007
H     -5.790073   -0.742872   -0.544816
H     -5.047551   -3.115726   -0.463053
H     -2.666699   -3.675854   -0.117317
H     -0.977774   -1.883109    0.155749
