%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
O      1.234096    1.185375    1.468554
C      1.620005    0.447987    0.578816
O      0.926191   -0.487733   -0.157552
C     -0.417070   -0.555961    0.210028
C     -0.804048   -1.430842    1.219511
C     -2.155841   -1.532468    1.547608
C     -3.120962   -0.774176    0.861223
C     -4.484775   -0.867194    1.181241
C     -5.433545   -0.111271    0.490430
C     -5.032093    0.747271   -0.528900
C     -3.680506    0.852998   -0.860708
C     -2.712965    0.100432   -0.175938
C     -1.351533    0.196825   -0.499020
C      3.000156    0.420816    0.094168
C      3.935707   -0.421574    0.701737
Cl     3.518778   -1.478756    2.002289
C      5.258004   -0.432470    0.247999
C      5.644165    0.395883   -0.806662
C      4.708406    1.236194   -1.411487
C      3.384616    1.251662   -0.962290
Cl     2.272964    2.306224   -1.759300
H     -0.065972   -2.022870    1.751688
H     -2.453918   -2.209235    2.345096
H     -4.818885   -1.531172    1.975281
H     -6.485695   -0.192635    0.750377
H     -5.770175    1.337420   -1.065694
H     -3.386175    1.531632   -1.658180
H     -1.015934    0.857219   -1.293536
H      5.994302   -1.084880    0.713428
H      6.674374    0.386671   -1.157106
H      5.018330    1.878624   -2.233450

