%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      4.385132   -1.823312   -0.688220
O      3.732871   -0.580309   -0.465910
C      2.364289   -0.590105   -0.322594
C      1.588020   -1.747852   -0.383521
C      0.205928   -1.673574   -0.228372
C     -0.436796   -0.445022   -0.008826
C     -1.900682   -0.450964    0.145098
C     -2.487019   -0.605082    1.410715
C     -3.875885   -0.614331    1.556190
C     -4.695000   -0.471855    0.437758
C     -4.127578   -0.320434   -0.826068
C     -2.738737   -0.311150   -0.971787
C      0.337314    0.746815    0.056518
C     -0.241197    2.021203    0.274920
C      0.529296    3.182812    0.336913
C      1.903617    3.103721    0.181986
C      2.502524    1.863056   -0.034405
C      1.748336    0.668989   -0.102126
H      4.242002   -2.501809    0.159115
H      5.457803   -1.625890   -0.777082
H      4.053576   -2.281443   -1.625774
H      2.023858   -2.726654   -0.550373
H     -0.381174   -2.589332   -0.279400
H     -1.857548   -0.715029    2.291529
H     -4.316808   -0.732208    2.542540
H     -5.775961   -0.478707    0.551118
H     -4.764973   -0.208900   -1.699272
H     -2.307092   -0.190096   -1.963344
H     -1.317715    2.123001    0.401189
H      0.051215    4.143723    0.506013
H      2.513992    4.001237    0.228291
H      3.584393    1.829499   -0.152819
