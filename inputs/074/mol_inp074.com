%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      7.664013    1.598867   -0.008299
O      6.260232    1.825030   -0.013908
C      5.455402    0.719321   -0.048326
C      5.884093   -0.605469   -0.081949
C      4.948355   -1.645190   -0.115115
C      3.571084   -1.376201   -0.115428
C      2.621916   -2.407298   -0.147343
C      1.253853   -2.118280   -0.146914
C      0.794596   -0.794060   -0.116016
C     -0.645402   -0.490687   -0.118138
C     -1.506891   -1.051280    0.833240
C     -2.879284   -0.763638    0.834232
C     -3.445550    0.096138   -0.118557
C     -4.945988    0.434106   -0.148597
C     -5.550112   -0.005336   -1.499851
C     -5.138967    1.956378    0.022934
C     -5.764610   -0.258332    0.964472
C     -2.575089    0.655379   -1.071741
C     -1.204155    0.368781   -1.071221
C      1.751186    0.235269   -0.083938
C      3.131502   -0.036886   -0.082788
C      4.084639    0.994913   -0.048764
H      8.159541    2.574044    0.022114
H      7.985652    1.090529   -0.923291
H      7.970714    1.041765    0.883136
H      6.936507   -0.867705   -0.083251
H      5.306181   -2.672218   -0.140576
H      2.938337   -3.447441   -0.175848
H      0.543147   -2.941635   -0.184058
H     -1.112597   -1.715228    1.600366
H     -3.483603   -1.232569    1.605721
H     -5.402904   -1.078933   -1.667224
H     -6.627807    0.192928   -1.534195
H     -5.100266    0.524119   -2.346879
H     -4.693042    2.307249    0.961194
H     -6.202851    2.220533    0.039958
H     -4.680712    2.526854   -0.792362
H     -6.825380    0.012786    0.901543
H     -5.706588   -1.350386    0.888111
H     -5.416618    0.033469    1.962235
H     -2.953083    1.327828   -1.838907
H     -0.574998    0.818620   -1.836860
H      1.413600    1.270162   -0.047246
H      3.761947    2.033704   -0.021959
