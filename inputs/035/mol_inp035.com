%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      5.070496    0.590436   -0.698614
O      4.372323   -0.342482    0.116310
C      3.007312   -0.343178    0.022943
C      2.365087   -1.236681    0.887797
C      0.973630   -1.340564    0.899287
C      0.226586   -0.534322    0.046468
O     -1.163906   -0.706750    0.044154
C     -1.937434    0.408599    0.361806
O     -1.477824    1.507725    0.656653
N     -3.301449    0.140993    0.303868
C     -4.237422    1.237186    0.462086
C     -3.835263   -1.111572   -0.201465
C      0.848827    0.344023   -0.838327
C      2.243494    0.444768   -0.842552
H      6.138876    0.485465   -0.486457
H      4.783010    1.619391   -0.458151
H      4.919825    0.374980   -1.761552
H      2.957011   -1.855805    1.557295
H      0.483205   -2.039850    1.568860
H     -5.082208    0.902271    1.070961
H     -4.594908    1.537307   -0.527417
H     -3.775106    2.100353    0.948207
H     -3.124374   -1.934011   -0.094086
H     -4.075511   -0.984392   -1.261082
H     -4.743867   -1.364939    0.352607
H      0.259371    0.955574   -1.514916
H      2.700221    1.145476   -1.532960
