%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      5.446616   -0.896672    2.483282
C      5.293779    0.317236    1.580962
N      4.349202    0.068597    0.498180
C      4.861521   -0.492529   -0.748902
C      5.247749    0.595215   -1.738114
C      3.007554    0.382038    0.689249
O      2.562996    0.826740    1.742846
O      2.240314    0.138241   -0.449313
C      0.866480    0.409063   -0.367856
C      0.049964   -0.247258    0.554953
C     -1.320257    0.015215    0.561071
C     -1.859457    0.902732   -0.378403
C     -3.305691    1.203634   -0.378808
O     -3.638619    2.360277   -0.129817
N     -4.215319    0.180647   -0.660044
C     -3.854452   -1.136645   -1.195453
C     -4.155058   -2.269879   -0.228158
C     -5.639275    0.500050   -0.553121
C     -6.123389    0.496738    0.889473
C     -1.033830    1.559556   -1.298094
C      0.336534    1.300024   -1.301976
H      5.814530   -1.761787    1.922277
H      4.488472   -1.175026    2.934308
H      6.155965   -0.684811    3.289104
H      6.260655    0.599371    1.149992
H      4.947128    1.167217    2.179401
H      4.101600   -1.140537   -1.198440
H      5.726705   -1.125466   -0.522777
H      5.628817    0.150623   -2.662520
H      6.026036    1.244751   -1.324570
H      4.388150    1.224751   -1.989995
H      0.474291   -0.939802    1.276068
H     -1.959937   -0.457916    1.300445
H     -4.415097   -1.281112   -2.126552
H     -2.797865   -1.167237   -1.472803
H     -3.656898   -2.116420    0.733552
H     -5.229572   -2.355209   -0.038966
H     -3.813341   -3.222696   -0.644351
H     -6.218439   -0.217891   -1.144225
H     -5.812624    1.490136   -0.990628
H     -5.969203   -0.479964    1.357934
H     -7.191808    0.730082    0.930439
H     -5.590491    1.239958    1.491011
H     -1.459330    2.269870   -2.002644
H      0.984895    1.796094   -2.018017
