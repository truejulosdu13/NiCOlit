%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      3.166586    1.400015   -1.316004
N      2.975456    0.615773   -0.108654
C      4.151178    0.360019    0.700952
C      1.826036   -0.132403    0.126173
O      1.710020   -0.913330    1.065666
O      0.836920    0.105955   -0.824100
C     -0.345736   -0.621288   -0.618326
C     -0.409850   -1.919381   -1.130325
C     -1.581907   -2.651822   -0.982981
C     -2.682416   -2.079838   -0.341379
C     -2.631498   -0.764235    0.159768
C     -3.736334   -0.177188    0.796518
F     -4.875356   -0.873789    0.941710
C     -3.694755    1.124700    1.287440
C     -2.532459    1.874078    1.148503
C     -1.418268    1.313577    0.519633
C     -1.445411   -0.008063    0.019279
H      3.695332    0.784950   -2.050017
H      2.216922    1.728535   -1.744663
H      3.764274    2.285273   -1.079853
H      3.896220   -0.132117    1.643294
H      4.830811   -0.285429    0.136608
H      4.646976    1.309246    0.923832
H      0.449246   -2.357094   -1.629393
H     -1.642651   -3.667756   -1.364265
H     -3.593396   -2.666617   -0.231997
H     -4.570515    1.541545    1.774243
H     -2.488531    2.890913    1.529740
H     -0.516893    1.915772    0.424058
