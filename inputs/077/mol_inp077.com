%chk=opt.chk
%mem=2gb
%nproc=1
#p b3lyp 6-31G* opt scf(xqc,tight)

opt calculation

0 1
C      2.626946   -0.929707   -0.093256
O      1.714041    0.153932    0.017894
C      0.376323   -0.144513   -0.014518
C     -0.152683   -1.432387   -0.148412
C     -1.535651   -1.631005   -0.170544
C     -2.399757   -0.546078   -0.059242
C     -1.883075    0.741256    0.074583
C     -0.496121    0.953967    0.098179
C      0.041481    2.350924    0.243370
H      2.515610   -1.627635    0.743085
H      3.640111   -0.518516   -0.049552
H      2.517631   -1.441495   -1.055071
H      0.483983   -2.305823   -0.238083
H     -1.937250   -2.635593   -0.274974
H     -3.475238   -0.701295   -0.076533
H     -2.569133    1.580740    0.160685
H      0.650473    2.615272   -0.627185
H      0.648466    2.431237    1.150967
H     -0.766156    3.086718    0.318610
