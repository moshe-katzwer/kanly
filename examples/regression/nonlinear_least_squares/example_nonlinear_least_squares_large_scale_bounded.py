import numpy as np
import pandas as pd

from kanly.api import nlls

n = 5_000_000
np.random.seed(0)

num_geo = 400

df = pd.DataFrame({
    'x': np.random.randn(n),
    'day': np.random.randint(0, 7, n),
    'geo': np.random.randint(0, num_geo, n),
})

day_fe = .1 * np.random.randn(7);          day_fe[0] = 0
geo_fe = 6 * np.random.rand(num_geo) - 1;  geo_fe[0] = 0

df['y'] = (1 + 2 * np.exp(df.x)) \
          * (1 + df.geo.map(dict(zip(range(num_geo), geo_fe)))) \
          * (1 + df.day.map(dict(zip(range(7), day_fe)))) \
          * (1 + .25 * np.random.randn(n))

fit = nlls('[y] ~ ({alpha} + {beta} * np.exp([x])) * (1 + [C(geo,-1)]) * (1 + [C(day,-1)])',
           df,
           debug=True,
           subsample=10_000,  # solve a smaller problem to get a starting point
           max_iter=100,
           bounds=np.array([(-np.inf, np.inf)] * 7 + [(-.4, 2)] * num_geo),
           )

print(fit)

"""
Nonlinear Least Squares...constructing model...

	Checking weights...no weights  (0.00s).
	Getting endog column ['y'] (0.03s).
	Parsing variable names and parameter names
	Building prediction function callable...

		Getting column 'x' of type 'NonlinearTermType.MONOMIAL'...0.04s
		Getting column 'C(geo,-1)' of type 'NonlinearTermType.CATEGORICAL'...1.04s
		Getting column 'C(day,-1)' of type 'NonlinearTermType.CATEGORICAL'...0.62s
	Prediction function complete (1.85s).

Checking valid rows across endog, prediction function and weights...all valid (1.86s).

Model complete!
======================================================================
Nonlinear Least Squares Model
----------------------------------------------------------------------
Dep Var:      y
Nobs:         5000000
Num Params:   407
Weights:      None

formula:      [y] ~ ({alpha} + {beta} * np.exp([x])) * (1 +
              [C(geo,-1)]) * (1 + [C(day,-1)])

Params:       alpha, beta, C(geo)[1], C(geo)[2], C(geo)[3],
              C(geo)[4], C(geo)[5], C(geo)[6],
              C(geo)[7], C(geo)[8], C(geo)[9],
              C(geo)[10], C(geo)[11], C(geo)[12],
              C(geo)[13], C(geo)[14], C(geo)[15],
              C(geo)[16], C(geo)[17], C(geo)[18],
              C(geo)[19], C(geo)[20], C(geo)[21],
              C(geo)[22], C(geo)[23], C(geo)[24],
              C(geo)[25], C(geo)[26], C(geo)[27],
              C(geo)[28], C(geo)[29], C(geo)[30],
              C(geo)[31], C(geo)[32], C(geo)[33],
              C(geo)[34], C(geo)[35], C(geo)[36],
              C(geo)[37], C(geo)[38], C(geo)[39],
              C(geo)[40], C(geo)[41], C(geo)[42],
              C(geo)[43], C(geo)[44], C(geo)[45],
              C(geo)[46], C(geo)[47], C(geo)[48],
              C(geo)[49], C(geo)[50], C(geo)[51],
              C(geo)[52], C(geo)[53], C(geo)[54],
              C(geo)[55], C(geo)[56], C(geo)[57],
              C(geo)[58], C(geo)[59], C(geo)[60],
              C(geo)[61], C(geo)[62], C(geo)[63],
              C(geo)[64], C(geo)[65], C(geo)[66],
              C(geo)[67], C(geo)[68], C(geo)[69],
              C(geo)[70], C(geo)[71], C(geo)[72],
              C(geo)[73], C(geo)[74], C(geo)[75],
              C(geo)[76], C(geo)[77], C(geo)[78],
              C(geo)[79], C(geo)[80], C(geo)[81],
              C(geo)[82], C(geo)[83], C(geo)[84],
              C(geo)[85], C(geo)[86], C(geo)[87],
              C(geo)[88], C(geo)[89], C(geo)[90],
              C(geo)[91], C(geo)[92], C(geo)[93],
              C(geo)[94], C(geo)[95], C(geo)[96],
              C(geo)[97], C(geo)[98], C(geo)[99],
              C(geo)[100], C(geo)[101], C(geo)[102],
              C(geo)[103], C(geo)[104], C(geo)[105],
              C(geo)[106], C(geo)[107], C(geo)[108],
              C(geo)[109], C(geo)[110], C(geo)[111],
              C(geo)[112], C(geo)[113], C(geo)[114],
              C(geo)[115], C(geo)[116], C(geo)[117],
              C(geo)[118], C(geo)[119], C(geo)[120],
              C(geo)[121], C(geo)[122], C(geo)[123],
              C(geo)[124], C(geo)[125], C(geo)[126],
              C(geo)[127], C(geo)[128], C(geo)[129],
              C(geo)[130], C(geo)[131], C(geo)[132],
              C(geo)[133], C(geo)[134], C(geo)[135],
              C(geo)[136], C(geo)[137], C(geo)[138],
              C(geo)[139], C(geo)[140], C(geo)[141],
              C(geo)[142], C(geo)[143], C(geo)[144],
              C(geo)[145], C(geo)[146], C(geo)[147],
              C(geo)[148], C(geo)[149], C(geo)[150],
              C(geo)[151], C(geo)[152], C(geo)[153],
              C(geo)[154], C(geo)[155], C(geo)[156],
              C(geo)[157], C(geo)[158], C(geo)[159],
              C(geo)[160], C(geo)[161], C(geo)[162],
              C(geo)[163], C(geo)[164], C(geo)[165],
              C(geo)[166], C(geo)[167], C(geo)[168],
              C(geo)[169], C(geo)[170], C(geo)[171],
              C(geo)[172], C(geo)[173], C(geo)[174],
              C(geo)[175], C(geo)[176], C(geo)[177],
              C(geo)[178], C(geo)[179], C(geo)[180],
              C(geo)[181], C(geo)[182], C(geo)[183],
              C(geo)[184], C(geo)[185], C(geo)[186],
              C(geo)[187], C(geo)[188], C(geo)[189],
              C(geo)[190], C(geo)[191], C(geo)[192],
              C(geo)[193], C(geo)[194], C(geo)[195],
              C(geo)[196], C(geo)[197], C(geo)[198],
              C(geo)[199], C(geo)[200], C(geo)[201],
              C(geo)[202], C(geo)[203], C(geo)[204],
              C(geo)[205], C(geo)[206], C(geo)[207],
              C(geo)[208], C(geo)[209], C(geo)[210],
              C(geo)[211], C(geo)[212], C(geo)[213],
              C(geo)[214], C(geo)[215], C(geo)[216],
              C(geo)[217], C(geo)[218], C(geo)[219],
              C(geo)[220], C(geo)[221], C(geo)[222],
              C(geo)[223], C(geo)[224], C(geo)[225],
              C(geo)[226], C(geo)[227], C(geo)[228],
              C(geo)[229], C(geo)[230], C(geo)[231],
              C(geo)[232], C(geo)[233], C(geo)[234],
              C(geo)[235], C(geo)[236], C(geo)[237],
              C(geo)[238], C(geo)[239], C(geo)[240],
              C(geo)[241], C(geo)[242], C(geo)[243],
              C(geo)[244], C(geo)[245], C(geo)[246],
              C(geo)[247], C(geo)[248], C(geo)[249],
              C(geo)[250], C(geo)[251], C(geo)[252],
              C(geo)[253], C(geo)[254], C(geo)[255],
              C(geo)[256], C(geo)[257], C(geo)[258],
              C(geo)[259], C(geo)[260], C(geo)[261],
              C(geo)[262], C(geo)[263], C(geo)[264],
              C(geo)[265], C(geo)[266], C(geo)[267],
              C(geo)[268], C(geo)[269], C(geo)[270],
              C(geo)[271], C(geo)[272], C(geo)[273],
              C(geo)[274], C(geo)[275], C(geo)[276],
              C(geo)[277], C(geo)[278], C(geo)[279],
              C(geo)[280], C(geo)[281], C(geo)[282],
              C(geo)[283], C(geo)[284], C(geo)[285],
              C(geo)[286], C(geo)[287], C(geo)[288],
              C(geo)[289], C(geo)[290], C(geo)[291],
              C(geo)[292], C(geo)[293], C(geo)[294],
              C(geo)[295], C(geo)[296], C(geo)[297],
              C(geo)[298], C(geo)[299], C(geo)[300],
              C(geo)[301], C(geo)[302], C(geo)[303],
              C(geo)[304], C(geo)[305], C(geo)[306],
              C(geo)[307], C(geo)[308], C(geo)[309],
              C(geo)[310], C(geo)[311], C(geo)[312],
              C(geo)[313], C(geo)[314], C(geo)[315],
              C(geo)[316], C(geo)[317], C(geo)[318],
              C(geo)[319], C(geo)[320], C(geo)[321],
              C(geo)[322], C(geo)[323], C(geo)[324],
              C(geo)[325], C(geo)[326], C(geo)[327],
              C(geo)[328], C(geo)[329], C(geo)[330],
              C(geo)[331], C(geo)[332], C(geo)[333],
              C(geo)[334], C(geo)[335], C(geo)[336],
              C(geo)[337], C(geo)[338], C(geo)[339],
              C(geo)[340], C(geo)[341], C(geo)[342],
              C(geo)[343], C(geo)[344], C(geo)[345],
              C(geo)[346], C(geo)[347], C(geo)[348],
              C(geo)[349], C(geo)[350], C(geo)[351],
              C(geo)[352], C(geo)[353], C(geo)[354],
              C(geo)[355], C(geo)[356], C(geo)[357],
              C(geo)[358], C(geo)[359], C(geo)[360],
              C(geo)[361], C(geo)[362], C(geo)[363],
              C(geo)[364], C(geo)[365], C(geo)[366],
              C(geo)[367], C(geo)[368], C(geo)[369],
              C(geo)[370], C(geo)[371], C(geo)[372],
              C(geo)[373], C(geo)[374], C(geo)[375],
              C(geo)[376], C(geo)[377], C(geo)[378],
              C(geo)[379], C(geo)[380], C(geo)[381],
              C(geo)[382], C(geo)[383], C(geo)[384],
              C(geo)[385], C(geo)[386], C(geo)[387],
              C(geo)[388], C(geo)[389], C(geo)[390],
              C(geo)[391], C(geo)[392], C(geo)[393],
              C(geo)[394], C(geo)[395], C(geo)[396],
              C(geo)[397], C(geo)[398], C(geo)[399],
              C(day)[1], C(day)[2], C(day)[3],
              C(day)[4], C(day)[5], C(day)[6]
======================================================================


Estimating a starting point on 10000/5000000 random observations...

==============================
Nonlinear Least Squares
------------------------------
Nobs:         10000
Num Params:   407
Initial Cost: 2.52e+06
Bounded:      True

maxiter:      100
gtol:         1.49e-06
ftol:         1.49e-06
xtol:         1.49e-06
Delta:        2.017e+01

rho_reject:          2.50e-01
rho_accept:          7.50e-01
min_rho_step:        1.00e-01
Delta_scale_up:      1.50e+00
Delta_scale_down:    3.00e+00
Delta_floor:         1.49e-08
==============================

==============================================================================================================================================
  iter       F=cost       dF/F     pred/F          F/n       rho     Delta      |dx|  optimality   active bnds  accepted       time  step type
----------------------------------------------------------------------------------------------------------------------------------------------
     0   7.2847e+05  -7.10e-01  -7.10e-01   7.2847e+01  1.00e+00  2.02e+01  6.91e+00    9.79e+04             0      True      4.44s   steihaug
     1   2.9965e+05  -5.89e-01  -5.33e-01   2.9965e+01  7.60e-01  2.02e+01  7.25e+00    2.20e+05             0      True      4.51s   steihaug
     2   1.5210e+05  -4.92e-01  -3.65e-01   1.5210e+01  1.05e+00  2.02e+01  5.05e+00    3.14e+04             0      True      4.58s   steihaug
     3   1.3118e+05  -1.38e-01  -9.96e-02   1.3118e+01  8.05e-01  2.02e+01  4.45e+00    3.73e+04             0      True      4.65s   steihaug
     4   1.2492e+05  -4.77e-02  -3.10e-02   1.2492e+01  9.42e-01  2.02e+01  2.71e+00    7.28e+03             0      True      4.73s   stei-bnd
     5   1.2300e+05  -1.53e-02  -9.58e-03   1.2300e+01  9.24e-01  2.02e+01  2.43e+00    5.56e+03            22      True      4.82s    reflect
     6   1.2271e+05  -2.35e-03  -1.47e-03   1.2271e+01  9.87e-01  2.02e+01  9.28e-01    4.94e+02            53      True      4.91s    reflect
     7   1.2269e+05  -2.25e-04  -1.51e-04   1.2269e+01  9.94e-01  2.02e+01  4.57e-01    1.77e+02            66      True      4.99s    reflect
     8   1.2268e+05  -2.01e-05  -1.30e-05   1.2268e+01  1.00e+00  2.02e+01  1.81e-01    2.92e+01            81      True      5.09s    reflect
     9   1.2268e+05  -1.87e-06  -1.18e-06   1.2268e+01  1.00e+00  2.02e+01  8.20e-02    6.07e+00            87      True      5.19s    reflect
    10   1.2268e+05  -1.85e-07  -1.17e-07   1.2268e+01  1.00e+00  2.02e+01  3.55e-02    2.62e+00            91      True      5.30s    reflect
==============================================================================================================================================

	Converged: |dx| < xtol * max(1, |x|)


Beginning full estimation...

==============================
Nonlinear Least Squares
------------------------------
Nobs:         5000000
Num Params:   407
Initial Cost: 9.54e+07
Bounded:      True

maxiter:      100
gtol:         1.49e-08
ftol:         1.49e-08
xtol:         1.49e-08
Delta:        9.897e+01

rho_reject:          2.50e-01
rho_accept:          7.50e-01
min_rho_step:        1.00e-01
Delta_scale_up:      1.50e+00
Delta_scale_down:    3.00e+00
Delta_floor:         1.49e-08
==============================

==============================================================================================================================================
  iter       F=cost       dF/F     pred/F          F/n       rho     Delta      |dx|  optimality   active bnds  accepted       time  step type
----------------------------------------------------------------------------------------------------------------------------------------------
     0   8.6348e+07  -9.47e-02  -8.55e-02   1.7270e+01  9.90e-01  9.90e+01  1.91e+00    1.39e+07             0      True     98.82s    reflect
     1   8.5300e+07  -1.21e-02  -1.17e-02   1.7060e+01  9.95e-01  9.90e+01  5.33e-01    1.02e+07             0      True    140.36s   stei-bnd
     2   8.4670e+07  -7.39e-03  -7.13e-03   1.6934e+01  9.96e-01  9.90e+01  4.40e-01    7.85e+06             1      True    182.63s   stei-bnd
     3   8.4045e+07  -7.39e-03  -6.92e-03   1.6809e+01  9.94e-01  9.90e+01  6.18e-01    5.12e+06             1      True    220.89s   stei-bnd
     4   8.4021e+07  -2.85e-04  -2.84e-04   1.6804e+01  1.00e+00  9.90e+01  8.43e-03    5.85e+05             1      True    260.63s     cauchy
     5   8.3361e+07  -7.85e-03  -5.94e-03   1.6672e+01  9.85e-01  9.90e+01  1.39e+00    9.89e+05             1      True    297.95s    reflect
     6   8.3268e+07  -1.12e-03  -7.74e-04   1.6654e+01  9.95e-01  9.90e+01  6.84e-01    1.60e+05             1      True    336.35s   stei-bnd
     7   8.3244e+07  -2.88e-04  -1.73e-04   1.6649e+01  9.97e-01  9.90e+01  4.66e-01    2.20e+04            52      True    372.94s    reflect
     8   8.3242e+07  -2.64e-05  -1.71e-05   1.6648e+01  9.99e-01  9.90e+01  1.85e-01    3.73e+03            82      True    409.90s    reflect
     9   8.3241e+07  -2.84e-06  -2.01e-06   1.6648e+01  1.00e+00  9.90e+01  9.22e-02    5.55e+02            95      True    446.48s    reflect
    10   8.3241e+07  -4.53e-07  -3.26e-07   1.6648e+01  1.00e+00  9.90e+01  5.11e-02    2.40e+02           103      True    485.82s    reflect
    11   8.3241e+07  -7.21e-08  -4.93e-08   1.6648e+01  1.00e+00  9.90e+01  2.95e-02    9.25e+01           105      True    526.58s    reflect
    12   8.3241e+07  -1.15e-08  -6.37e-09   1.6648e+01  1.00e+00  9.90e+01  1.53e-02    2.11e+01           105      True    569.85s   stei-bnd
==============================================================================================================================================

	Converged: |dF| < ftol * max(1, |F|)

Cannot compute variance covariance with active constraints!
NLLS Estimation Complete!


==========================================================================
Nonlinear Least Squares Results
==========================================================================

Dep. Variable: y

Date:                  Sep 22, 2022    Adj. R-squared:              0.8982
Time:                      13:14:04    Model Time:                   1.86s
Weights:                       None    Fit Time:                   585.45s
Nobs:                       5000000    Cov Time:                     0.00s
Df Residuals:               4999593    Iterations:                      24
Df Model:                       407    Converged:                     True
Cost:                    8.3241e+07    Status:                           2
Optimality:                2.11e+01    Covariance Type:               None
R-squared:                   0.8982    Active Constraints:             105

==========================================================================
                coef
--------------------------------------------------------------------------
alpha        1.82490
beta         3.64381
C(geo)[1]   -0.98312
C(geo)[2]   -0.96217
C(geo)[3]    1.71288
C(geo)[4]    1.72589
C(geo)[5]    1.25073
C(geo)[6]    0.03878
C(geo)[7]   -0.40000
C(geo)[8]    1.34365
C(geo)[9]    0.54408
C(geo)[10]   0.53964
C(geo)[11]   0.45603
C(geo)[12]   1.68549
C(geo)[13]   1.74580
C(geo)[14]   1.27054
C(geo)[15]   0.16429
C(geo)[16]  -0.40000
C(geo)[17]   0.63911
C(geo)[18]   0.87158
C(geo)[19]  -0.40000
C(geo)[20]  -0.27478
C(geo)[21]   1.44908
C(geo)[22]   2.00000
C(geo)[23]   2.00000
C(geo)[24]   1.02230
C(geo)[25]   1.97898
C(geo)[26]   0.26930
C(geo)[27]   0.65943
C(geo)[28]  -0.40000
C(geo)[29]   2.00000
C(geo)[30]   1.31517
C(geo)[31]   1.86930
C(geo)[32]   0.98712
C(geo)[33]   1.30726
C(geo)[34]   1.99230
C(geo)[35]  -0.40000
C(geo)[36]  -0.02190
C(geo)[37]   1.53772
C(geo)[38]   0.35449
C(geo)[39]   0.81309
C(geo)[40]  -0.12502
C(geo)[41]   1.94181
C(geo)[42]   0.58080
C(geo)[43]  -0.40000
C(geo)[44]   2.00000
C(geo)[45]   1.72246
C(geo)[46]   0.46642
C(geo)[47]   1.22377
C(geo)[48]   0.38587
C(geo)[49]   1.38707
C(geo)[50]   1.57668
C(geo)[51]   0.79953
C(geo)[52]   1.90737
C(geo)[53]   1.77133
C(geo)[54]   0.94280
C(geo)[55]   2.00000
C(geo)[56]   1.66225
C(geo)[57]   1.90744
C(geo)[58]  -0.39551
C(geo)[59]   0.92206
C(geo)[60]   0.94535
C(geo)[61]  -0.40000
C(geo)[62]   1.52175
C(geo)[63]   2.00000
C(geo)[64]   0.73491
C(geo)[65]   0.40652
C(geo)[66]   0.71112
C(geo)[67]  -0.40000
C(geo)[68]   0.51787
C(geo)[69]   1.93725
C(geo)[70]   0.56346
C(geo)[71]   1.97427
C(geo)[72]  -0.40000
C(geo)[73]  -0.02473
C(geo)[74]   1.93148
C(geo)[75]   1.88208
C(geo)[76]   2.00000
C(geo)[77]   1.08679
C(geo)[78]  -0.10430
C(geo)[79]  -0.27568
C(geo)[80]  -0.13755
C(geo)[81]   2.00000
C(geo)[82]   1.04655
C(geo)[83]  -0.40000
C(geo)[84]  -0.30465
C(geo)[85]   1.25651
C(geo)[86]   1.95469
C(geo)[87]   0.06165
C(geo)[88]   0.50652
C(geo)[89]   1.73049
C(geo)[90]   0.51169
C(geo)[91]  -0.21400
C(geo)[92]   2.00000
C(geo)[93]  -0.40000
C(geo)[94]   0.36233
C(geo)[95]   1.71053
C(geo)[96]   0.85342
C(geo)[97]   1.62737
C(geo)[98]  -0.40000
C(geo)[99]   0.73974
C(geo)[100]  1.50765
C(geo)[101]  0.51691
C(geo)[102]  1.70268
C(geo)[103]  1.83084
C(geo)[104] -0.40000
C(geo)[105] -0.40000
C(geo)[106]  1.13705
C(geo)[107]  0.56143
C(geo)[108] -0.40000
C(geo)[109] -0.40000
C(geo)[110]  1.51225
C(geo)[111]  0.28042
C(geo)[112]  0.91565
C(geo)[113] -0.40000
C(geo)[114] -0.26965
C(geo)[115] -0.36522
C(geo)[116]  2.00000
C(geo)[117]  1.38044
C(geo)[118]  1.03691
C(geo)[119]  1.94836
C(geo)[120]  0.31686
C(geo)[121]  1.12389
C(geo)[122] -0.10864
C(geo)[123]  0.64710
C(geo)[124] -0.40000
C(geo)[125]  1.16156
C(geo)[126] -0.18331
C(geo)[127] -0.24875
C(geo)[128]  0.69170
C(geo)[129] -0.31566
C(geo)[130] -0.40000
C(geo)[131]  1.05390
C(geo)[132] -0.40000
C(geo)[133] -0.24799
C(geo)[134]  2.00000
C(geo)[135]  2.00000
C(geo)[136]  0.67105
C(geo)[137]  1.45601
C(geo)[138]  0.66370
C(geo)[139] -0.34764
C(geo)[140] -0.37720
C(geo)[141] -0.40000
C(geo)[142]  0.21236
C(geo)[143]  2.00000
C(geo)[144]  0.36976
C(geo)[145] -0.40000
C(geo)[146]  1.13975
C(geo)[147] -0.40000
C(geo)[148]  0.48987
C(geo)[149]  0.41917
C(geo)[150]  1.60503
C(geo)[151]  2.00000
C(geo)[152]  1.37089
C(geo)[153]  0.49201
C(geo)[154]  2.00000
C(geo)[155]  1.54870
C(geo)[156] -0.40000
C(geo)[157] -0.13670
C(geo)[158]  1.12595
C(geo)[159]  0.56495
C(geo)[160]  0.95534
C(geo)[161] -0.40000
C(geo)[162] -0.40000
C(geo)[163] -0.30098
C(geo)[164]  1.74955
C(geo)[165] -0.40000
C(geo)[166]  0.24535
C(geo)[167]  0.21875
C(geo)[168]  0.77082
C(geo)[169]  0.31976
C(geo)[170]  0.60427
C(geo)[171]  0.34526
C(geo)[172]  0.84249
C(geo)[173]  0.12380
C(geo)[174]  2.00000
C(geo)[175]  0.75629
C(geo)[176]  1.21256
C(geo)[177]  2.00000
C(geo)[178]  0.06684
C(geo)[179] -0.13824
C(geo)[180] -0.40000
C(geo)[181]  1.19830
C(geo)[182] -0.40000
C(geo)[183]  0.89245
C(geo)[184]  0.71972
C(geo)[185] -0.40000
C(geo)[186]  0.22793
C(geo)[187]  0.43887
C(geo)[188] -0.40000
C(geo)[189]  2.00000
C(geo)[190] -0.11712
C(geo)[191]  1.51226
C(geo)[192] -0.34887
C(geo)[193] -0.40000
C(geo)[194]  0.75219
C(geo)[195]  1.79790
C(geo)[196]  1.23462
C(geo)[197]  0.03142
C(geo)[198] -0.40000
C(geo)[199]  1.42710
C(geo)[200] -0.40000
C(geo)[201] -0.40000
C(geo)[202]  1.35359
C(geo)[203]  0.49962
C(geo)[204]  0.30381
C(geo)[205] -0.28835
C(geo)[206] -0.14986
C(geo)[207]  2.00000
C(geo)[208]  1.88391
C(geo)[209]  2.00000
C(geo)[210]  0.85821
C(geo)[211]  1.38802
C(geo)[212]  1.92801
C(geo)[213]  0.69284
C(geo)[214]  0.89090
C(geo)[215] -0.40000
C(geo)[216]  1.96515
C(geo)[217]  0.57485
C(geo)[218]  1.10689
C(geo)[219] -0.40000
C(geo)[220]  1.23416
C(geo)[221] -0.40000
C(geo)[222]  1.43331
C(geo)[223]  0.71587
C(geo)[224]  0.42567
C(geo)[225] -0.22240
C(geo)[226] -0.25984
C(geo)[227]  1.61538
C(geo)[228]  0.67118
C(geo)[229]  2.00000
C(geo)[230]  1.32363
C(geo)[231]  0.81661
C(geo)[232]  1.65559
C(geo)[233]  0.75831
C(geo)[234]  2.00000
C(geo)[235]  1.87529
C(geo)[236] -0.40000
C(geo)[237]  1.73873
C(geo)[238] -0.40000
C(geo)[239]  1.37592
C(geo)[240]  2.00000
C(geo)[241]  1.49009
C(geo)[242]  0.83171
C(geo)[243]  0.15569
C(geo)[244]  1.26577
C(geo)[245] -0.16627
C(geo)[246] -0.40000
C(geo)[247] -0.40000
C(geo)[248] -0.40000
C(geo)[249]  1.99522
C(geo)[250] -0.40000
C(geo)[251]  0.59870
C(geo)[252]  1.54760
C(geo)[253] -0.40000
C(geo)[254]  1.94340
C(geo)[255]  0.37822
C(geo)[256]  0.58828
C(geo)[257]  0.88663
C(geo)[258] -0.40000
C(geo)[259]  0.24222
C(geo)[260]  0.58453
C(geo)[261]  1.91747
C(geo)[262] -0.40000
C(geo)[263]  0.40285
C(geo)[264] -0.40000
C(geo)[265]  0.55251
C(geo)[266]  0.89491
C(geo)[267]  0.15262
C(geo)[268]  0.39744
C(geo)[269]  0.16683
C(geo)[270] -0.40000
C(geo)[271]  1.29687
C(geo)[272]  1.52332
C(geo)[273]  1.60847
C(geo)[274]  0.06370
C(geo)[275]  0.45259
C(geo)[276]  1.30888
C(geo)[277]  1.90262
C(geo)[278]  1.83640
C(geo)[279] -0.40000
C(geo)[280]  1.54972
C(geo)[281]  0.17675
C(geo)[282]  0.22325
C(geo)[283]  0.74675
C(geo)[284]  1.19453
C(geo)[285]  1.51349
C(geo)[286]  0.96072
C(geo)[287]  2.00000
C(geo)[288] -0.40000
C(geo)[289] -0.04833
C(geo)[290] -0.40000
C(geo)[291] -0.40000
C(geo)[292] -0.05512
C(geo)[293]  0.92491
C(geo)[294] -0.40000
C(geo)[295]  1.15033
C(geo)[296] -0.40000
C(geo)[297]  1.99167
C(geo)[298] -0.14186
C(geo)[299]  1.33004
C(geo)[300] -0.40000
C(geo)[301]  1.38761
C(geo)[302]  1.54516
C(geo)[303]  0.55185
C(geo)[304]  1.93422
C(geo)[305]  1.43196
C(geo)[306]  1.99997
C(geo)[307]  0.96975
C(geo)[308]  1.38764
C(geo)[309]  0.87261
C(geo)[310]  0.91560
C(geo)[311]  0.24346
C(geo)[312]  1.41990
C(geo)[313]  0.52522
C(geo)[314]  2.00000
C(geo)[315] -0.40000
C(geo)[316]  1.20571
C(geo)[317]  2.00000
C(geo)[318]  0.76309
C(geo)[319] -0.40000
C(geo)[320] -0.13983
C(geo)[321]  1.38592
C(geo)[322]  0.20285
C(geo)[323]  1.22572
C(geo)[324] -0.16001
C(geo)[325]  0.00708
C(geo)[326] -0.30610
C(geo)[327] -0.35413
C(geo)[328]  0.12324
C(geo)[329] -0.32093
C(geo)[330]  2.00000
C(geo)[331]  0.82484
C(geo)[332]  1.19857
C(geo)[333]  0.59120
C(geo)[334] -0.40000
C(geo)[335]  0.82064
C(geo)[336]  0.94721
C(geo)[337]  0.77102
C(geo)[338]  0.56856
C(geo)[339]  1.36602
C(geo)[340]  1.63822
C(geo)[341] -0.40000
C(geo)[342]  2.00000
C(geo)[343]  0.36646
C(geo)[344]  1.89888
C(geo)[345] -0.40000
C(geo)[346] -0.11284
C(geo)[347] -0.40000
C(geo)[348]  0.86094
C(geo)[349]  1.49406
C(geo)[350]  0.23960
C(geo)[351] -0.40000
C(geo)[352] -0.06261
C(geo)[353]  2.00000
C(geo)[354]  1.99123
C(geo)[355]  0.58230
C(geo)[356]  0.71187
C(geo)[357]  2.00000
C(geo)[358]  1.06835
C(geo)[359]  1.41170
C(geo)[360]  1.67457
C(geo)[361]  1.30000
C(geo)[362]  1.56612
C(geo)[363]  0.35219
C(geo)[364] -0.30986
C(geo)[365]  0.56700
C(geo)[366]  0.98289
C(geo)[367]  1.73164
C(geo)[368]  2.00000
C(geo)[369] -0.40000
C(geo)[370]  0.14623
C(geo)[371]  1.88229
C(geo)[372]  0.48297
C(geo)[373]  0.39628
C(geo)[374] -0.40000
C(geo)[375]  1.19851
C(geo)[376]  0.25695
C(geo)[377] -0.40000
C(geo)[378]  1.39509
C(geo)[379]  0.27723
C(geo)[380]  2.00000
C(geo)[381]  1.07700
C(geo)[382] -0.21908
C(geo)[383] -0.40000
C(geo)[384] -0.40000
C(geo)[385]  1.82956
C(geo)[386]  2.00000
C(geo)[387]  0.61905
C(geo)[388] -0.14785
C(geo)[389]  1.51453
C(geo)[390]  1.47853
C(geo)[391] -0.01001
C(geo)[392]  1.73104
C(geo)[393] -0.40000
C(geo)[394] -0.40000
C(geo)[395]  0.26978
C(geo)[396] -0.40000
C(geo)[397] -0.18436
C(geo)[398] -0.25009
C(geo)[399]  2.00000
C(day)[1]    0.01260
C(day)[2]    0.02739
C(day)[3]    0.14066
C(day)[4]   -0.22560
C(day)[5]    0.13127
C(day)[6]    0.02722
==========================================================================


[y] ~ ({alpha} + {beta} * np.exp([x])) * (1 + [C(geo,-1)]) * (1 +
[C(day,-1)])

message: |dF| < ftol * max(1, |F|)

                                      [kanly package by moshe, v=0.0.255]


Process finished with exit code 0
"""
