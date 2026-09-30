# Chronological yield-curve hedge research

## Measurement and units

The input is the committed Federal Reserve nominal zero-curve snapshot. Rates are converted from percentages to decimals. A yield shock is measured in yield basis points; portfolio P&L is measured in basis points of the target's market value at the current annual calibration. Observations are consecutive complete published curves, which can be separated by holidays or missing rows.

For cash flows $c_j$ at annual maturities $t_j$ and continuously compounded zero rates $y_j$,

$$P(y)=\sum_j c_j e^{-t_jy_j},\qquad
\frac{\partial P}{\partial y_j}=-t_jc_je^{-t_jy_j},\qquad
\frac{\partial^2P}{\partial y_j^2}=t_j^2c_je^{-t_jy_j}.$$

There are no cross-maturity Hessian terms under independently quoted zero rates. The target has 15 annual cash flows, a 4% coupon and unit principal. Hedges are zero-coupon claims at 2, 5, 10, 20 and 30 years.

At calibration $y_0$, let $d$ be the target sensitivity divided by its price and let $B$ contain normalized hedge sensitivities. The first-order net exposure is $e(w)=d+Bw$. The normalization makes $e^T\Delta y_{bp}$ directly comparable with portfolio P&L in basis points.

## Common feasible set and covariance objectives

All hedge comparisons use

$$\mathbf1^T(d+Bw)=0,\qquad \lVert w\rVert_1\le1.5.$$

The first constraint removes parallel DV01. It does not eliminate convexity, basis risk or nonparallel yield changes. Positive and negative parts represent the L1 bound; SLSQP uses analytic gradients. A $10^{-10}$ numerical regularizer on the split variables selects among nearly flat solutions.

The empirical covariance $S$ uses centered shocks and denominator $T-1$. Diagonal shrinkage is

$$\widehat\Sigma_\alpha=(1-\alpha)S+\alpha\operatorname{diag}(S).$$

The covariance hedge minimizes $e^T\widehat\Sigma_\alpha e$. The PCA hedge replaces this matrix with $V_k\Lambda_kV_k^T$, retaining the largest $k$ empirical eigenvalues. Truncation intentionally assigns zero objective weight to the omitted directions. This can improve factor neutrality while overlooking residual risk.

The minimax hedge uses an epigraph:

$$\min_{w,t}t\quad\text{subject to}\quad
e(w)^T\widehat\Sigma_{r,\alpha}e(w)\le t,\ r=1,2,3,$$

together with the common constraints. The three regimes are equal chronological blocks of the same past window. Their shrinkage intensity matches the selected global covariance model, reducing a confound in the original fixed-calibration comparison. These blocks are historical partitions, not estimated latent regimes.

## Selection and chronology

Each annual evaluation uses 1,260 earlier observed shocks. The first 1,008 fit candidate models and the last 252 validate them. Calibration for validation occurs immediately before the first validation shock. Candidate scores are means of squared exact repricing P&L, not in-sample quadratic objectives.

The fixed grids contain seven shrinkage intensities and five factor counts. A tie selects the smaller parameter. The selected settings are then refitted on all 1,260 past shocks, and faces remain fixed until the next annual calibration. Validation dates, chosen parameters and weight fits are saved.

Annual target faces are reset to one unit of market value. Therefore, the concatenated series compares consistently normalized shock risks and does not represent a self-financing trading strategy. The fixed-calibration study in the parent results directory uses a different experiment and should not be pooled with this one.

## Conditional variance and tail forecasts

An EWMA second-moment model assumes a zero conditional shock mean:

$$Q_{t+1}=\lambda Q_t+(1-\lambda)\Delta y_t\Delta y_t^T,\qquad \lambda=0.97.$$

The initial state is a training covariance subsequently passed through the past window. At every held-out date, current discounted cash flows and hedge prices give a dynamic exposure $e_t$ for the fixed face positions. The pre-shock variance is $\sigma_t^2=e_t^TQ_te_t$. Today's shock enters the filter only after today's forecast and P&L are recorded.

For a 1% upper loss tail, the Gaussian threshold is $\sigma_t\Phi^{-1}(0.99)$. The Student threshold is $\sigma_t\sqrt{(\nu-2)/\nu}\,t_\nu^{-1}(0.99)$ with fixed $\nu=5$. The variance correction matters because an ordinary Student t random variable has variance $\nu/(\nu-2)$. Neither the decay nor the degrees of freedom is selected using held-out outcomes.

Exception frequency is compared with 1% using an exact binomial test, whose null assumes independent identically distributed exceptions. Transition counts also support a first-order independence likelihood-ratio diagnostic. Empty counts contribute zero to the log likelihood. Tables with any cell below five are explicitly marked sparse; their asymptotic chi-square approximation can be unreliable. Multiple tail diagnostics have no familywise adjustment and should remain exploratory. See [Christoffersen, Evaluating Interval Forecasts](https://users.ssc.wisc.edu/~bhansen/718/Christoffersen1998.pdf).

## Attribution and implementation costs

Exact repricing is compared with $e_t^T\Delta y_t$ and the quadratic correction

$$\mathrm{P\&L}_{bp}^{(2)}=e_t^T\Delta y_{bp}
+\frac{1}{2\cdot10^4}\sum_j H_{t,jj}(\Delta y_{bp,j})^2.$$

The Hessian is evaluated for the current discounted target and hedge faces. The first-order P&L is also split into the top three pre-shock EWMA directions and the orthogonal complement. These are additive projections, not independent risk contributions.

At an annual refit, traded hedge market value is $\sum_i|f_i^{new}-f_i^{old}|P_i(y_t)$. Multiplying its ratio to target calibration value by an assumed 2 bp gives implementation cost in basis points. Initial hedge establishment is included. Target rescaling, liquidation at the experiment endpoint, financing and daily margin maintenance are excluded. The cost-adjusted column is a scenario sensitivity on the shock accounting, not a tradable net-return record.

## Statistical comparisons

For candidate $m$, the paired loss difference is $D_t=(\mathrm{P\&L}_{m,t})^2-(\mathrm{P\&L}_{DV01,t})^2$. A negative mean favors the candidate. With centered differences, the Newey-West estimate is

$$\widehat\Omega=\widehat\gamma_0+2\sum_{j=1}^{10}
\left(1-\frac{j}{11}\right)\widehat\gamma_j,
\qquad \mathrm{SE}(\bar D)=\sqrt{\widehat\Omega/T}.$$

Two-sided normal-approximation p-values receive Holm adjustment across the three baseline comparisons. The reported inference conditions on the specified walk-forward procedure; changing regimes and heavy tails can weaken the approximation. The HAC construction follows [Newey and West, A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix](https://www.nber.org/papers/t0055).

Circular paired block bootstraps preserve candidate/baseline alignment. Blocks of 5, 10 and 20 observed sessions show sensitivity to the dependence choice. The percentile intervals describe RMS reduction with the fitted historical paths held fixed. They do not resample the entire model-selection procedure or correct all reported intervals simultaneously.

The 99% empirical expected shortfall averages exactly the largest 15 losses among 1,500 holdout observations. Full outputs preserve date-level losses, validation trials, solver residuals, package versions, SHA-256 checksums and the assumptions above.
