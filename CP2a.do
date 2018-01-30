
/// (1)
regress  ltrade fta

** Having a FTA increases trade between partner countries with 100*0.60 = 60%. It is unlikely that we can interpret the estimated coefficient as the causal effect
** of a free trade agreement (FTA) on trade. A potential threat to the internal validity
** is omitted variable bias. For example international trade flows depend heavily on size of
** and distance between trading partners.

/// (2)
regress  ltrade  fta  lgdp  ldist

** The estimated coefficient is positive and therefore suggests that a free trade agreement increases 
** international trade. The FTA coefficient changed compared to the result in (1)
** because we have added variables (log of gdp and distance) that are correlated with free trade agreements and that affect
** international trade. These variables were omitted variables in the simple regression in (1),
** causing positive bias. However, the OLS estimator of the coefficient in the mutiple regression including
** these control variables might still be biased because of other, unobserved omitted variables.

/// (3)
tab  pairid, gen(PAIRID)
drop PAIRID1
regress  ltrade fta lgdp PAIRID*

** The coefficient on the FTA dummy is now somewhat larger than in (2). 
** The reason is that we now include country-pair fixed effects that control for all 
** observed and unobserved variables that differ between country-pairs but not within a country-pair over time.
** Some country-pair characterists might be related to free trade agreements and affect international trade.
** These characteristics were omitted variables in the regression in (2) but are no longer
** causing omitted variable bias in the regression that includes entity fixed effects. 

//NOTE//
** With so many coefficients to be estimated, your get long tables with regression output.
** In panel data specifications most of these coefficients are fixed effects, and you are not particularly interested in their actual values.
** The stata package 'estout' (introduced in the first practicum) facilitates printing only key regression results.
** Below is an example for the specification (3), but this can be used for all other models too.

** first run (3) 'quietly', so not immediately showing the regression output
quietly regress  ltrade fta lgdp PAIRID* 
** next store the regression output in 'spec3'
est store spec3
** finally, show key estimation results (b and se) for fta and lgdp regressors only
esttab spec3, b se keep(fta lgdp)

/// (4)
bys pairid: egen Mltrade=mean(ltrade)
bys pairid: egen Mlgdp=mean(lgdp)
bys pairid: egen Mldist=mean(ldist)
bys pairid: egen Mfta=mean(fta)

gen Dltrade=ltrade-Mltrade
gen Dlgdp=lgdp-Mlgdp
gen Dldist=ldist-Mldist
gen Dfta=fta-Mfta

regress  Dltrade Dfta Dlgdp Dldist

** The estimated coefficients are identical to those in (3). The LSDV method and within estimation 
** are two ways to estimate the same fixed effects panel data model and give identical estimates.
** Note that Stata recognizes perfect multicollinearity between the entity fixed effects and distance, and dropped distance from the model.
** In general, coefficients of time invariant regressors are not identified by the LSDV/within estimator.

/// (5)
xtset pairid year
xtreg  ltrade fta lgdp, fe 

** The estimated coefficients are identical to those in (3) and (4). The LSDV method 
** and within estimation (either done in two steps or by using the xtreg command) are two ways to estimate the 
** same fixed effects panel data model and give identical FTA coefficients.

/// (6)
tab year, gen(YEAR)
drop YEAR1
regress  ltrade fta lgdp PAIRID* YEAR*

** The estimated FTA coefficient hardly changed compared with (3).
** Apparently year fixed effects, capturing all variables that vary over time
** but not across country-pairs, are not correlated with free trade agreements. 

/// (7)
xtreg  ltrade fta lgdp YEAR*, fe

** The estimated FTA coefficient is identical to the estimate obtained in (6)
** The LSDV method with dummies for states & years is identical to within estimation + time dummies.

/// (8)
xtreg  ltrade fta lgdp YEAR*, fe cluster(pairid) 

** The estimated coefficients are identical to the estimates obtained in (7) but the
** estimated standard errors are larger. The reason is that the standard errors in (7)
** did not account for serial correlation and are therefore incorrect, while the estimated 
** standard errors using the cluster command are robust for any type of serial correlation
** within the times series of a country-pair.

/// (9)
predict res, e
tsline res if pairid == 3
tsline res if pairid == 4
tsline res if pairid == 5
egen res0 = mean(res) if emu == 0, by(year)
egen res1 = mean(res) if emu == 1, by(year)
tsline res0 res1

** The line graph of residuals averaged over EMU country-pairs shows an upward trend,
** while averaged residuals over non-EMU country-pairs are trending downwards.
** This shows that remaining time- and cross-section unobserved heterogeneity in trade 
** is trending over time. This may cause omitted variable bias in the estimation results,
** even after accounting for entity and year specific fixed effects.
** Also note that using clustered standard errors as in (8) are not an effective solution in this case.

/// (10)
forvalues i = 2/171{
gen t`i' = year*PAIRID`i'
}
xtreg ltrade fta lgdp YEAR* t*, fe 

** Applying the incidental trend model leads to
** a much smaller FTA effect compared with all previous estimates.

/// (11)
drop res res0 res1
predict res, e
egen res0 = mean(res) if emu == 0, by(year)
egen res1 = mean(res) if emu == 1, by(year)
tsline res0 res1

** Contrary to the residuals of (8), the line graphs of averaged residuals across EMU and non-EMU country-pairs do not show trends,
** suggesting that the incidental trend specification (10) has been effective.

/// (12)

regress res l.res

** However, there is still significant residual autocorrelation left in (10), even after accounting for incidental trends.
** The regression of the residuals on their own lagged values shows a significant coefficient indicating that residuals are autocorrelated.

/// (13) 

xtreg ltrade fta lgdp YEAR* t*, fe cluster(pairid) 

** If we believe that this remaining autocorrelation is not causing omitted variables bias, we can exploit clustered standard errors.   
** The clustered standard errors are substantially larger than the uncorrected counterparts.
** In this final specification the FTA effect on bilateral trade is statistically significant only at the 10% level.
** And with an estimated impact of only 4% its economic significance is moderate too. 



