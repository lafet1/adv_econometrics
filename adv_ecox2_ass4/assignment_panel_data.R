library(foreign)
library(plm)
library(tidyverse)
options(warn=-1)


fig <- read.dta('FIG.dta')


#################
###### OLS ######
#################

fig_panel <- pdata.frame(fig, index = c("country", 'period'))
formula_plm_core <- as.formula(growth ~ privo
                               + log(initial) 
                               + privo * log(initial))
formula_plm_control <- as.formula(growth ~ privo
                                  + privo * log(initial)
                                  + school + gov + log1p(pi) 
                                  + log1p(bmp) + sec)

regr_core <- plm(formula_plm_core, data = fig_panel, effect = 'twoways')
summary(regr_core, vcov = vcovHC)

regr_full <- plm(formula_plm_control, data = fig_panel, effect = 'twoways')
summary(regr_full, vcov = vcovHC)


#################
###### GMM ######
#################

formula_gpmm_core <- as.formula(growth ~ plm::lag(growth, 1:2) + privo
                               + log(initial) 
                               + privo * log(initial)
                               | plm::lag(growth, 3:5) + plm::lag(privo, 3:5))
formula_gpmm_full <- as.formula(growth ~ plm::lag(growth, 1:2) + privo
                                + privo * log(initial)
                                + school + gov + log1p(pi) 
                                + log1p(bmp) + sec
                                | plm::lag(growth, 3:5) + plm::lag(privo, 3:5))

gmm_core <- pgmm(formula_gpmm_core, data = fig_panel, transformation = 'd', 
                 model = 'twosteps', robust = T)
summary(gmm_core)
gmm_full <- pgmm(formula_gpmm_full, data = fig_panel, transformation = 'd', 
                 model = 'twosteps', robust = T)
summary(gmm_full)


# all model summaries
summary(regr_core, vcov = vcovHC)
summary(regr_full, vcov = vcovHC)
summary(gmm_core)
summary(gmm_full)


