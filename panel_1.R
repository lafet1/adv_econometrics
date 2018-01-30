library(foreign)
library(plm)
library(dummies)
library(dplyr)
library(tibble)
library(stringr)

gravity <- read.dta("gravity.dta")

# Exercise 1
regr1 <- lm(gravity$ltrade ~ gravity$fta)


# Exercise 2
regr2 <- lm(ltrade ~ fta + lgdp + ldist, data = gravity)


# Exercise 3
dummies3 <- dummy(gravity$pairid)[, - 1]
regr3 <- lm(ltrade ~ fta + lgdp + dummies3, data = gravity)


# Exercise 4
mean_ltrade <- gravity %>% select(pairid, ltrade) %>% 
  group_by(pairid) %>% summarise_all(funs(mean))
mean_lgdp <- gravity %>% select(pairid, lgdp) %>% 
  group_by(pairid) %>% summarise_all(funs(mean))
mean_ldist <- gravity %>% select(pairid, ldist) %>% 
  group_by(pairid) %>% summarise_all(funs(mean))
mean_fta <- gravity %>% select(pairid, fta) %>% 
  group_by(pairid) %>% summarise_all(funs(mean))

means_4 <- bind_cols(mean_ltrade, mean_lgdp %>% select(lgdp), mean_ldist %>% select(ldist),
                  mean_fta %>% select(fta))

gravity_4 <- left_join(gravity, means_4, by = "pairid") %>%
  mutate(new_ltrade = ltrade.x - ltrade.y,
         new_lgdp = lgdp.x - lgdp.y,
         new_ldist = ldist.x - ldist.y,
         new_fta = fta.x - fta.y)
regr4 <- lm(new_ltrade ~ new_fta + new_lgdp + new_ldist, data = gravity_4)


# Exercise 5
gravity_5 <- pdata.frame(gravity, index = c('pairid', 'year'))
regr5 <- plm(ltrade ~ fta + lgdp, data = gravity_5, model = "within", effect = "individual")


# Exercise 6
dummies6 <- dummy(gravity$year)[, - 1]
regr6 <- lm(ltrade ~ fta + lgdp + dummies6 + dummies3, data = gravity)


# Exercise 7
regr7 <- plm(ltrade ~ fta + lgdp + dummies6, data = gravity_5, 
             model = "within", effect = "individual")


# Exercise 8
summary(regr7, vcov = function(x) vcovHC(x, cluster = 'group'))


# Exercise 9
res_9 <- regr7$residuals
plot(res_9[str_detect(names(regr7$residuals), '^3-')], type = "l")
plot(res_9[str_detect(names(regr7$residuals), '^4-')], type = "l")
plot(res_9[str_detect(names(regr7$residuals), '^5-')], type = "l")

gravity_9_0 <- as.tibble(add_column(gravity, res_9) %>% filter(emu == 0) %>% 
                           select(year, res_9)) %>% group_by(year) %>% summarise(mean(res_9))
gravity_9_1 <- as.tibble(add_column(gravity, res_9) %>% filter(emu == 1) %>% 
                           select(year, res_9)) %>% group_by(year) %>% summarise(mean(res_9))

# Exercise 10
t <- matrix(0, nrow = 6156, ncol = 170)
for (i in 2:171) {
  t[(1 + (i - 1) * 36):(i * 36), (i - 1)] <- unique(gravity$year)
}

regr10 <- plm(ltrade ~ fta + lgdp + dummies6 + t, data = gravity_5, 
             model = "within", effect = "individual")


# Exercise 11
res_11 <- regr10$residuals
plot(res_11[str_detect(names(regr10$residuals), '^3-')], type = "l")
plot(res_11[str_detect(names(regr10$residuals), '^4-')], type = "l")
plot(res_11[str_detect(names(regr10$residuals), '^5-')], type = "l")

gravity_11_0 <- as.tibble(add_column(gravity, res_11) %>% filter(emu == 0) %>% 
                           select(year, res_11)) %>% group_by(year) %>% summarise(mean(res_11))
gravity_11_1 <- as.tibble(add_column(gravity, res_11) %>% filter(emu == 1) %>% 
                           select(year, res_11)) %>% group_by(year) %>% summarise(mean(res_11))


# Exercise 12

# check serial correlation again, basically justa a bit of annoying wrangling


# Exercise 13
summary(regr10, vcov = vcovDC)



