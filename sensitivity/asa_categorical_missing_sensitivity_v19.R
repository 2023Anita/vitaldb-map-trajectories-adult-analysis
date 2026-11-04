args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L, !file.exists(args[2]))
d <- read.csv(args[1], check.names = FALSE, na.strings = c('', 'NA'))
stopifnot(nrow(d) == 5501L, !anyDuplicated(d$caseid), !anyNA(d$subjectid),
          all(is.finite(d$age)), all(d$age >= 18),
          all(d$icu_use == as.integer(d$icu_days > 0)))
for (n in grep('^dept_', names(d), value = TRUE)) {
  d[[n]] <- as.integer(d[[n]] %in% c(TRUE, 'True', 'TRUE', '1', 1))
}
names(d) <- make.names(names(d), unique = TRUE)
base <- c('age', 'male', 'bmi', 'emop', 'op_duration_min', 'preop_hb', 'preop_cr',
          'preop_alb', 'source_invasive', grep('^dept_', names(d), value = TRUE))
for (n in base) {
  d[[n]] <- as.numeric(d[[n]])
  mu <- mean(d[[n]], na.rm = TRUE)
  stopifnot(is.finite(mu))
  d[[n]][is.na(d[[n]])] <- mu
  stopifnot(all(is.finite(d[[n]])))
}
stopifnot(all(is.na(d$asa) | d$asa %in% c(1, 2, 3, 4, 6)))
d$asa_cat <- factor(ifelse(is.na(d$asa), 'missing', as.character(d$asa)),
                    levels = c('1', '2', '3', '4', '6', 'missing'))
d$trajectory_phenotype <- relevel(factor(d$trajectory_phenotype), ref = 'preserved_high_map')
warnings <- character()
fit <- withCallingHandlers(
  glm(reformulate(c(base, 'asa_cat', 'trajectory_phenotype'), response = 'icu_use'),
      data = d, family = binomial(), control = glm.control(maxit = 100)),
  warning = function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart('muffleWarning') }
)
stopifnot(fit$converged, !anyNA(coef(fit)))
co <- summary(fit)$coefficients
ix <- grep('^trajectory_phenotype', rownames(co))
out <- data.frame(scenario = 'original clusters, categorical ASA with missing category',
                  n = nrow(d), subjects = length(unique(d$subjectid)), events = sum(d$icu_use),
                  asa_missing = sum(is.na(d$asa)), term = rownames(co)[ix],
                  adjusted_or = exp(co[ix, 1]), wald_low = exp(co[ix, 1] - qnorm(.975) * co[ix, 2]),
                  wald_high = exp(co[ix, 1] + qnorm(.975) * co[ix, 2]),
                  warnings = if (length(warnings)) paste(warnings, collapse = '; ') else 'none',
                  row.names = NULL)
write.csv(out, args[2], row.names = FALSE)
print(out)
