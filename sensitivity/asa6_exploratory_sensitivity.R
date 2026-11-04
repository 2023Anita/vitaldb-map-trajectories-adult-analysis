args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L, dir.exists(args[2]))
df <- read.csv(args[1], check.names=FALSE, na.strings=c('', 'NA'))
stopifnot(nrow(df) == 5501, is.numeric(df$age), all(is.finite(df$age)), all(df$age >= 18), sum(df$asa == 6, na.rm=TRUE) == 11)
target <- file.path(args[2], 'asa6_exploratory_sensitivity.csv')
if (file.exists(target)) stop('Output already exists')
base <- c('age', 'male', 'bmi', 'asa', 'emop', 'op_duration_min', 'preop_hb', 'preop_cr', 'preop_alb', 'source_invasive', grep('^dept_', names(df), value=TRUE))
fit_scenario <- function(x, label) {
  for (name in grep('^dept_', names(x), value=TRUE)) x[[name]] <- as.integer(x[[name]] %in% c(TRUE, 'True', 'TRUE', '1', 1))
  names(x) <- make.names(names(x), unique=TRUE)
  variables <- make.names(base, unique=TRUE)
  for (name in variables) {
    x[[name]] <- as.numeric(x[[name]])
    x[[name]][is.na(x[[name]])] <- mean(x[[name]], na.rm=TRUE)
    stopifnot(all(is.finite(x[[name]])))
  }
  x$trajectory_phenotype <- relevel(factor(x$trajectory_phenotype), ref='preserved_high_map')
  fit <- glm(reformulate(c(variables, 'trajectory_phenotype'), response='icu_use'), data=x, family=binomial(), control=glm.control(maxit=100))
  stopifnot(fit$converged, !anyNA(coef(fit)))
  estimates <- summary(fit)$coefficients
  ids <- grep('^trajectory_phenotype', rownames(estimates))
  data.frame(scenario=label, n=nrow(x), subjects=length(unique(x$subjectid)), events=sum(x$icu_use),
             term=rownames(estimates)[ids], or=exp(estimates[ids, 1]),
             wald_low=exp(estimates[ids, 1]-qnorm(.975)*estimates[ids, 2]),
             wald_high=exp(estimates[ids, 1]+qnorm(.975)*estimates[ids, 2]), row.names=NULL)
}
out <- rbind(fit_scenario(df, 'original labels, all adult cases'), fit_scenario(df[is.na(df$asa) | df$asa != 6, ], 'original labels, omit recorded ASA 6'))
write.csv(out, target, row.names=FALSE)
print(out)
