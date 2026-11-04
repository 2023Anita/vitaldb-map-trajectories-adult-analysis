args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
df <- read.csv(args[1], check.names = FALSE, na.strings = c("", "NA"))
stopifnot(nrow(df) > 0L, !anyDuplicated(df$caseid), all(df$icu_use %in% c(0, 1)), all(!is.na(df$icu_days)), all(df$icu_use == as.integer(df$icu_days > 0)))
if (!"age" %in% names(df) || !is.numeric(df$age) || any(!is.finite(df$age)) || any(df$age < 18)) stop("Adult cohort requires finite age >= 18 for every case")
stopifnot(!anyNA(df$subjectid))
raw_df <- df
base <- c("age", "male", "bmi", "asa", "emop", "op_duration_min", "preop_hb", "preop_cr", "preop_alb", "source_invasive", grep("^dept_", names(df), value = TRUE))
for (name in grep("^dept_", names(df), value = TRUE)) df[[name]] <- as.integer(df[[name]] %in% c(TRUE, "True", "TRUE", "1", 1))
names(df) <- make.names(names(df), unique = TRUE)
base <- make.names(base, unique = TRUE)
for (name in base) {
  x <- as.numeric(df[[name]])
  if (anyNA(x)) x[is.na(x)] <- mean(x, na.rm = TRUE)
  stopifnot(all(is.finite(x)))
  df[[name]] <- x
}
df$trajectory_phenotype <- relevel(factor(df$trajectory_phenotype), ref = "preserved_high_map")
formula <- reformulate(c(base, "trajectory_phenotype"), response = "icu_use")
fit <- glm(formula, data = df, family = binomial(), control = glm.control(maxit = 100))
if (!fit$converged || anyNA(coef(fit))) stop("Model failed convergence or has aliased coefficients")
tab <- summary(fit)$coefficients
sel <- grep("^trajectory_phenotype", rownames(tab))
out <- data.frame(term = rownames(tab)[sel], beta = tab[sel, 1], se = tab[sel, 2], adjusted_or = exp(tab[sel, 1]), ci_low_wald = exp(tab[sel, 1] - qnorm(.975)*tab[sel, 2]), ci_high_wald = exp(tab[sel, 1] + qnorm(.975)*tab[sel, 2]), n = nrow(df), events = sum(df$icu_use), converged = fit$converged, iterations = fit$iter)
set.seed(20261102)
subjects <- unique(df$subjectid)
groups <- split(seq_len(nrow(df)), df$subjectid)
boot <- matrix(NA_real_, nrow=300, ncol=nrow(out), dimnames=list(NULL, out$term))
for (i in seq_len(nrow(boot))) {
  chosen <- sample(subjects, length(subjects), replace=TRUE)
  ix <- unlist(groups[as.character(chosen)], use.names=FALSE)
  b <- raw_df[ix, , drop=FALSE]
  for (n in grep("^dept_", names(b), value=TRUE)) b[[n]] <- as.integer(b[[n]] %in% c(TRUE,"True","TRUE","1",1))
  names(b) <- make.names(names(b), unique=TRUE)
  for (n in base) { b[[n]] <- as.numeric(b[[n]]); b[[n]][is.na(b[[n]])] <- mean(b[[n]], na.rm=TRUE) }
  b$trajectory_phenotype <- factor(b$trajectory_phenotype, levels=levels(df$trajectory_phenotype))
  bf <- suppressWarnings(try(glm(formula, data=b, family=binomial(), control=glm.control(maxit=100)), silent=TRUE))
  if (!inherits(bf, "try-error") && bf$converged && all(is.finite(coef(bf)[out$term]))) boot[i, ] <- exp(coef(bf)[out$term])
}
valid <- complete.cases(boot)
if (sum(valid) < 270) stop("Insufficient successful subject-level bootstrap samples")
out$ci_low_subject_bootstrap <- apply(boot[valid, , drop=FALSE], 2, quantile, probs=.025)
out$ci_high_subject_bootstrap <- apply(boot[valid, , drop=FALSE], 2, quantile, probs=.975)
out$bootstrap_success <- sum(valid)
write.csv(out, file.path(args[2], "or_glm_crosscheck.csv"), row.names = FALSE)
cat("n=", nrow(df), "events=", sum(df$icu_use), "converged=", fit$converged, "iterations=", fit$iter, "\n")
print(out)
