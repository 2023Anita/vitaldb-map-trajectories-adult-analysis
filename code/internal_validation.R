args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) %in% c(2L, 3L))
seed <- if (length(args) == 3L) as.integer(args[3]) else 20261102L
stopifnot(is.finite(seed))
suffix <- if (length(args) == 3L) paste0("_", seed) else ""
d <- read.csv(args[1], check.names = FALSE, na.strings = c("", "NA"))
stopifnot(nrow(d) > 0L, !anyDuplicated(d$caseid), !anyNA(d$subjectid), !anyNA(d$icu_days), all(d$icu_use == as.integer(d$icu_days > 0)))
if (!"age" %in% names(d) || !is.numeric(d$age) || any(!is.finite(d$age)) || any(d$age < 18)) stop("Adult cohort requires finite age >= 18 for every case")
for (n in grep("^dept_", names(d), value = TRUE)) d[[n]] <- as.integer(d[[n]] %in% c(TRUE, "True", "TRUE", "1", 1))
names(d) <- make.names(names(d), unique = TRUE)
base <- c("age", "male", "bmi", "asa", "emop", "op_duration_min", "preop_hb", "preop_cr", "preop_alb", "source_invasive", grep("^dept_", names(d), value=TRUE))
traditional <- c("minutes_map_lt_65", "twa_map_lt_65")
features <- c("mean_map", "min_map", "sd_map", "auc_per_hour_lt_65", "pct_time_lt_65", "early_map", "middle_map", "late_map", "slope_norm", "recovery_from_nadir", "episode_count_lt_65", "longest_episode_min_lt_65")
free <- c("mean_map", "min_map", "sd_map", "early_map", "middle_map", "late_map", "slope_norm", "recovery_from_nadir")
stopifnot(all(complete.cases(d[, features])), all(complete.cases(d[, traditional])))
set.seed(seed)
ids <- sample(unique(d$subjectid))
idfold <- setNames(rep(seq_len(5), length.out=length(ids)), ids)
d$fold <- unname(idfold[as.character(d$subjectid)])
auc <- function(y, p) { r <- rank(p, ties.method="average"); pos <- sum(y==1); neg <- sum(y==0); if (!pos || !neg) return(NA_real_); (sum(r[y==1])-pos*(pos+1)/2)/(pos*neg) }
models <- c("base", "traditional", "phenotype", "combined", "threshold_free_combined")
preds <- list(); warnings <- character()
for (fold in 1:5) {
  tr <- d[d$fold != fold, ]; te <- d[d$fold == fold, ]
  stopifnot(!length(intersect(tr$subjectid, te$subjectid)))
  for (n in base) { tr[[n]] <- as.numeric(tr[[n]]); te[[n]] <- as.numeric(te[[n]]); mu <- mean(tr[[n]], na.rm=TRUE); stopifnot(is.finite(mu)); tr[[n]][is.na(tr[[n]])] <- mu; te[[n]][is.na(te[[n]])] <- mu }
  add_clusters <- function(tr, te, cols, label) {
    x <- as.matrix(tr[, cols]); z <- as.matrix(te[, cols]); mu <- colMeans(x); s <- apply(x, 2, sd); s[s == 0] <- 1
    x <- sweep(sweep(x, 2, mu), 2, s, "/"); z <- sweep(sweep(z, 2, mu), 2, s, "/")
    set.seed(seed + fold); km <- kmeans(x, centers=4, nstart=10, iter.max=100)
    dist2 <- sapply(seq_len(4), function(i) rowSums((z - matrix(km$centers[i, ], nrow(z), length(cols), byrow=TRUE))^2))
    tr[[label]] <- factor(km$cluster, levels=1:4); te[[label]] <- factor(max.col(-dist2, ties.method="first"), levels=1:4)
    list(tr=tr, te=te)
  }
  obj <- add_clusters(tr, te, features, "cluster_main"); tr <- obj$tr; te <- obj$te
  obj <- add_clusters(tr, te, free, "cluster_free"); tr <- obj$tr; te <- obj$te
  specs <- list(base=base, traditional=c(base,traditional), phenotype=c(base,"cluster_main"), combined=c(base,traditional,"cluster_main"), threshold_free_combined=c(base,traditional,"cluster_free"))
  for (m in models) {
    fit <- withCallingHandlers(glm(reformulate(specs[[m]],response="icu_use"), data=tr, family=binomial(), control=glm.control(maxit=100)), warning=function(w) { warnings <<- c(warnings, paste("fold", fold, m, conditionMessage(w))); invokeRestart("muffleWarning") })
    if (!fit$converged || anyNA(coef(fit))) stop(paste("model convergence or alias failure", fold, m))
    p <- as.numeric(predict(fit, newdata=te, type="response")); stopifnot(all(is.finite(p)))
    preds[[length(preds)+1]] <- data.frame(caseid=te$caseid, subjectid=te$subjectid, fold=fold, y=te$icu_use, model=m, p=p)
  }
  cat("completed fold",fold,"\n")
}
p <- do.call(rbind,preds)
metrics <- do.call(rbind,lapply(split(p,p$model),function(x) {
  q <- pmin(pmax(x$p, 1e-6), 1-1e-6); cal <- glm(x$y ~ qlogis(q),family=binomial())
  data.frame(model=x$model[1], n=nrow(x), events=sum(x$y), auroc=auc(x$y,x$p), brier=mean((x$y-x$p)^2), calibration_intercept=unname(coef(cal)[1]), calibration_slope=unname(coef(cal)[2]))
}))
write.csv(p, file.path(args[2],paste0("cv_predictions",suffix,".csv")), row.names=FALSE)
write.csv(metrics, file.path(args[2],paste0("cv_metrics",suffix,".csv")), row.names=FALSE)
writeLines(if(length(warnings)) warnings else "none", file.path(args[2],paste0("cv_warnings",suffix,".txt")))
print(metrics)
