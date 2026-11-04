args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L)
d <- read.csv(args[1],check.names=FALSE,na.strings=c("","NA"))
stopifnot(nrow(d)>0L, !anyNA(d$icu_days), !anyNA(d$trajectory_cluster))
if (!"age" %in% names(d) || !is.numeric(d$age) || any(!is.finite(d$age)) || any(d$age < 18)) stop("Adult cohort requires finite age >= 18 for every case")
free <- c("mean_map","min_map","sd_map","early_map","middle_map","late_map","slope_norm","recovery_from_nadir")
allfeatures <- c("mean_map","min_map","sd_map","auc_per_hour_lt_65","pct_time_lt_65","early_map","middle_map","late_map","slope_norm","recovery_from_nadir","episode_count_lt_65","longest_episode_min_lt_65")
ari <- function(a,b) { t<-table(a,b); c2<-function(x) x*(x-1)/2; s<-sum(c2(t)); x<-sum(c2(rowSums(t))); y<-sum(c2(colSums(t))); n<-c2(sum(t)); (s-x*y/n)/((x+y)/2-x*y/n) }
set.seed(20261102)
kfree <- kmeans(scale(d[,free]),centers=4,nstart=20,iter.max=100)
results <- data.frame(comparison="original_vs_threshold_free_kmeans", n=nrow(d), ari=ari(d$trajectory_cluster,kfree$cluster))
if (requireNamespace("cluster",quietly=TRUE)) {
  set.seed(20261102)
  ca <- cluster::clara(scale(d[,allfeatures]),k=4,samples=10,sampsize=300,metric="euclidean")
  results <- rbind(results,data.frame(comparison="original_vs_clara_medoids",n=nrow(d),ari=ari(d$trajectory_cluster,ca$clustering)))
}
write.csv(results,file.path(args[2],"sensitivity_cluster_ari.csv"),row.names=FALSE)
x <- d[d$source_invasive==1,]
for (name in grep("^dept_", names(x), value=TRUE)) x[[name]] <- as.integer(x[[name]] %in% c(TRUE,"True","TRUE","1",1))
names(x)<-make.names(names(x),unique=TRUE)
base <- c("age","male","bmi","asa","emop","op_duration_min","preop_hb","preop_cr","preop_alb",grep("^dept_",names(x),value=TRUE))
for (name in base) {x[[name]]<-as.numeric(x[[name]]); x[[name]][is.na(x[[name]])]<-mean(x[[name]],na.rm=TRUE)}
x$trajectory_phenotype<-relevel(factor(x$trajectory_phenotype),ref="preserved_high_map")
fit<-glm(reformulate(c(base,"trajectory_phenotype"),response="icu_use"), data=x,family=binomial(),control=glm.control(maxit=100))
if(!fit$converged || anyNA(coef(fit))) stop("invasive model failed")
t<-summary(fit)$coefficients; ix<-grep("^trajectory_phenotype",rownames(t)); o<-data.frame(term=rownames(t)[ix],adjusted_or=exp(t[ix,1]),ci_low_wald=exp(t[ix,1]-qnorm(.975)*t[ix,2]),ci_high_wald=exp(t[ix,1]+qnorm(.975)*t[ix,2]),n=nrow(x),events=sum(x$icu_use))
write.csv(o,file.path(args[2],"sensitivity_invasive_or.csv"),row.names=FALSE)
print(results); print(o)
