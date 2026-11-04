args<-commandArgs(trailingOnly=TRUE); stopifnot(length(args)==2L)
d<-read.csv(args[1],check.names=FALSE,na.strings=c("","NA"))
features<-c("mean_map","min_map","sd_map","auc_per_hour_lt_65","pct_time_lt_65","early_map","middle_map","late_map","slope_norm","recovery_from_nadir","episode_count_lt_65","longest_episode_min_lt_65")
stopifnot(nrow(d)>0,!anyNA(d$subjectid),!anyNA(d$trajectory_cluster),all(complete.cases(d[,features])))
if (!"age" %in% names(d) || !is.numeric(d$age) || any(!is.finite(d$age)) || any(d$age < 18)) stop("Adult cohort requires finite age >= 18 for every case")
X<-as.matrix(d[,features]); ids<-unique(d$subjectid); groups<-split(seq_len(nrow(d)),d$subjectid)
ari<-function(a,b){tt<-table(a,b);c2<-function(x)x*(x-1)/2; s<-sum(c2(tt));u<-sum(c2(rowSums(tt)));v<-sum(c2(colSums(tt)));n<-c2(sum(tt));(s-u*v/n)/((u+v)/2-u*v/n)}
set.seed(20261102)
out<-vector("list",100)
for(i in 1:100){
  chosen<-sample(ids,length(ids),replace=TRUE);ix<-unlist(groups[as.character(chosen)],use.names=FALSE)
  train<-X[ix,,drop=FALSE];mu<-colMeans(train);sdv<-apply(train,2,sd);if(any(!is.finite(sdv)|sdv==0)) stop("invalid training standard deviation")
  z<-sweep(sweep(train,2,mu),2,sdv,"/");zall<-sweep(sweep(X,2,mu),2,sdv,"/")
  issue <- character()
  fit<-withCallingHandlers(try(kmeans(z,centers=4,nstart=10,iter.max=100),silent=TRUE), warning=function(w) {issue <<- c(issue,conditionMessage(w));invokeRestart("muffleWarning")})
  if(inherits(fit,"try-error")) {out[[i]]<-data.frame(rep=i,status="failed",ari=NA,min_cluster=NA,max_cluster=NA,warning=paste(issue,collapse="; "));next}
  dst<-sapply(1:4,function(k)rowSums((zall-matrix(fit$centers[k,],nrow(zall),ncol(zall),byrow=TRUE))^2))
  lab<-max.col(-dst,ties.method="first");sz<-tabulate(lab,4)
  out[[i]]<-data.frame(rep=i,status=if(length(issue)) "warning" else "ok",ari=ari(d$trajectory_cluster,lab),min_cluster=min(sz),max_cluster=max(sz),warning=paste(issue,collapse="; "))
  if(i%%10==0)cat("bootstrap",i,"/ 100\n")
}
out<-do.call(rbind,out);write.csv(out,file.path(args[2],"cluster_bootstrap_subject.csv"),row.names=FALSE)
cat("no warning",sum(out$status=="ok"),"warnings",sum(out$status=="warning"),"failed",sum(out$status=="failed"),"ARI median",median(out$ari,na.rm=TRUE),"quantiles",quantile(out$ari,c(.025,.975),na.rm=TRUE),"\n")
