args <- commandArgs(trailingOnly=TRUE); stopifnot(length(args)==3L, !file.exists(args[3]))
d<-read.csv(args[1],check.names=FALSE,na.strings=c("","NA")); p<-read.csv(args[2]); folds<-unique(p[,c("caseid","fold")]); stopifnot(nrow(folds)==nrow(d),!anyDuplicated(folds$caseid))
stopifnot(nrow(d)==5501L,!anyDuplicated(d$caseid),is.numeric(d$age),all(is.finite(d$age)),all(d$age>=18))
d<-merge(d,folds,by="caseid",sort=FALSE); stopifnot(nrow(d)==5501L)
for(n in grep("^dept_",names(d),value=TRUE)) d[[n]]<-as.integer(d[[n]] %in% c(TRUE,"True","TRUE","1",1))
names(d)<-make.names(names(d),unique=TRUE)
base<-c("age","male","bmi","asa","emop","op_duration_min","preop_hb","preop_cr","preop_alb","source_invasive",grep("^dept_",names(d),value=TRUE))
specs<-list(base=base,base_without_source=setdiff(base,"source_invasive"),source_only="source_invasive")
auc<-function(y,p){r<-rank(p,ties.method="average");n<-sum(y==1);m<-sum(y==0);(sum(r[y==1])-n*(n+1)/2)/(n*m)}
pred<-list()
for(f in 1:5){tr<-d[d$fold!=f,];te<-d[d$fold==f,];stopifnot(!length(intersect(tr$subjectid,te$subjectid)));for(n in base){tr[[n]]<-as.numeric(tr[[n]]);te[[n]]<-as.numeric(te[[n]]);mu<-mean(tr[[n]],na.rm=TRUE);tr[[n]][is.na(tr[[n]])]<-mu;te[[n]][is.na(te[[n]])]<-mu};for(m in names(specs)){fit<-glm(reformulate(specs[[m]],response="icu_use"),data=tr,family=binomial(),control=glm.control(maxit=100));stopifnot(fit$converged,!anyNA(coef(fit)));pred[[length(pred)+1]]<-data.frame(caseid=te$caseid,y=te$icu_use,model=m,p=as.numeric(predict(fit,newdata=te,type="response")))}}
q<-do.call(rbind,pred); out<-do.call(rbind,lapply(split(q,q$model),function(z)data.frame(model=z$model[1],n=nrow(z),auroc=auc(z$y,z$p),brier=mean((z$y-z$p)^2))))
write.csv(out,args[3],row.names=FALSE);print(out)
