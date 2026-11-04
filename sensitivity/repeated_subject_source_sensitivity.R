args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L, dir.exists(args[2]))
paths <- file.path(args[2],c('source_phenotype_icu_working.csv','single_record_subject_or_working.csv'))
if(any(file.exists(paths))) stop('Refusing to replace existing output')
d <- read.csv(args[1],check.names=FALSE,na.strings=c('','NA'))
stopifnot(nrow(d)==5501, !anyDuplicated(d$caseid), !anyNA(d$subjectid), is.numeric(d$age), all(is.finite(d$age)), all(d$age>=18), all(d$icu_use==as.integer(d$icu_days>0)))
levels <- c('preserved_high_map','moderate_low_burden_or_variable_map','early_low_map_with_recovery','sustained_severe_hypotension_burden')
stopifnot(setequal(unique(d$trajectory_phenotype),levels),setequal(unique(d$map_source_type),c('invasive','nibp')))
source_table <- do.call(rbind,lapply(c('invasive','nibp'),function(source) do.call(rbind,lapply(levels,function(p){
  x <- d[d$map_source_type==source & d$trajectory_phenotype==p,]
  data.frame(source=source, phenotype=p, cases=nrow(x), subjects=length(unique(x$subjectid)), events=sum(x$icu_use), event_fraction=mean(x$icu_use))
}))))
counts <- table(d$subjectid)
single <- d[as.integer(counts[as.character(d$subjectid)])==1L,]
stopifnot(!anyDuplicated(single$subjectid),nrow(single)<nrow(d))
model <- function(x,scenario) {
  for (name in grep('^dept_',names(x),value=TRUE)) x[[name]] <- as.integer(x[[name]] %in% c(TRUE,'True','TRUE','1',1))
  names(x) <- make.names(names(x),unique=TRUE)
  base <- c('age','male','bmi','asa','emop','op_duration_min','preop_hb','preop_cr','preop_alb','source_invasive',grep('^dept_',names(x),value=TRUE))
  for (name in base) {
    x[[name]] <- as.numeric(x[[name]])
    x[[name]][is.na(x[[name]])] <- mean(x[[name]],na.rm=TRUE)
    stopifnot(all(is.finite(x[[name]])))
  }
  x$trajectory_phenotype <- relevel(factor(x$trajectory_phenotype),ref='preserved_high_map')
  warnings <- character()
  fit <- withCallingHandlers(glm(reformulate(c(base,'trajectory_phenotype'),response='icu_use'),data=x,family=binomial(),control=glm.control(maxit=100)),warning=function(w){warnings <<- c(warnings,conditionMessage(w));invokeRestart('muffleWarning')})
  stopifnot(fit$converged,!anyNA(coef(fit)))
  t <- summary(fit)$coefficients; ix <- grep('^trajectory_phenotype',rownames(t))
  data.frame(scenario=scenario,n=nrow(x),subjects=length(unique(x$subjectid)),events=sum(x$icu_use),term=rownames(t)[ix],adjusted_or=exp(t[ix,1]),wald_low=exp(t[ix,1]-qnorm(.975)*t[ix,2]),wald_high=exp(t[ix,1]+qnorm(.975)*t[ix,2]),warnings=if(length(warnings)) paste(warnings,collapse='; ') else 'none',row.names=NULL)
}
results <- rbind(model(d,'all adult cases'),model(single,'subjects with one eligible case only'))
write.csv(source_table,paths[1],row.names=FALSE)
write.csv(results,paths[2],row.names=FALSE)
print(source_table); print(results)
