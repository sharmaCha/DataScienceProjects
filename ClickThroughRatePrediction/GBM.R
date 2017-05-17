trying Rpostgresql
library(dplyr)
library(excel.link)
library(magrittr)
# library(caret)
library(forecast)
library(lubridate)
library(cleanr)
library(gbm)
library(RPostgreSQL)
# library(randomForest)

options(stringsAsFactors = FALSE)

model_path = "models/one/"
prediction_path = "predictions/result1.csv"

logloss <- function(truth, pred){
    
    epsilon <- .000000000000001
    yhat <- pmin(pmax(pred, epsilon), 1-epsilon)
    logloss <- -mean(truth*log(yhat) + (1-truth)*log(1 - yhat))
    logloss
    
}



#con = dbConnect(dbDriver("PostgreSQL"), dbname = "avazu",user="postgres",password="root")


in_test = character(0)
to_rec = c("c1",  "device_type", "device_conn_type", "c15", "c16", "c18","site_id","banner_pos", "site_category", "app_category","site_domain" ,"app_id","app_domain","device_model", "c14","c17","c19", "c20","c21") 
# "device_id","device_ip"
not_rec = character(0)

need_vars =  "id,click,hour,c1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_model,device_type,device_conn_type,c14,c15,c16,c17,c18,c19,c20,c21"
# device_id,device_ip,

if(!file.exists("data/hour_freq.rdata")){
    hour_freq = dbGetQuery(con, "select hour,count(*), avg(click) as clicks from train group by hour order by hour;")
    save(hour_freq,file="data/hour_freq.rdata")
}else {
    load("data/hour_freq.rdata")
    
}
need_hours = hour_freq[!grepl("^(141030)",hour_freq$hour),]
need_hours = hour_freq
prop_for_sample = 900000/sum(need_hours$count)

new_day = hour_freq[grepl("^(141030)",hour_freq$hour),"hour"]





dbSendQuery(con, "select setseed(0.129);")
set.seed(20150127)

# curr_model = 1
for (curr_model in 1:10) {
    cat("Model:",curr_model,"\n")
    temp = lapply(need_hours$hour, function(h){
        dbGetQuery(con, sprintf("select %s from train  where random()<%s and hour=%s;",need_vars,prop_for_sample,h))
    })
    w = do.call(rbind,temp)
    rm(temp); gc()
    cat(nrow(w),"\n")
    flush.console()
    colnames(w)[colnames(w) =="hour"] = "hor"
    
    w$h = factor(substr(w$hor,7,8))
    w$wd = wday(ymd(substr(w$hor,1,6)),label = TRUE)
    w$day = factor(substr(w$hor,1,6))
    
    
    for (each in colnames(w)[-(1:2)]){
        w[,each] = as.factor(w[,each])
    }
    
    
    #### build model
    
    
    w %>%  group_by(day,h)  %>% 
        summarize(p=mean(click))  %>%
        arrange(day,h)  -> aggr_h
    
    
    
    # with kmeans
    Nclus = 70
    enc_table = list()
    #     each = to_rec[1]
    for (each in to_rec){
        w  %>% group_by_(each)  %>% summarize(N = n(),p = ifelse(N<30,0.16,mean(click)))  %>%  arrange(p) -> curr_aggr
        N = min(length(unique(curr_aggr$p))-1,Nclus)
        cat(each,":",N,":",mean(curr_aggr$N<30),"\n")
        grouping = kmeans(curr_aggr$p, N)
        curr_aggr$cluster = factor(grouping$cluster)
        # by now we encoded new values into cluster with maximum number of distinct values. 
        # May be its better to encode new values in cluster with maximum number of cases 
        enc_table[[each]] = list(curr_aggr, factor(grouping$cluster)[which.max(grouping$size)])
        w[,each] = vlookup(w[,each],curr_aggr,result_columns = "cluster")
        
    }
    
    
    w %<>% left_join(aggr_h)

    res = gbm(click ~ ., data=w[, setdiff(colnames(w),c("id", "day","hor"))] , verbose = TRUE, n.trees =150, interaction.depth=3,shrinkage = 0.5,keep.data=FALSE) 
    p = predict(res, newdata = w, n.trees = 150, type = "response")
    ll = logloss(w$click,p)
    print(ll)
    
    model = list(model=res,aggr_h = aggr_h,enc_table=enc_table,ll=ll)
    
    save(model,file=paste0(model_path,curr_model,".rdata"))
    rm(w)
    gc()

}

### predict new day

dbGetQuery(con, "select setseed(0.9);")

for (each_hour in seq_along(new_day)) {

    cat("Hour",each_hour,"\n")
    flush.console()
    new_hour = dbGetQuery(con, sprintf("select %s from train  where hour=%s;",need_vars,new_day[each_hour],prop_for_sample*10))
    
    print(nrow(new_hour))
     
    colnames(new_hour)[colnames(new_hour) =="hour"] = "hor"
    
    new_hour$h = factor(substr(new_hour$hor,7,8),levels = substr(new_day,7,8))
    new_hour$wd = wday(ymd(substr(new_hour$hor,1,6)),label = TRUE)
    
    # table(w$day)
    
    
    for (each in colnames(new_hour)[-(1:2)]){
        new_hour[,each] = as.factor(new_hour[,each])
    }
    
   
    i = 1 # model counter
    
    pr = lapply(dir(model_path, full.names = TRUE),function(path){
        # apply different models
        
        load(path)
        w_new = new_hour
        
        for (each in to_rec){
            lookup_table = model$enc_table[[each]][[1]]
            w_new[,each] = vlookup(w_new[,each],lookup_table,result_columns = "cluster")
#             w_new[is.na(unlist(w_new[,each])),each] = model$enc_table[[each]][[2]]
#             lookup_table  %>% group_by(cluster)  %>% summarise(big_sum = sum(N))  -> temp
#             w_new[is.na(unlist(w_new[,each])),each] = temp$cluster[which.max(temp$big_sum)]
#             cat(each,":",mean(is.na(unlist(w_new[,each]))),"\n")
        } 
        
        aggr_h = model$aggr_h
        ts_clicks = ts(aggr_h$p,frequency = 24)
        fc = ets(ts_clicks)
        pr_fc = forecast(fc,h=24)
        w_new$p = pr_fc$mean[each_hour]


        p = predict(model$model, newdata = w_new, n.trees =150, type = "response")
 
        cat(i,":",model$ll,":",logloss(w_new$click,p),"\n")
        rm(model)
        gc()
        
        i <<- i +1
        p
    })
    hour_res = do.call(cbind,pr)
    cat("Averaged p:",logloss(new_hour$click,rowMeans(hour_res)),"\n")
    if (each_hour==1) {
        write.table(data.frame(id=new_hour$id,click=new_hour$click,hour=new_hour$hor,hour_res),
                    file = prediction_path,
                    row.names = FALSE,
                    quote = FALSE,
                    sep = ",")
    } else {
        write.table(data.frame(id=new_hour$id,click=new_hour$click,hour=new_hour$hor,hour_res),
                    file = prediction_path,
                    row.names = FALSE,
                    col.names = FALSE,
                    append = TRUE,
                    quote = FALSE,
                    sep = ",")    
        
    }
    rm(hour_res)

}

ll_train = sapply(dir(model_path, full.names = TRUE),function(path){
    
    load(path)
    model$ll
    
})

plot(sort(ll_train,decreasing = TRUE))

sum(ll_train>0.3)

good = ll_train>0.3 & ll_train<0.4 
good = ll_train<0.4 
good = ll_train>0.3

sum(good)
res = read.table(prediction_path,header=TRUE, sep=",",colClasses="numeric")
click = res$click
pr_m = do.call(cbind,select(res,-(1:3)))
p = rowMeans(pr_m)
# p = rowMeans(pr_m[,good])
logloss(click,p) # 0.4047725



