library(data.table)
source('knn_smoother.R')
r2 <- function(actual, pred)  return(1000*(1-var(actual-pred)/var(actual)))
train <- fread('./data/train_content.csv')
setnames(train, 1:4, c("group_id", "post_id", "timestamp", "text"))
setkey(train, post_id)
test <- fread('./data/test_content.csv')
setnames(test, 1:4, c("group_id", "post_id", "timestamp", "text"))
train_likes_count <- fread('./data/train_likes_count.csv')
setkey(train_likes_count, post_id)
train=train[train_likes_count]
test[,likes:=NA_integer_]
setcolorder(test, c("post_id", "group_id", "timestamp", "text", "likes"))
all<-rbindlist(list(train, test))
rm(test, train, train_likes_count)
setkey(all, group_id, timestamp, post_id)
all[, timestamp:=as.numeric(timestamp)]
all[, images:= (grepl("Images", text))]
all[, url:= (grepl("http|.ru|.com", text))]
# calculate knn smoothing; prediction = geometric mean
system.time(all[, likes_knn := knnreg(timestamp, likes, k=min(1+floor(sum(!is.na(likes))/10),20), fun=function(x) exp(mean(log(x+1)))), by=group_id])

# k-fold crossvalidation
nfolds<-10
len<-nrow(all[!is.na(likes),])
foldid = sample(rep(seq(nfolds), length = len))
scores<-c()
for (i in 1:nfolds)
{
  which = foldid == i
  fit <- lm(likes ~ likes_knn * (images + images * url), data = all[!is.na(likes),][which])
  pred <- predict(fit, all[!is.na(likes),][-which])
  r_2 <- r2(actual = all[!is.na(likes),][-which][,likes], pred = pred)
  print(r_2)
  scores <- append(scores, r_2)
}
print(mean(scores))
print(sd(scores))
################################################################################################
# fit full model
fit <- lm(likes ~ likes_knn * (images + images * url), data = all[!is.na(likes),])
# predict
pred <- predict(fit, all[is.na(likes), ])
pred[is.na(pred)] <- 0
pred[pred < 0] <- 0
res <- data.table(post_id = all[is.na(likes), post_id], likes = pred2)
write.table(res, file = 'submission.csv', row.names = F, col.names = F, append = F, sep=',')
