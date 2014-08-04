library(Hmisc)

labels = read.table("list.txt")
order.thing = order(labels$V2);
labels$V2 <- labels$V2[order.thing];
labels$V1 <- labels$V1[order.thing];
dotchart2(labels$V2,labels=labels$V1,cex=.7, horizontal=FALSE, dotsize = 2, lines = FALSE, pch = 8,
         main="Sensitivity Analysis", 
         xlab="Verification Rate")