
labels=read.table("SimMat.txt", sep="\n");
predi =read.table("SimMat1.txt", sep="\n");

pred <- prediction(predi, labels);
perf <- performance(pred, measure = "tpr", x.measure = "fpr") 

pdf('plot.pdf')
plot(perf, col=rainbow(12))
dev.off();