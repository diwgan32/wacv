library(ROCR);

for(n in 1:24){
  
  str = paste("..\\..\\Results\\", n, "\\SimMat.txt", sep="");
  labels = read.table(str, sep = "\n");
  
  str = paste("..\\..\\Results\\", n, "\\SimMat1.txt", sep="");
  predi = read.table(str, sep = "\n");
  
  pred <- prediction(predi, labels);
  perf <- performance(pred, "tpr", "fpr");
  
  plot(perf, xlab = "False alarm rate", ylab = "Verification Rate")
  cutoffs <- data.frame(fpr=perf@x.values[[1]], 
                        tpr=perf@y.values[[1]])
  
  data2 <- subset(cutoffs, fpr>=.1);
  
  print(data2[1, "tpr"])
}

