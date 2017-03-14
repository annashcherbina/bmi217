# visualize_training_validation_loss.R

# This script is responsible for plotting training vs. validation loss curves by epoch, based on .csv files
# generated in the construction of CNN models. To sucessfully run my code, the data directory (data.directory)
# should be changed based on the user's computers.

# By David Cohn

# Libraries in R used in this script

library(ggplot2)
library(dplyr)

# Path to directory storing training/validation loss results data
data.directory = '/Users/davidcohniii/Documents/BMI_217_Final_Project'
# Name of training/validation loss results csv file
results.file = 'training_results_deep_learning_model5.csv'

setwd(data.directory)

# Training/Validation Loss data I/O
training.validation.epoch.results = read.csv(results.file)
names(training.validation.epoch.results) = c("Epoch", "Train.Loss", "Validation.Loss")

# Plotting Training/Validation Loss results by Epoch
loss.plot = ggplot(training.validation.epoch.results, aes(Epoch)) + 
  geom_line(aes(y = Train.Loss, colour = "Train.Loss")) +
  geom_line(aes(y = Validation.Loss, colour = "Validation.Loss")) + 
  scale_colour_manual("Loss.Values", breaks = c("Train.Loss", "Validation.Loss"),
                     values = c("blue", "red")) + ylab("Loss") + 
  ggtitle("Training/Validation Loss Results: Basset-Like Model") + theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))
print(loss.plot)

  

