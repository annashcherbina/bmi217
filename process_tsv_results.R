# process_tsv_results.R

# This script is responsible for processing and compiling a series of tsv results files that
# are generated for each CNN or RNN architecture. In order to effectively run
# this script, the data directory and results directory should be changed based on the user's computer.

# By David Cohn

# Libraries used in script
library(grid)
library(gridExtra)

# Path to data directory
data.directory = '/Users/davidcohniii/Documents/BMI_217_Final_Project/best_model_results_tsv'
# Path to directory where compiled CNN results should be stored
results.directory = '/Users/davidcohniii/Documents/BMI_217_Final_Project/'

setwd(data.directory)
# Identification of .tsv result files in data directory
tsv.results = list.files(pattern = "\\.tsv$")
# Dataframe storing model results for each CNN or RNN architecture
tsv.metrics.dataframe = matrix(0, nrow = length(tsv.results), ncol = 10)

# Vector storing DNN model names
tsv.model.names = vector("character")

# For loop responsible for processing each model's results
for (i in 1:length(tsv.results)){
  tsv.file = tsv.results[i]
  # Read in model results
  tsv.metrics = read.table(tsv.file)
  # Identification of model metric categories
  tsv.metric.categories = as.character(tsv.metrics[, 1])
  tsv.metrics = tsv.metrics[,-c(1,2)]
  # Averaging model metric results by chromatin accessibility task
  tsv.averaged.metrics = apply(tsv.metrics, MARGIN = 1, mean)
  # Storing averaged model metric results
  for (j in 1:length(tsv.averaged.metrics)){
    tsv.metrics.dataframe[i, j] = tsv.averaged.metrics[j]
  }
  tsv.file = gsub(".tsv", "", tsv.file)
  tsv.model.names = c(tsv.model.names, tsv.file)
}

setwd(results.directory)

# Formatting row and column names of metrics dataframe
tsv.metrics.dataframe = data.frame(tsv.metrics.dataframe)
row.names(tsv.metrics.dataframe) = tsv.model.names
names(tsv.metrics.dataframe) = tsv.metric.categories

# Removing metrics from final table
tsv.display.metrics = tsv.metrics.dataframe[,-c(2,5, 9, 10)]
# Rounding each metric to three significant digits
tsv.display.metrics = signif(tsv.display.metrics, 3)
names(tsv.display.metrics) = c("RecallAtFDR50", "AUC-ROC", "AUC-PRC", 
                               "Balanced Accuracy %", "Positive Accuracy %", 
                               "Negative Accuracy %")

# Generate table with Model performance metrics
tsv.metrics.plot = grid.table(tsv.display.metrics)
