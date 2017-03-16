# autoencoder_heatmap.R

# This script is responsible for producing heatmap visualizations of the output from the final 
# convolutional layer of our convolutional_autoencoder. These heatmap visualizations enable us to
# assess which sections of the input sequence were retained by the CAE. To successfully run my code,
# the variable data.directory  must be changed, based on the user's computer.

# By David Cohn

# R libraries used in generating heatmap visualizations
library(ggplot2)

# Path to directory storing data
data.directory = '/Users/David/Downloads/auto_encoder_results'

# Length of input sequence
sequence.length = 2000

setwd(data.directory)

tsv.results = list.files(pattern = "\\.tsv$")

base.position = seq(1, sequence.length, by = 1)

base.vector = c("Base.1", "Base.2", "Base.3", "Base.4")

# For loop, iterating over autoencoder result .tsv files
for(i in 1:length(tsv.results)){
  # Convert Inputted tsv data from wide to long format
  base.position.data = expand.grid(base.position, base.vector)
  names(base.position.data) = c("Base.Position", "Base")
  tsv.file = tsv.results[i]
  tsv.file.name = tsv.file
  tsv.file.name = strsplit(tsv.file.name, "." , fixed = TRUE)
  tsv.file.name = unlist(tsv.file.name)
  file.type = tsv.file.name[3]
  tsv.data = read.table(tsv.file)
  results.vector = vector("numeric")
  # Place data in vector, to be stored in long format
  for (j in 1:ncol(tsv.data)){
    for(k in 1:nrow(tsv.data)){
      value = tsv.data[k, j]
      results.vector = append(results.vector, value)
    }
  }
  # Binarize autoencoder output
  results.vector = round(results.vector)
  base.position.data$Value = results.vector
  # Create heatmap diagram
  base.position.heat.map = ggplot(base.position.data, aes(x = Base.Position, y = Base)) + 
    geom_tile(aes(fill = Value)) + ggtitle(paste(file.type, "Heatmap", sep = " ")) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  print(base.position.heat.map)
}
