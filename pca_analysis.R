# pca_analysis.R

# This script is responsible for ensuring that ChIP-seq samples from similar data origins (i.e. Colon samples, Crypt samples etc.), cluster together when principal component analysis (PCA) is applied to chromatin accessibility data for these samples. Performing PCA on the chromatin accessbility data constitutes one form of quality assurance (QA) analysis peformed as part of our project. In order to effectively run my code, the user should change the data file and data directory based on the user's inputted data and computer respectively.

# By David Cohn

# R Libraries Used in Script
library(ggfortify)
library(caret)

# Directory where data is stored
data.directory = '/Users/davidcohniii/Documents/BMI_217_Final_Project'
# File storing chromating acessibility data
data.file = 'chip_seq_h3k27ac.tsv'
# Variance Threshold applied to chromatin accessibility data prior to performing PCA
variance.threshold = 400

setwd(data.directory)

chip.seq.data = read.table(data.file)

# Vector storing sample names
sample.names = t(chip.seq.data[1, 4:ncol(chip.seq.data)])
sample.names = sample.names[, 1]

# This function is responsible for identifying the sample type, based on the
# sample name
process_sample_names = function(sample.name){
  sample.name = strsplit(sample.name, "_")
  sample.name = unlist(sample.name)
  sample.string = sample.name[1]
  sample.status = ""
  # Testing if sample has a colon crypt sample type
  if(grepl("Crypt", sample.string)){
    sample.status = "Crypt"
  # Testing if sample has a colon sample type
  }else if(grepl("COLO", sample.string)){
    sample.status = "Colon"
  # Testing if sample has a normal tissue sample type
  }else if(grepl("c", sample.string)){
    sample.status = "Control"
  # Testing if sample has a cancerous tissue sample type
  }else if(grepl("V", sample.string)){
    sample.status = "Tissue"
  # Testing if sample has a SW sample type
  }else if(grepl("SW", sample.string)){
    sample.status = "SW"
  }
  return(sample.status)
}

sample.status = lapply(sample.names, process_sample_names)
sample.status = unlist(sample.status)

# Setting dataframe column headers
chip.seq.headers = t(chip.seq.data[1, ])
chip.seq.headers = chip.seq.headers[, 1]
names(chip.seq.data) = chip.seq.headers
chip.seq.data = chip.seq.data[-1, ]

chip.seq.data.matrix = chip.seq.data[, 4:ncol(chip.seq.data)]
# Initial removal of chromatin regions from CHIP-seq data via unsupervised variance thresholding
variance.statistics = data.frame(apply(chip.seq.data.matrix, MARGIN = 1, 
                                       var))
variance.statistics = variance.statistics[, 1]
variance.boolean = rep(0, times = nrow(chip.seq.data))
variance.boolean[variance.statistics > variance.threshold] = 1
# Identification of indices, corresponding to chromatin regions, that meet pre-defined variance threshold
variance.indices = which(variance.boolean == 1)
variance.indices = variance.indices + 3
# Removal of low-variance chromatin regions from data set
chip.seq.data = chip.seq.data[variance.indices, ]

# Conversion of CHIP-seq data to type numeric
chip.seq.data.matrix = chip.seq.data[, 4:ncol(chip.seq.data)]
chip.seq.data.matrix = chip.seq.data.matrix
chip.seq.data.matrix = apply(chip.seq.data.matrix, MARGIN = 2, as.numeric)
# PCA calculation for CHIP-seq data
chip.seq.components = prcomp(chip.seq.data.matrix, 
                             center = TRUE)
# Selection of first two principal components for visualization
chip.seq.components = (chip.seq.components$rotation)[, 1:2]
chip.seq.components = data.frame(chip.seq.components)
chip.seq.components = cbind(chip.seq.components, row.names(chip.seq.components), 
                            sample.status)
names(chip.seq.components) = c("PC1", "PC2", "Sample.Name", "Sample.Status")

# Plotting of first two principal components data for each H3K27ac CHIP-seq sample
chip.seq.plot = ggplot(chip.seq.components, aes(x = PC1, y = PC2, 
                colour = Sample.Status)) + geom_point() + 
 ggtitle("H3K27ac CHIP-seq PCA Analysis") + theme(plot.title = element_text(hjust = 0.5))
print(chip.seq.plot)
