# Load in useful libraries
library(limma)
library(dplyr)

# Set the working directory
setwd('/Users/turnberry2012/Documents/BMI217/Project/Differential\ Gene\ Analysis')
folder_directory = list.files(pattern = ".*.txt")

# Read in all of the time-series RNA seq in proper sequence
datafr = lapply(folder_directory, FUN=read.table, header=TRUE)

# Instantiate a fpkm_matrix for the time-series
fpkm_matrix = as.data.frame((as.data.frame(datafr[1]))[,1])
colnames(fpkm_matrix) = "Gene.Names"

# Cycle through all time series and bind them together
for(iter in seq(1,30,1)){
  
  temp_fpkm = as.data.frame(as.data.frame(datafr[iter])$FPKM)
  fpkm_matrix = cbind(fpkm_matrix,temp_fpkm)
  
}

# Read in the control RNA seq samples
control = read.csv('/Users/turnberry2012/Documents/BMI217/Project/rna.all.fpkm.merged_replicates.txt', header = FALSE, sep = "\t")

# Remove the column headers
control = control[-1,]

# Find the overlap of genes common to both sets
control_matched_genes = control$V1 %in% fpkm_matrix$Gene.Names

# Extract just the corresponding rows
control = control[control_matched_genes,]

# Remove any duplicated rows
control = control[!duplicated(control$V1),]

# Find the remaining overlap of genes
fpkm_matched_genes = fpkm_matrix$Gene.Names %in% control$V1

# Extract just these rows
fpkm_matrix = fpkm_matrix[fpkm_matched_genes,]

# Remove any duplicated rows
fpkm_matrix = fpkm_matrix[!duplicated(fpkm_matrix$Gene.Names),]

# Give each column of our time-series matrix unique names
uniq_colnames = seq(1,ncol(fpkm_matrix))
colnames(fpkm_matrix) = uniq_colnames

# Join the two data sets together
all_data = inner_join(fpkm_matrix,control, by = c(`1` = 'V1'))

# Create a categorical design matrix for the linea rmodel
time_vector = as.data.frame(c(1,0,2,4,3,1,0,2,4,3,1,0,2,4,3,1,0,2,4,3,1,0,2,4,3,1,0,2,4,3))
#time_vector = as.data.frame(c(0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1))
colnames(time_vector) = 'Value'

# Add the values for all of the control samples
control_vector = as.data.frame(replicate(141,5))
colnames(control_vector) = 'Value'
time_vector = rbind(time_vector,control_vector)

# Create the design matrix
designMatrix = model.matrix(~ unlist(time_vector))

# Remove the gene names to create the raw expressin set
fpkm_eset = all_data[,-1]

# Convert to numeric
fpkm_eset = apply(fpkm_eset,1,function(x) as.numeric(as.character(x)))

# Apply the log2 approximation
fpkm_eset = asinh(fpkm_eset)

# Fit a linear model to each sample based on output for design matrix
exp_model = lmFit(t(fpkm_eset),designMatrix)

# Calculate the statistically significant differences between genes
exp_model = eBayes(exp_model, trend = TRUE)

# Extract these statistics from our model and put them in a nice table
sigGenes = topTable(exp_model, number = nrow(exp_model))

# Reattach the gene names
sigGenes = cbind(as.numeric(as.character(rownames(sigGenes))),sigGenes)
colnames(sigGenes) = "GeneNumber"

# Order by gene name
attach(sigGenes)
sigGenes = sigGenes[order(GeneNumber),]
detach(sigGenes)

# Remove gene numbers now that they are ordered
sigGenes = sigGenes[,-1]

# Reattach real gene names since they are in proper order
sigGenes = cbind(fpkm_matrix[,1],sigGenes)

# Name the columns
colnames(sigGenes) = c("Gene.Name","LogFC","AveExpr","t","P.Value","adj.P.Val","B")

# Extract the top genes
filtered_sigGenes = filter(sigGenes, adj.P.Val <= 0.05 & abs(LogFC) > 2.0)

# Extract the most significant gene names
sig_gene_names = as.character(filtered_sigGenes$Gene.Name)

# Use these names to grab the appropriate expression rows
allData_sig_truth = as.character(all_data$`1`) %in% sig_gene_names 
all_data_sig = all_data[allData_sig_truth,]

# Remove the names
all_data_sig_noNames = all_data_sig[,-1]

# For each time point extract the data and calculate the average expression of each gene
Hour_0_data = all_data_sig_noNames[,(time_vector == 0)]
Hour_0_data = as.data.frame(rowMeans(Hour_0_data))

Hour0p5_data = all_data_sig_noNames[,(time_vector == 1)]
Hour0p5_data = as.data.frame(rowMeans(Hour0p5_data))

Hour1_data = all_data_sig_noNames[,(time_vector == 2)]
Hour1_data = as.data.frame(rowMeans(Hour1_data))

Hour6_data = all_data_sig_noNames[,(time_vector == 3)]
Hour6_data = as.data.frame(rowMeans(Hour6_data))

Hour24_data = all_data_sig_noNames[,(time_vector == 4)]
Hour24_data = as.data.frame(rowMeans(Hour24_data))

controlSig_data = all_data_sig_noNames[,(time_vector == 5)]
controlSig_data = t(apply(controlSig_data,1,function(x) as.numeric(as.character(x))))
controlSig_data = as.data.frame(rowMeans(controlSig_data))

# Combine all expression sets
time_matrix = cbind(controlSig_data,Hour_0_data,Hour0p5_data,Hour1_data,Hour6_data,Hour24_data)
row.names(time_matrix) = as.character(all_data_sig$`1`)
colnames(time_matrix) = c("Control","0 Hours","0.5 Hours","1 Hour","6 Hours","24 Hours")

# Plot the heatmap of expression over time
library(lattice)
gene_plot = levelplot(asinh(t(as.matrix(time_matrix))), xlab = 'Time Point', ylab = 'Genes')