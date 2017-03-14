# hyper_parameter_generation.R

# This script is responsible for generating model hyperparameters to use in Basset-like convolutional neural network models as part of a random grid search.

# By David Cohn

# Randomly select number of convolution layers between 1 and 5
num.conv.layers = sample(seq(1,5, by = 1), 1)
print(num.conv.layers)

# Randomly select number of convolution filters, between 30 and 400
num.conv.filters = sample(seq(30, 400, by = 10), 1)
print(num.conv.filters)

# Vector to store kernel width values for each convolution layer
kernel.width.values = rep(0, times = num.conv.layers)
# Vector to store boolean, indicating whether batch normalization will be performed
# for each convolutional layer (0 = No, 1 = Yes)
batch.normalization.boolean.vector = rep(0, times = num.conv.layers)

# List of possible kernel widths to sample
kernel.width.sample = seq(4, 20, by = 1)
# Random selection of kernel width and batch normalization status for each
# convolution layer. In order to ensure model compilation, the kernel width
# for a successive layer cannot be larger than the kernel width of the prior
# layer.
for (i in 1:num.conv.layers){
  kernel.width.values[i] = sample(kernel.width.sample, 1)
  batch.normalization.boolean.vector[i] = sample(seq(0, 1, by = 1), 1)
  kernel.width.sample = kernel.width.sample[kernel.width.sample <= kernel.width.values[i]]
}
print(kernel.width.values)

# Random selection of learning rate multiplier
learning.rate.multiplier = sample(seq(1, 10, by = 1), 1)
print(learning.rate.multiplier)

# Random selection of pooling stride value
pooling.stride = sample(seq(3, 30, by = 1), 1)
print(pooling.stride)

# Random Selection of Number of Fully Connected Layers
num.fully.connected.layers = sample(seq(0, 3, by = 1), 1)

# Random selection of layer size for each fully connected layer. In order to
# ensure model compilation, the size for a successive layer cannot be larger than
# the size of the prior layer.
if(num.fully.connected.layers > 0){
  fully.connected.layer.size.vector = rep(0, times = num.fully.connected.layers)
  fully.connected.sample = c(10, 50, 100, 250, 500, 750, 1000)
  for(i in 1:num.fully.connected.layers){
    fully.connected.layer.size.vector[i] = sample(fully.connected.sample, 1)
    fully.connected.sample = fully.connected.sample[fully.connected.sample <= fully.connected.layer.size.vector[i]]
  }
}
print(fully.connected.layer.size.vector)

