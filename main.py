from DeconvNet import *

# Remove use_cpu=True if you have enough GPU memory
deconvNet = DeconvNet(device='/cpu:0')
deconvNet.train()