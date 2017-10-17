import numpy as np
import caffe

def useModel(img, net):
  # load input and configure preprocessing
  transformer = caffe.io.Transformer({'data': net.blobs['image'].data.shape})
  #transformer.set_mean('data', np.load('ilsvrc_2012_mean.npy').mean(1).mean(1))
  transformer.set_transpose('data', (2,0,1))
  transformer.set_channel_swap('data', (2,1,0))
  transformer.set_raw_scale('data', 255.0)
  
  #note we can change the batch size on-the-fly
  #since we classify only one image, we change batch size from 10 to 1
  
  #load the image in the data layer
  #print net.blobs['image'].data.shape
  
  net.blobs['image'].data[0] = transformer.preprocess('data', img)
  #net.blobs['image'].reshape(1,3,368,368)
  
  #print net.blobs['image'].data.shape
  
  #compute
  out = net.forward()
  
  y = out['net_output']
  
  return y

#load the model

#fnameModel = "models/face/pose_deploy.prototxt"
#fnameParams = "models/face/pose_iter_116000.caffemodel"

#fnameModel = "models/hand/pose_deploy.prototxt"
#fnameParams = "models/hand/pose_iter_102000.caffemodel"

#fnameModel = "models/pose/coco/pose_deploy_linevec.prototxt"
#fnameParams = "models/pose/coco/pose_iter_440000.caffemodel"

#fnameModel = "models/pose/mpi/pose_deploy_linevec.prototxt"
#fnameParams = "models/pose/mpi/pose_iter_160000.caffemodel"

fnameModel = "models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
fnameParams = "models/pose/mpi/pose_iter_160000.caffemodel"

fnameImg = "debug.jpg" 

img = caffe.io.load_image(fnameImg)

net = caffe.Net(fnameModel, fnameParams, caffe.TEST)

y = useModel(img, net)


for i in range(y.shape[1]):
  print y[0, i, 0, 0],
print
