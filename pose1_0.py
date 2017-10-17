#!/usr/bin/env python
# -*- coding: utf8 -*- 

import numpy as np
import caffe

class Pose:

  def __init__(self, fnameConfig = None, dtype = np.float32):
    self.dimout = 238 # todo: zjistit automaticky
    self.dtype = dtype
    self.nets = []
    # nacitani siti
    # todo: natvrdo napsane cesty nahradit nacitanim z nejakeho config souboru
    for fnameModel, fnameParams in [
      ["openpose/models/face/pose_deploy.prototxt", "models/face/pose_iter_116000.caffemodel"],
      ["openpose/models/hand/pose_deploy.prototxt", "models/hand/pose_iter_102000.caffemodel"],
      ["openpose/models/pose/coco/pose_deploy_linevec.prototxt", "models/pose/coco/pose_iter_440000.caffemodel"],
      ["openpose/models/pose/mpi/pose_deploy_linevec.prototxt", "models/pose/mpi/pose_iter_160000.caffemodel"],
      ["openpose/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt", "models/pose/mpi/pose_iter_160000.caffemodel"],
    ]:
      net = caffe.Net(fnameModel, fnameParams, caffe.TEST)
      self.nets.append(net)
      
  def get(self, img):
    # img je obrazek nacteny caffe.io.load_image nebo necim kompatabilnim
    ys = []
    for net in self.nets:
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
      for i in range(y.shape[1]):
        ys.append(y[0, i, 0, 0])
  
    return np.asarray(ys, self.dtype)

if __name__ == "__main__":
  # Spustit v dockeru. Treba takhle:
  # sudo docker run -v /home/machine/docker:/workspace -ti bvlc/caffe:cpu /bin/bash
  # Pak zavolat: ipython poseX.py.
  # Pozn.: bude to psat spoustu veci jako "Ignoring source layer Mrelu5_stage6_L2" do error streamu, ale to je v poradku.


  fnameImg = "debug/debug.jpg"
  
  img = caffe.io.load_image(fnameImg)
  
  print("Caffe is starting to complain.") 
  
  pose = Pose()
  
  print("The end of the caffe's complains.")
  print(4 * "\n")
  
  y = pose.get(img)
  
  print("The pose yields %d %s numbers." % (pose.dimout, str(pose.dtype)))

  print("y = ")  
  print(y) 

  print("len(y) = %d" % len(y)) 
