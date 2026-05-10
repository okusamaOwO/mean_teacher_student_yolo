# YOLOv9

# parameters
nc: 80  # number of classes
depth_multiple: 1.0  # model depth multiple
width_multiple: 1.0  # layer channel multiple
#activation: nn.LeakyReLU(0.1)
#activation: nn.ReLU()

# anchors
anchors: 4 

# gelan backbone
backbone:
  [
   # conv down
   [-1, 1, Conv, [32, 3, 2]],  # 0-P1/2

   # conv down
   [-1, 1, Conv, [64, 3, 2]],  # 1-P2/4

   # elan-1 block
   [-1, 1, ELAN1, [64, 64, 32]],  # 2

   # avg-conv down
   [-1, 1, AConv, [128]],  # 3-P3/8

   # elan-2 block
   [-1, 1, RepNCSPELAN4, [128, 128, 64, 3]],  # 4

   # avg-conv down
   [-1, 1, AConv, [192]],  # 5-P4/16

   # elan-2 block
   [-1, 1, RepNCSPELAN4, [192, 192, 96, 3]],  # 6

   # avg-conv down
   [-1, 1, AConv, [256]],  # 7-P5/32

   # elan-2 block
   [-1, 1, RepNCSPELAN4, [256, 256, 128, 3]],  # 8
  ]

# elan head
head:
  [
   # elan-spp block
   [-1, 1, SPPELAN, [256, 128]],  # 9

   # up-concat merge
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], # 10 
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4 # 11

   # elan-2 block
   [-1, 1, RepNCSPELAN4, [192, 192, 96, 3]],  # 12

   # up-concat merge
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], # 13 
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3 # 14 

   # elan-2 block
   [-1, 1, RepNCSPELAN4, [128, 128, 64, 3]],  # 15

# MODIFY FROM HERE

   # up-concat merge
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], # 16 
   [[-1, 2], 1, Concat, [1]],  # cat backbone P2 # 17

   # cbam attention
   [-1, 1, CBAM, [192]], # 18

   # elan-2 block
   [-1, 1, RepNCSPELAN4, [96, 96, 48, 3]],  # 19 (P2/4-small)

   # avg-conv-down merge
   [-1, 1, AConv, [64]], # 20
   [[-1, 15], 1, Concat, [1]], # 21
    
   # elan-2 block
   [-1, 1, RepNCSPELAN4, [128, 128, 64, 3]],  # 22 # P3/8-medium
# END AT THIS POINT 

   # avg-conv-down merge
   [-1, 1, AConv, [96]], # 23 
   [[-1, 12], 1, Concat, [1]],  # cat head P4 # 24 

   # elan-2 block
   [-1, 1, RepNCSPELAN4, [192, 192, 96, 3]],  # 25 (P4/16-medium)

   # avg-conv-down merge
   [-1, 1, AConv, [128]], # 26
   [[-1, 9], 1, Concat, [1]],  # cat head P5 # 27

   # elan-2 block
   [-1, 1, RepNCSPELAN4, [256, 256, 128, 3]],  # 28 (P5/32-large)
   
   # elan-spp block
   [8, 1, SPPELAN, [256, 128]],  # 29

   # up-concat merge
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], # 30 
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4 # 31 

   # elan-2 block
   [-1, 1, RepNCSPELAN4, [192, 192, 96, 3]],  # 32

   # up-concat merge
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], # 33
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3 # 34

   # elan-2 block
   [-1, 1, RepNCSPELAN4, [128, 128, 64, 3]],  # 35 

# START  MODIFY FROM HERE
   # up-concat merge
    [-1, 1, nn.Upsample, [None, 2, 'nearest']], # 36
    [[-1, 2], 1, Concat, [1]],  # cat backbone P2 # 37

   # cbam attention
   [-1, 1, CBAM, [192]], # 38

   # elan-2 block
   [-1, 1, RepNCSPELAN4, [96, 96, 48, 3]],  # 39 (P2/4-small)

# END AT THIS POINT

   # detect
   [[39, 35, 32, 29, 19, 22, 25, 28], 1, DualDDetect, [nc]],  # Detect(P2, P3, P4, P5)
  ]