from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter('saved/')
r = 5
for i in range(100):
  writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                'xcosx':i*np.cos(i/r),
                                'tanx': np.tan(i/r)}, i)
writer.close()
# This call adds three values to the same scalar plot with the tag
# 'run_14h' in TensorBoard's scalar section.