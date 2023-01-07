from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .exdet import ExdetDetector
from .ddd import DddDetector

# Multi_pose
from .multi_pose import MultiPoseDetector
#from .multipose_FHPE import MultiPoseDetector
#from .multipose_DCA import MultiPoseDetector
#from .multipose_DCAL import MultiPoseDetector
from .multipose_uni import MultiPoseDetector

# Ctdet
from .ctdet import CtdetDetector
from .ctdet_DCA import CtdetDetector
#from .ctdet_DCAL2 import CtdetDetector
#from .ctdet_DCAS import CtdetDetector
#from .ctdet_SCA import CtdetDetector
#from .ctdet_deepfool import CtdetDetector
detector_factory = {
  'exdet': ExdetDetector, 
  'ddd': DddDetector,
  'ctdet': CtdetDetector,
  'multi_pose': MultiPoseDetector, 
}
