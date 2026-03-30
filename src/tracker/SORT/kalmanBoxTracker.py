from __future__ import print_function

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from filterpy.kalman import KalmanFilter

class KalmanBoxTracker(object):
  """Track a single object state with a Kalman filter in box space."""

  count = 0
  def __init__(self,bbox):
    """Initialize the tracker from an observed bounding box.

    Args:
      bbox: Bounding box in `x1, y1, x2, y2` format.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = self.convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """Update the tracker state with a new observed bounding box.

    Args:
      bbox: Bounding box in `x1, y1, x2, y2` format.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(self.convert_bbox_to_z(bbox))

  def predict(self):
    """Advance the Kalman filter and return the predicted bounding box.

    Returns:
      numpy.ndarray: Predicted bounding box in `x1, y1, x2, y2` format.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(self.convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """Return the current estimated bounding box.

    Returns:
      numpy.ndarray: Current bounding box in `x1, y1, x2, y2` format.
    """
    return self.convert_x_to_bbox(self.kf.x)

  def convert_bbox_to_z(self, bbox):
    """Convert a corner-format box into the Kalman filter measurement space.

    Args:
      bbox: Bounding box in `x1, y1, x2, y2` format.

    Returns:
      numpy.ndarray: Measurement vector `[x, y, s, r]`.
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

  def convert_x_to_bbox(self, x,score=None):
    """Convert the Kalman state vector back into corner-format bounding boxes.

    Args:
      x: Kalman state vector.
      score: Optional detection score to append to the output.

    Returns:
      numpy.ndarray: Bounding box array in `x1, y1, x2, y2` form, optionally
      with a trailing score.
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))
