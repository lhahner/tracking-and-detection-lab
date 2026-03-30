"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public L:icense for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import numpy as np
np.random.seed(0)

from .kalmanBoxTracker import KalmanBoxTracker

class Sort(object):
  """Track multiple objects online using the SORT algorithm."""

  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """Initialize the SORT tracker.

    Args:
      max_age: Maximum number of missed frames before dropping a track.
      min_hits: Minimum hits before a track is returned.
      iou_threshold: Minimum IoU used for detection-to-track assignment.
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def linear_assignment(self, cost_matrix):
    """Solve the linear assignment problem for a given cost matrix.

    Args:
      cost_matrix: Pairwise assignment cost matrix.

    Returns:
      numpy.ndarray: Matched index pairs.
    """
    try:
      import lap
      _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
      return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
      from scipy.optimize import linear_sum_assignment
      x, y = linear_sum_assignment(cost_matrix)
      return np.array(list(zip(x, y)))

  def update(self, dets=np.empty((0, 5))):
    """Update active SORT tracks with detections from the current frame.

    Args:
      dets: Detection array in `x1, y1, x2, y2, score` format.

    Returns:
      numpy.ndarray: Tracking results in `x1, y1, x2, y2, track_id` format.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets,trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))

  def associate_detections_to_trackers(self, detections,trackers,iou_threshold = 0.3):
      """Associate detections with tracker predictions using IoU matching.

      Args:
        detections: Detection array for the current frame.
        trackers: Predicted tracker boxes for the current frame.
        iou_threshold: Minimum IoU required to keep a match.

      Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Matches,
        unmatched detections, and unmatched trackers.
      """
      if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    
      iou_matrix = self.iou_batch(detections, trackers)
    
      if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
          matched_indices = self.linear_assignment(-iou_matrix)
      else:
        matched_indices = np.empty(shape=(0,2))
    
      unmatched_detections = []
      for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
          unmatched_detections.append(d)
      unmatched_trackers = []
      for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
          unmatched_trackers.append(t)
    
      #filter out matched with low IOU
      matches = []
      for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
          unmatched_detections.append(m[0])
          unmatched_trackers.append(m[1])
        else:
          matches.append(m.reshape(1,2))
      if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
      else:
          matches = np.concatenate(matches,axis=0)

      return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
  
  def iou_batch(self, bb_test, bb_gt):
        """Compute pairwise IoU between test boxes and ground-truth boxes.

        Args:
          bb_test: Candidate boxes.
          bb_gt: Reference boxes.

        Returns:
          numpy.ndarray: IoU matrix.
        """
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)
  
        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
                  + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
        return(o)  
