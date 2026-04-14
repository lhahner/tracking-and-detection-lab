import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from datasets.kitti3D import Kitti3D
from detector.pointnet.proposals import dbscan_clustering, cluster_to_proposal

class TestProposals(unittest.TestCase):
    def test_euclidean_clustering(self):
        points = np.genfromtxt('tests/point_samples.csv', delimiter=',')
        clusters  = np.array(dbscan_clustering(points), dtype=object)
        
        max_value_x = np.max(np.array(clusters[0])[:,0])
        min_value_x = np.min(np.array(clusters[0][:,0]))
        
        self.assertTrue((max_value_x - min_value_x) < 5)
        
        max_value_y = np.max(np.array(clusters[0])[:,1])
        min_value_y = np.min(np.array(clusters[0][:,1])) 
       
        self.assertTrue((max_value_y - min_value_y) < 5)
        
        max_value_z = np.max(np.array(clusters[0])[:,2])
        min_value_z = np.min(np.array(clusters[0][:,2])) 
        
        self.assertTrue((max_value_z - min_value_z) < 5)
        
    def test_cluster_to_proposal(self):
        points = np.genfromtxt('tests/point_samples.csv', delimiter=',')
        clusters  = np.array(dbscan_clustering(points), dtype=object)
        proposals = [cluster_to_proposal(cluster) for cluster in clusters] 
        self.assertTrue(proposals != None)