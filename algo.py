import numpy as np

#  Dectections should be from YOLO
#--------------------------------------------------------
### Generates a scene
def genDetections(num_people, mean=50, vari=0.1):
  frame = []
  for x in range(num_people): 
    person = mean + vari*np.random.randn(2, 2)  #N(mau = 1, var = 0.01):
    person = np.round(person, 2)
    frame.append(person[0])
  return frame

### From last scene generates next scene
def nextDetections(prev_frame):
  new_frame = []
  for i, p in enumerate(prev_frame):
    person = p + np.random.uniform(-1, 1.4 ,(1,2)) 
    person = np.round(person, 2)
    new_frame.append(person[0])
  return new_frame

#  Using frames to create vectors
#--------------------------------------------------------
def genVectors(frame1, frame2):
  # TODO speed, angle
  vectors = []
  for i, p in enumerate(frame2):  
    speed = round(0.0+np.random.uniform(0.7, 2.1),2) # average human speed is 1.4ms, (f2-f1)/time
    angle = round(int(np.random.uniform(60, 180)),2)
    vectors.append( [p[0],p[1],speed,angle] ) 
  return vectors

def groundTruth(matrix, k_clusters):

  from itertools import combinations
  nodes = [ vec for vec in matrix ]
  #edges1 = [ (vec1,vec2) for i, vec1 in enumerate(matrix) for vec2 in matrix[i:] ]
  edges2 = list(combinations(nodes, 2))

  DTH, STH, ATH = 7, 0.3, 60   # dist speed angle threshold
  label = np.zeros((len(matrix)))
  for i, v1 in enumerate(matrix):
    for j, v2 in enumerate(matrix[i:]):
      dist = np.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2)
      #if dist < DTH and dist > 0: label[i] = 1 
      label[i] = 1 if dist < DTH and dist > 0 else 0

      #speed = label[i] * abs(v1[2] - v2[2])
      speed = abs(v1[2] - v2[2])
      label[i] = 1 if speed < STH and speed > 0 else 0
      angle = label[i] * abs(v1[3] - v2[3])
      label[i] = 1 if angle < ATH and angle > 0 else 0

  label = label.astype(int)
  return label


def KMeans(matrix, k_clusters):
  from sklearn.cluster import KMeans
  model = KMeans(n_clusters=k_clusters, n_init=200)

  model.fit(matrix)
  label = np.array(model.labels_)
  return label

def AgglomerativeClustering(matrix, k_clusters):
  from sklearn.cluster import AgglomerativeClustering
  model = AgglomerativeClustering(n_clusters=2)

  model.fit(matrix)
  label = np.array(model.labels_)
  return label

def AffinityPropagation(matrix, k_clusters):
  from sklearn import cluster
  model = cluster.AffinityPropagation(damping=0.75, random_state=None)

  model.fit(matrix)
  label = np.array(model.labels_)
  return label

def DBSCAN(matrix, k_clusters):
  from sklearn.cluster import DBSCAN
  model = DBSCAN()

  model.fit(matrix)
  label = np.array(model.labels_)
  return label


def BIRCH(matrix, k_clusters):
  from sklearn.cluster import Birch
  model = Birch(threshold=0.01, n_clusters=k_clusters)

  model.fit(matrix)
  label = np.array(model.labels_)
  return label

def KMedoids(matrix, k_clusters):
  from sklearn_extra.cluster import KMedoids
  model = KMedoids(n_clusters=k_clusters, random_state=0)

  model.fit(matrix)
  label = np.array(model.labels_)
  return label
