import numpy as np
from sklearn_lvq import GlvqModel
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
def calcEuclean(x, y):
  res = np.sqrt(np.sum(np.square(np.array(x) - np.array(y))))
  return res
def Nan(localpara_set):
  posindex = range(len(localpara_set))
  res = []
  for k in range(len(posindex)):
    reste = []
    for l in range(len(posindex)):
      dis = 0
      lpst = localpara_set.loc[posindex[k]]
      lpst1 = localpara_set.loc[posindex[l]]
      tedis = []
      tedis1 = []
      for key, var in lpst.items():
        key=np.array(key)
        var=np.array(var)
        tedis.append(var.tolist())
      for key,var in lpst1.items():
        key = np.array(key)
        var = np.array(var)
        tedis1.append(var.tolist())
      for c in range(len(tedis)):
        dis += calcEuclean(tedis[c],tedis1[c])
      reste.append(dis)
    res.append(reste)
  resindex = []
  for fi in range(len(res)):
    fiindex = list(np.argsort(res[fi]))
    fiindex.remove(fi)
    resindex.append(fiindex)
  num=len(resindex)
  numindex = [0 for i in range(num)]
  for ti in range(1, num):
    for tri in range(len(resindex)):
      numindex[resindex[tri][ti - 1]] += 1
    if (0 not in numindex):
      break
  resindex = np.array(resindex)
  resindex=resindex[:, :ti]
  return resindex
X, y = make_classification(n_samples=90, n_features=5, n_informative=3, 
                           n_redundant=0, n_classes=2, 
                           n_clusters_per_class=1, weights=[0.89, 0.11], 
                           )
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = GlvqModel(prototypes_per_class=1)
model.fit(X_scaled, y)
prototypes = model.prototypes_
prototype_labels = model.prototypes_labels_
prototype_1 = prototypes[prototype_labels == 1][0]
X_class_0 = X_scaled[y == 0]
X_combined = np.vstack([X_class_0, prototype_1])
nearest_neighbors = Nan(X_combined_df)
inserted_prototype_index = len(X_combined) - 1  
nearest_neighbors_for_prototype = nearest_neighbors[inserted_prototype_index]
rows_to_delete = np.append(nearest_neighbors_for_prototype, inserted_prototype_index)
X_class_0_new = np.delete(X_class_0, rows_to_delete[:-1], axis=0) 
X_class_1 = X_scaled[y == 1]
X_final_combined = np.vstack([X_class_0_new, X_class_1])


