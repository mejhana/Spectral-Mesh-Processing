import pyglet
pyglet.options['shadow_window'] = True
import pyrender #to display mesh
import numpy as np
import trimesh #to load mesh
# import igl
# import matplotlib
# import matplotlib.pyplot as plt
# from scipy import sparse
# from sklearn.neighbors import KDTree

from utils import *

fp = 'meshes/curvatures/plane.obj'

tm1 = trimesh.load(fp)
v_P = tm1.vertices
H = mean_curvature(tm1,laplaceFunc=laplace_beltrami_operator)
print('mean curvature min:',min(H),', max:',max(H))

Hconst = np.max(np.abs(H))
Hcolors = np.array([curv_colouring(H[i],Hconst) for i in range(v_P.shape[0])])

num = v_P.shape[0]
tm1.visual.vertex_colors=Hcolors
mesh_rd1 = pyrender.Mesh.from_trimesh(tm1)
tms=[tm1]
mesh_rds=[mesh_rd1]
show_mesh_gui(mesh_rds)



