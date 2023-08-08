import numpy as np
from scipy import sparse
from scipy.sparse import diags
import pyglet
pyglet.options['shadow_window'] = True
import pyrender
import igl

def scene_factory(render_list, return_nodes=False):
    
    scene = pyrender.Scene(ambient_light=0.5*np.array([1.0, 1.0, 1.0, 1.0]))
    nd_list=[]
    for m in render_list:
        nd=scene.add(m)
        nd_list.append(nd)
    
    if return_nodes:
        return scene, nd_list
    else:
        return scene
    
def show_mesh_gui(rdobi):
    scene = scene_factory(rdobi)
    vertices=pyrender.Viewer(scene, use_raymond_lighting=True,show_world_axes=True)
    del vertices

def curv_colouring(curv,const):
    if curv<0:
        return [0.2 + 0.8*(abs(curv)/const)**0.5,   0.2 - 0.2*(abs(curv)/const)**0.5,     0.2- 0.2*(abs(curv)/const)**0.5]
    else:
        return [0.2 - 0.2*(abs(curv)/const)**0.5,   0.2 + 0.8*(abs(curv)/const)**0.5,     0.2- 0.2*(abs(curv)/const)**0.5]
    
def uniform_laplace(mesh):
    """
    Calculates the uniform laplacian of a mesh with num_vertices vertices and vertex_neighbours as its one rings.
    ...

    Inputs
    ----------
    mesh : trimesh obiect
        vertices in the mesh

    Outputs
    -------
    Uniform Laplace operator of the mesh as a sparse matrix of size num_vertices x num_vertices
    """
    vertices = mesh.vertices
    vertex_neighbours = mesh.vertex_neighbors
    num_vertices = len(vertices)
    L = sparse.lil_matrix((num_vertices,num_vertices))
    for i in range(num_vertices):
        L[i,vertex_neighbours[i]] = 1/len(vertex_neighbours[i])
        L[i,i] = -1
    L = sparse.csr_matrix(L)
    return L

def mean_curvature(mesh,laplaceFunc=uniform_laplace):
    """
    Calculates the mean curvature H of the mesh— ∆x/2 
    ...

    Inputs
    ----------
    mesh : trimesh obiect
        vertices in the mesh

    Outputs
    -------
    Mean curvature H of the mesh
    """   
    vertices = mesh.vertices
    normals = mesh.vertex_normals

    if laplaceFunc == uniform_laplace:
        laplace_on_coords = laplaceFunc(mesh)@vertices
    else:
        M_inv, C = laplaceFunc(mesh)
        laplace_on_coords = M_inv@C@vertices
    
    signs = np.sign(np.sum(normals*laplace_on_coords, axis = 1))*(-1)
    H = 0.5*np.sum(laplace_on_coords **2,axis=1)
    H = H*signs
    
    return H
    
def gauss_curvature(mesh):
    """
    Calculates the Gaussian curvature of the mesh
    ...

    Inputs
    ----------
    mesh : trimesh obiect
        vertices in the mesh

    Outputs
    -------
    Gaussian curvature K of the mesh
    """    
    vertices = mesh.vertices
    vertex_neighbours = mesh.vertex_neighbors

    gaussCurvature = np.zeros(vertices.shape[0])
    for i in range(vertices.shape[0]):
        neigh = vertex_neighbours[i]
        numNeigh = len(neigh)
        
        # find the a*b*cos(theta) for each vertex with its neighbours
        dot_products = np.array([np.sum((vertices[i,:] - vertices[neigh[i],:])*(vertices[i,:] - vertices[neigh[(i+1)%numNeigh],:]))
                            for i in range(numNeigh)])
        # find ||a*b|| for each vertex with its neighbours i.e., its magnitude
        magnitudes = np.array([
            (np.linalg.norm(vertices[i,:] - vertices[neigh[i],:])*np.linalg.norm(vertices[i,:] - vertices[neigh[(i+1)%numNeigh],:])) 
                            for i in range(numNeigh)])
        # find cos(thetas) for each vertex with its neighbours
        cosines = np.clip(dot_products/magnitudes, -1,1)
        # find sin(thetas) for each vertex with its neighbours
        sines = np.clip((1 - cosines**2)**0.5, -1, 1)
        # angle deficit = 2pi - sum of angles around a vertex
        angle_deficit = 2*np.pi - np.sum(np.arccos(cosines))
        # total area of the triangle fan around a vertex (one ring neighbourhood) = sum of (1/2)absin(theta) for each triangle
        total_area = np.sum(0.5*magnitudes*sines)
        # normalize the local neighbourhood area by dividing by 3
        gaussCurvature[i] = angle_deficit/(total_area/3.0)
    return gaussCurvature


def laplace_beltrami_operator(mesh):
    """
    Calculates the laplacian beltrami operator using cotan weights
    Parameters
    ----------
    mesh : trimesh obiect

    Returns 
    -----------
    C    : (n,n) float, cotan weight matrix of mesh
    M_inv: (n,n) float, mass matrix with diagonal elements set to 1/2A
    """
    vertices = mesh.vertices
    num_vert = len(vertices)
    neighbors = [mesh.vertex_neighbors[i] for i in range(len(vertices))]
    Minv = sparse.lil_matrix((num_vert, num_vert))
    C = sparse.lil_matrix((num_vert, num_vert))
    for i in range(num_vert):
        print(f"for vertex {i}")
        # indices of neighbouring vertices
        vertex_neigh = np.array(neighbors[i])
        # number of neighbouring vertices   
        val = len(vertex_neigh)
        dot_products_row1 = np.array([np.sum((vertices[i,:] - vertices[vertex_neigh[(i+1)%val],:])*
                                     (vertices[vertex_neigh[i],:] - vertices[vertex_neigh[(i+1)%val],:]))
                                 for i in range(val)])#dot product gives (ab)cosC for each face
            
        magnitudes_row1 = np.array([#magnitudes of dot products (i.e the ab for each face)
            (np.linalg.norm(vertices[i,:] - vertices[vertex_neigh[(i+1)%val],:])*
                        np.linalg.norm(vertices[vertex_neigh[i],:] - vertices[vertex_neigh[(i+1)%val],:]))
                                for i in range(val)])
        
        
        dot_products_row2 = np.array([np.sum((vertices[i,:] - vertices[vertex_neigh[(i-1)%val],:])*
                                (vertices[vertex_neigh[i],:] - vertices[vertex_neigh[(i-1)%val],:]))
                                    for i in range(val)])#dot product gives (ab)cosC for each face
        
        
        magnitudes_row2 = np.array([#magnitudes of dot products (i.e the ab for each face)
            (np.linalg.norm(vertices[i,:] - vertices[vertex_neigh[(i-1)%val],:])*
                    np.linalg.norm(vertices[vertex_neigh[i],:] - vertices[vertex_neigh[(i-1)%val],:]))
                                for i in range(val)])
        
        cosines = np.vstack([dot_products_row1/magnitudes_row1,
                                dot_products_row2/magnitudes_row2])
        #print(cosines.shape)
        
        
        cosines = np.clip(cosines, -1,1)
        
        sines = np.clip((1 - cosines**2)**0.5, 0, 1)
        
        
        cotans = cosines/sines
        
        #####################work out area normalisation#####################
        
        
        dot_products = np.array([np.sum((vertices[i,:] - vertices[vertex_neigh[i],:])*
                                        (vertices[i,:] - vertices[vertex_neigh[(i+1)%val],:]))
                            for i in range(val)])#dot product gives (ab)cosC for each face
        
        magnitudes = np.array([#magnitudes of dot products (i.e the ab for each face)
            (np.linalg.norm(vertices[i,:] - vertices[vertex_neigh[i],:])*
                np.linalg.norm(vertices[i,:] - vertices[vertex_neigh[(i+1)%val],:])) 
                            for i in range(val)])
        
        cosines = np.clip(dot_products/magnitudes, -1,1)
        
        sines = np.clip((1 - cosines**2)**0.5, 0, 1)
        
        total_area = np.sum(0.5*magnitudes*sines)#because area of triangle is (1/2)absinC

        C[i,vertex_neigh[i]] = np.sum(cotans, axis = 0)
        C[i,i] = -1*np.sum(cotans)# -sum of all cot_alpha_ii + cot_beta_ii
                                                        #for each i
        
        M[i,i] = (2*total_area/3)
        Minv[i,i] = 1/(2*total_area/3)
       
    M = sparse.csr_matrix(M)
    Minv = sparse.csr_matrix(Minv)
    C = sparse.csr_matrix(C)
    L = Minv@C
    
    return Minv,C
    
