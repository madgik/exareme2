import numbers
import numpy as np
import numpy.lib.mixins

from typing import List,Dict

#from ListOfNpArrays import ListOfNpArrays
#import ListOfNpArrays as customnumpy

from ..arraybundle import ArrayBundle


import pdb

def generate_initial_centroids(num_of_dimensions:int, num_of_clusters:int, min_val:float, max_val:float) -> Dict:
    #l=[np.random.uniform(min_val,max_val,size=(num_of_clusters)) for i in range(num_of_dimensions)]
    l=[np.random.uniform(min_val,max_val,size=(num_of_dimensions)) for i in range(num_of_clusters)]
    #print(f"\n(generate_initial_centroids) l-> {l}\n")
    initial_centroids=ListOfNpArrays(l)
    return initial_centroids

def f1():
    a=ArrayBundle([np.array([1.,2.,4.])])
    print(f"a-> {a}")
#####def local_calc(centroids:ListOfNpArrays, datapoints:ListOfNpArrays) -> ListOfNpArrays :
####def local_calc(centroids_data_cross_prod:ListOfNpArrays) -> ListOfNpArrays :
####    num_of_dimensions=int(len(centroids_data_cross_prod)/2)
####    offset=num_of_dimensions+1
####    #distances=[]
####    distances=np.zeros((centroids_data_cross_prod.get_num_rows(),2))
####    for row_index in range(centroids_data_cross_prod.get_num_rows()):
####        row=centroids_data_cross_prod.get_row(row_index)
####        #print(f"(local_calc) row-> {row}")
####        datapoint_index=int(row[0])
####        #print(f"(local_calc) datapoint_index-> {datapoint_index}")
####        datapoint=[row[col_index] for col_index in range(1,offset)]
####        #print(f"(local_calc) datapoint-> {datapoint}")
####        centroid=[row[col] for col in range(offset,len(centroids_data_cross_prod))]
####        #print(f"(local_calc) centroid-> {centroid}")
####        dist=abs(np.linalg.norm(np.array(datapoint)-np.array(centroid))) #makes copies...
####        #distances.append([datapoint_index,dist])
####        distances[row_index][0]=datapoint_index
####        distances[row_index][1]=dist
####    print(f"(local_calc) distances-> {distances}")
####    #pdb.set_trace()
#####    distances=[]
#####    for i in range(len(datapoints[0])):
#####        datapoint=datapoints.get_row(i)
#####        dist_row=[]
#####        for j in range(len(centroids[0])):
#####            centroid=centroids.get_row(j)
#####            dist=abs(np.linalg.norm(datapoint-centroid))
#####            dist_row.append(dist)
#####        distances.append(dist_row)

####    #assign each datapoint to the closest centroid (assign to clusters)
####    #cluster_assignment=[]
####    #pdb.set_trace()
####    num_of_centroids=np.count_nonzero(distances[:,0]==0)
####    print(f"(local_calc) num_of_centroids-> {num_of_centroids}")
####    num_of_datapoints=int(np.max(distances[:,0]))+1
####    datapoints_assignment_shape=(num_of_datapoints,2)
####    datapoints_assignment=np.zeros(datapoints_assignment_shape)
####    #print(f"(local_calc) datapoints_assignment-> {datapoints_assignment}")
####    for i in range(num_of_datapoints):
####        distances_per_datapoint_indices=np.where(distances[:,0]==i)
####        distances_per_datapoint=distances[distances_per_datapoint_indices]
####        min_distance=np.amin(distances[distances_per_datapoint_indices,1])
####        datapoints_assignment[i][0]=i
####        datapoints_assignment[i][1]=np.where(distances_per_datapoint[:,1]==min_distance)[0][0]
####    print(f"(local_calc) datapoints_assignment-> {datapoints_assignment}")
####    #pdb.set_trace()
####    

#####    for datapoint_distances in distances:
#####        min_index=np.where(datapoint_distances==np.amin(datapoint_distances))
#####        cluster_assignment.append(min_index[0][0])
####    to_return=np.zeros((num_of_centroids,num_of_dimensions+1))
####    counter=0
####    for i in range(len(datapoints_assignment)):
####        to_return[int(datapoints_assignment[i][1]),-1]+=1

####        row=centroids_data_cross_prod.get_row(i)
####        #print(f"(local_calc) row-> {row}")
####        coordinates=[row[col_index] for col_index in range(1,offset)]

####        to_return[int(datapoints_assignment[i][1]),:-1]+=coordinates
####    print(f"(local_calc) to_return-> {to_return}")
####    #pdb.set_trace()

####    to_return=customnumpy.to_ListOfNpArrays(to_return)

####    return to_return
#####---------------------------------------------------------
#####    to_return_num_cols=num_of_dimensions+1
#####    to_return_num_rows=centroids.get_num_rows()
#####    to_return=customnumpy.zeros(to_return_num_cols,to_return_num_rows)
#####    for row in range(len(cluster_assignment)):
#####        for col in range(num_of_dimensions):
#####            to_return[col][cluster_assignment[row]]+=datapoints[col][row]
#####        to_return[-1][cluster_assignment[row]]+=1#this keeps track of the number of points assigned to this cluster

#####    to_return=ListOfNpArrays([l.flatten() for l in to_return])
#####    return to_return

####def global_calc(num_of_clusters:int, local_centroids:List[ListOfNpArrays]) -> ListOfNpArrays:
#####    summed=local_centroids[0]
#####    for i in range(1,len(local_centroids)):
#####        summed=summed+local_centroids[i]
#####    summed=ListOfNpArrays(summed)
#####    print(f"(gloabal_calc) summed-> {summed}")
####    
####    averaged=[]
####    for row_index in range(local_centroids.get_num_rows()):
#####        pdb.set_trace()
####        row=local_centroids.get_row(row_index)
####        if row[-1]!=0:
####            averaged.append(row[0:-1]/row[-1])
####        else:
####            num_of_dimensions=len(local_centroids)-1
####            l=[np.random.uniform(min_val,max_val,size=(num_of_dimensions)) for i in range(num_of_clusters)]
####            pdb.set_trace()
####            pass

#####    pdb.set_trace()
####    to_return=customnumpy.to_ListOfNpArrays(averaged)
####    return to_return
####    #return ListOfNpArrays([l for l in tmp])
####    


#-----------------------------------------------------------------------------------------------------
###import numbers
###import numpy as np
###import numpy.lib.mixins

###from typing import List

###from ListOfNpArrays import ListOfNpArrays
###import ListOfNpArrays as customnumpy

###def generate_initial_centroids(num_of_dimensions:int, num_of_clusters:int, min_val:float, max_val:float) -> numpy.ndarray:
###    l=[np.random.uniform(min_val,max_val,size=(num_of_clusters)) for i in range(num_of_dimensions)]
###    initial_centroids=ListOfNpArrays(l)
###    return initial_centroids

###def local_calc(centroids:ListOfNpArrays, datapoints:ListOfNpArrays) -> ListOfNpArrays :
###    '''
###    parameters:
###        centroids: numpy array of shape Cx2, where C the number of clusters
###        datapoints: numpy array of shape Dx2, where D the number of datapoints
###    returns:
###        a numpy array of shape Cx3, where C teh number of clusters. The 2 columns correspond
###        to the sum of x and y coordinates of the datapoints assigned to each centroid and number of
###        datapoints assigned to each centroids. These will be used in the global step to calculate averages 
###    '''
###    num_of_dimensions=len(datapoints)
###    distances=[]
###    for i in range(len(datapoints[0])):
###        datapoint=datapoints.get_row(i)
###        dist_row=[]
###        for j in range(len(centroids[0])):
###            centroid=centroids.get_row(j)
###            dist=abs(np.linalg.norm(datapoint-centroid))
###            dist_row.append(dist)
###        distances.append(dist_row)

###    #assign each datapoint to the closest centroid (assign to clusters)
###    cluster_assignment=[] 
###    for datapoint_distances in distances:
###        min_index=np.where(datapoint_distances==np.amin(datapoint_distances))
###        cluster_assignment.append(min_index[0][0])
###    
###    to_return_num_cols=num_of_dimensions+1
###    to_return_num_rows=centroids.get_num_rows()
###    to_return=customnumpy.zeros(to_return_num_cols,to_return_num_rows)
###    for row in range(len(cluster_assignment)):
###        for col in range(num_of_dimensions):
###            to_return[col][cluster_assignment[row]]+=datapoints[col][row]
###        to_return[-1][cluster_assignment[row]]+=1#this keeps track of the number of points assigned to this cluster

###    to_return=ListOfNpArrays([l.flatten() for l in to_return])
###    return to_return

###def global_calc(num_of_clusters:int, local_centroids:List[ListOfNpArrays]) -> ListOfNpArrays:
###    '''
###    parameters:
###        num_of_clusters: the number of desired clusters
###        local_centroids: numpy array of shape NxCx3 where N:number of Nodes, C:number of Clusters. The last
###                        size 3 corresponds to x,y,number of occurences
###    returns:
###        the new centroids
###    '''
###    summed=local_centroids[0]
###    for i in range(1,len(local_centroids)):
###        summed=summed+local_centroids[i]
###    summed=ListOfNpArrays(summed)
###    tmp=summed[0:-1]/summed[-1]
###    return ListOfNpArrays([l for l in tmp])
###    
