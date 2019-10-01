import copy
from Bio.Phylo import BaseTree
from Bio._py3k import zip, range
from Bio.Phylo.TreeConstruction import DistanceMatrix
import time
import numpy as np

# pycuda initialization
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule



class NJTree:
    def nj(self, distance_matrix):
        if not isinstance(distance_matrix, DistanceMatrix):
            raise TypeError("Must provide a DistanceMatrix object.")

        # make a copy of the distance matrix to be used
        dm = copy.deepcopy(distance_matrix)
        # init terminal clades
        clades = [BaseTree.Clade(None, name) for name in dm.names]
        # init node distance
        node_dist = [0] * len(dm)
        # init minimum index
        min_i = 0
        min_j = 0
        inner_count = 0
        total_time = 0
        total_time2 = 0
        # special cases for Minimum Alignment Matrices
        if len(dm) == 1:
            root = clades[0]

            return BaseTree.Tree(root, rooted=False)
        elif len(dm) == 2:
            # minimum distance will always be [1,0]
            min_i = 1
            min_j = 0
            clade1 = clades[min_i]
            clade2 = clades[min_j]
            clade1.branch_length = dm[min_i, min_j] / 2.0
            clade2.branch_length = dm[min_i, min_j] - clade1.branch_length
            inner_clade = BaseTree.Clade(None, "Inner")
            inner_clade.clades.append(clade1)
            inner_clade.clades.append(clade2)
            clades[0] = inner_clade
            root = clades[0]

            return BaseTree.Tree(root, rooted=False)

        mod = SourceModule("""
          #include <stdio.h>
          #include <stdlib.h>
          __global__ void DeviceNodeDist(double *device_dm, double *device_node_dist, int N)
          {
              const int tid = threadIdx.y + blockIdx.y* blockDim.y;
              if (tid >= N) return;
              for(int i = 0; i< N; i++){
                if(tid< i){
                    device_node_dist[tid] += device_dm[(i*(i+1))/2 + tid];

                }else{
                    device_node_dist[tid] += device_dm[(tid*(tid+1))/2 + i];

                }

               }

          device_node_dist[tid]= (double)(device_node_dist[tid]/ (N-2));
          }""")

        mod1 = SourceModule("""
         __global__ void findMin(double *dm, double *node_dist, long long *index_x, long long *index_y, double *local_min, int c, int l, int dm_length)
        {
            int k = threadIdx.y + blockIdx.y*blockDim.y;

            double min_dist = 0.0;  

            int min_x =0;
            int min_y =0;
            int x = 0;
            int y = 0;

            for(int i= k*c ; i< (k+1)*c; i++){
                if(i<l)
                {
                    for(int j=0; j<dm_length; j++){
                        if(i==0){
                            x=1;
                            y=0;
                            break;
                        }else{
                            int t_val = ((j+1)*(j+2))/2 ;
                            if(i < t_val){
                                x=j+1;
                                y= i-(t_val-j-1);
                                break;
                            }else if(i== t_val){
                                x = j+2;
                                y = 0;
                                break;
                            }

                        }
                    }


                    double temp = dm[i] - (node_dist[x] + node_dist[y] );
                    if(min_dist > temp){
                        min_dist = temp;
                        min_x = x;
                        min_y = y;
                    }
                } 
            }
            local_min[k]=min_dist;
            index_x[k]= min_x;
            index_y[k]= min_y;

        }""")
        # print("Time taken to run SourceModule %s" % (time.time()-in_t1))
        while len(dm) > 2:


            # calculate nodeDist
            host_dm = []  # 1D list for distance matrix
            for list in dm.matrix:
                host_dm.extend(list)

            host_dm = np.array(host_dm)
            # host_dm = host_dm.astype(np.float32)
            length = len(dm)
            host_node_dist = np.zeros((length,), dtype=float)
            # host_node_dist = host_node_dist.astype(np.float32)

            ###GPU code
            start = cuda.Event()
            end = cuda.Event()

            # get the optimum block size based on dataset size
            if (length < 128):
                BLOCKSIZE = 128
            elif (length < 256):
                BLOCKSIZE = 256
            elif (length < 512):
                BLOCKSIZE = 512
            else:
                BLOCKSIZE = 1024

            ###Allocate GPU device memory
            device_dm = cuda.mem_alloc(host_dm.nbytes)
            device_node_dist = cuda.mem_alloc(host_node_dist.nbytes)

            ###Memcopy from host to device
            cuda.memcpy_htod(device_dm, host_dm)

            DeviceNodeDist = mod.get_function("DeviceNodeDist")

            blockDim = (1, BLOCKSIZE, 1)
            gridDim = (1, length / BLOCKSIZE + 1, 1)

            start.record()

            DeviceNodeDist(device_dm, device_node_dist, np.int32(length), block=blockDim, grid=gridDim)
            end.record()
            end.synchronize()

            node_dist1 = np.empty_like(host_node_dist)
            cuda.memcpy_dtoh(node_dist1, device_node_dist)
            node_dist2 = node_dist1.tolist()
            node_dist[0:len(node_dist2)] = node_dist2

            device_dm.free()







            in_t2 = time.time()

            start1 = cuda.Event()
            end1 = cuda.Event()

            mat = dm.matrix
            dm_cpu = np.array(mat[1][:-1])
            for i in range(2, len(dm)):
                dm_cpu = np.append(dm_cpu, mat[i][:-1])

            combinations = int(((len(dm) - 1) * len(dm)) / 2)

            if combinations < 1024 * 128:
                block_size = int(round((len(dm)) / 2))
            else:
                block_size = 512

            local_count = int(round(combinations / block_size))
            index_x = np.zeros(block_size, dtype=int)
            index_y = np.zeros(block_size, dtype=int)
            min_val = np.zeros(block_size, dtype=float)


            local_min_array_gpu = cuda.mem_alloc(dm_cpu.nbytes)
            local_index_gpux = cuda.mem_alloc(index_x.nbytes)
            local_index_gpuy = cuda.mem_alloc(index_y.nbytes)
            local_min_gpu = cuda.mem_alloc(min_val.nbytes)

            cuda.memcpy_htod(local_min_array_gpu, dm_cpu)

            func = mod1.get_function("findMin")
            start1.record()
            func(local_min_array_gpu, device_node_dist, local_index_gpux, local_index_gpuy, local_min_gpu,
                 np.int32(local_count), np.int32(len(dm_cpu)), np.int32(len(dm)),
                 block=(1, block_size, 1))
            end1.record()
            end1.synchronize()

            cuda.memcpy_dtoh(min_val, local_min_gpu)
            cuda.memcpy_dtoh(index_x, local_index_gpux)
            cuda.memcpy_dtoh(index_y, local_index_gpuy)



            min_val_new = min_val.tolist()


            local_min_array_gpu.free()
            local_min_gpu.free()
            local_index_gpux.free()
            local_index_gpuy.free()
            device_node_dist.free()

            min_dist = min(min_val_new)

            for i in range(len(min_val)):
                if min_dist == min_val[i]:
                    min_i = index_x[i]
                    min_j = index_y[i]
                    break

            del host_dm
            del host_node_dist
            del dm_cpu

            total_time2 += time.time() - in_t2








            # create clade
            clade1 = clades[min_i]
            clade2 = clades[min_j]
            inner_count += 1
            inner_clade = BaseTree.Clade(None, "Inner" + str(inner_count))
            inner_clade.clades.append(clade1)
            inner_clade.clades.append(clade2)
            # assign branch length
            clade1.branch_length = (dm[min_i, min_j] + node_dist[min_i] -
                                    node_dist[min_j]) / 2.0
            clade2.branch_length = dm[min_i, min_j] - clade1.branch_length

            # update node list
            clades[min_j] = inner_clade
            del clades[min_i]

            # rebuild distance matrix,
            # set the distances of new node at the index of min_j
            for k in range(0, len(dm)):
                if k != min_i and k != min_j:
                    dm[min_j, k] = (dm[min_i, k] + dm[min_j, k] -
                                    dm[min_i, min_j]) / 2.0

            dm.names[min_j] = "Inner" + str(inner_count)
            del dm[min_i]

        #print("Time taken for min dist node calculation= %s" % total_time2)
        # set the last clade as one of the child of the inner_clade
        root = None
        if clades[0] == inner_clade:
            clades[0].branch_length = 0
            clades[1].branch_length = dm[1, 0]
            clades[0].clades.append(clades[1])
            root = clades[0]
        else:
            clades[0].branch_length = dm[1, 0]
            clades[1].branch_length = 0
            clades[1].clades.append(clades[0])
            root = clades[1]

        return BaseTree.Tree(root, rooted=False)

    def _height_of(self, clade):
        """Calculate clade height -- the longest path to any terminal (PRIVATE)."""
        height = 0
        if clade.is_terminal():
            height = clade.branch_length
        else:
            height = height + max(self._height_of(c) for c in clade.clades)
        return height


