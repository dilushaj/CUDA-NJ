from Bio.SubsMat import MatrixInfo
from Bio.Phylo.TreeConstruction import _Matrix, DistanceMatrix
from Bio.Align import MultipleSeqAlignment
import itertools
import sys
import numpy as np

# --- PyCUDA initialization
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


class DistanceCalculator(object):
    dna_alphabet = ['A', 'T', 'C', 'G']
    # BLAST nucleic acid scoring matrix
    blastn = [[5],
              [-4, 5],
              [-4, -4, 5],
              [-4, -4, -4, 5]]

    # transition/transversion scoring matrix
    trans = [[6],
             [-5, 6],
             [-5, -1, 6],
             [-1, -5, -5, 6]]

    protein_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y',
                        'Z']

    # matrices available
    dna_matrices = {'blastn': blastn, 'trans': trans}
    protein_models = MatrixInfo.available_matrices
    protein_matrices = {name: getattr(MatrixInfo, name)
                        for name in protein_models}

    dna_models = list(dna_matrices.keys())

    models = ['identity'] + dna_models + protein_models

    def __init__(self, model='identity', skip_letters=None):
        """Initialize with a distance model."""
        # Shim for backward compatibility (#491)
        if skip_letters:
            self.skip_letters = skip_letters
        elif model == 'identity':
            self.skip_letters = ()
        else:
            self.skip_letters = ('-', '*')

        if model == 'identity':
            self.scoring_matrix = None
        elif model in self.dna_models:
            self.scoring_matrix = _Matrix(self.dna_alphabet,
                                          self.dna_matrices[model])
        elif model in self.protein_models:
            self.scoring_matrix = self._build_protein_matrix(
                self.protein_matrices[model])
        else:
            raise ValueError("Model not supported. Available models: " +
                             ", ".join(self.models))

    ###################
    # iDivUp FUNCTION #
    ###################
    def iDivUp(a, b):
        return a // b + 1

    def _pairwise(self, seq1, seq2):
        """Calculate pairwise distance from two sequences (PRIVATE).
        Returns a value between 0 (identical sequences) and 1 (completely
        different, or seq1 is an empty string.)
        """

        score = 0
        max_score = 0
        if self.scoring_matrix:
            max_score1 = 0
            max_score2 = 0
            for i in range(0, len(seq1)):
                l1 = seq1[i]
                l2 = seq2[i]
                if l1 in self.skip_letters or l2 in self.skip_letters:
                    continue
                if l1 not in self.scoring_matrix.names:
                    raise ValueError("Bad alphabet '%s' in sequence '%s' at position '%s'"
                                     % (l1, seq1.id, i))
                if l2 not in self.scoring_matrix.names:
                    raise ValueError("Bad alphabet '%s' in sequence '%s' at position '%s'"
                                     % (l2, seq2.id, i))
                max_score1 += self.scoring_matrix[l1, l1]  # get the distance between two alphabets in score matrix
                max_score2 += self.scoring_matrix[l2, l2]
                score += self.scoring_matrix[l1, l2]
            # Take the higher score if the matrix is asymmetrical
            max_score = max(max_score1, max_score2)
        else:
            # Score by character identity, not skipping any special letters
            score = sum(l1 == l2
                        for l1, l2 in zip(seq1, seq2)
                        if l1 not in self.skip_letters and l2 not in self.skip_letters)
            max_score = len(seq1)
        if max_score == 0:
            return 1  # max possible scaled distance
        return 1 - (score * 1.0 / max_score)

    def get_distance(self, msa):

        if not isinstance(msa, MultipleSeqAlignment):
            raise TypeError("Must provide a MultipleSeqAlignment object.")
        i = 0
        for record in msa:
            record.index = i
            i += 1
        names = [record.id for record in msa]
        indices = [record.index for record in msa]

        dm = DistanceMatrix(names)
        pair_combinations = list(itertools.combinations(msa, 2))  # in order to combine take from here.
        combinations = len(pair_combinations)
        seqLength = len(pair_combinations[0][0])


        # host arrays
        host_combinations = []
        for pair in range(combinations):
            couple = ["%s" % (pair_combinations[pair][0].seq), "%s" % (pair_combinations[pair][1].seq)]
            host_combinations.extend(couple)

        host_names = self.scoring_matrix.names
        attributes = len(host_names)

        hst_scoring_matrix = []
        for name in host_names:
            sequence = self.scoring_matrix[name]
            hst_scoring_matrix.extend(sequence)

        host_scoring_matrix = np.array(hst_scoring_matrix)
        host_scoring_matrix = host_scoring_matrix.astype(np.float64)
        host_d_matrix = np.zeros((combinations,), dtype=float)
        host_d_matrix = host_d_matrix.astype(np.float64)
        host_names = np.asarray(host_names)
        host_combinations = np.asarray(host_combinations)

        ###GPU code
        start = cuda.Event()
        end = cuda.Event()

        # get the optimum block size based on dataset size
        if (combinations < 128):
            BLOCKSIZE = 128
        elif (combinations < 256):
            BLOCKSIZE = 256
        elif (combinations < 512):
            BLOCKSIZE = 512
        else:
            BLOCKSIZE = 1024

        # Allocate GPU device memory
        device_scoring_matrix = cuda.mem_alloc(host_scoring_matrix.nbytes)
        device_names = cuda.mem_alloc(sys.getsizeof(host_names))
        device_combinations = cuda.mem_alloc(sys.getsizeof(host_combinations))
        device_d_matrix = cuda.mem_alloc(host_d_matrix.nbytes)

        # Memcopy from host to device
        cuda.memcpy_htod(device_combinations, host_combinations)
        cuda.memcpy_htod(device_names, host_names)
        cuda.memcpy_htod(device_scoring_matrix, host_scoring_matrix)



        mod = SourceModule("""
          #include <stdio.h>
          #include <string.h>
          #include <stdlib.h>
          __global__ void DeviceDM(char device_combinations[] , char device_names[], int n,  int N, const int seqLength, double *device_scoring_matrix, double *device_d_matrix)
          {
            const int tid = threadIdx.y + blockIdx.y* blockDim.y;
            if (tid >= N) return;



            int start1= (tid*2)*(seqLength);
            int start2= (tid*2+1)*(seqLength);

            char skip_letters[] = {'-', '*'};

            int score = 0;
            int max_score = 0;
            
            if(device_scoring_matrix){
                double max_score1 = 0.0;
                double max_score2 = 0.0;
                
                for(int i=0; i < seqLength; i++){
                    char l1 = device_combinations[start1+i];
                    char l2 = device_combinations[start2+i];
                    int l1rank = 0;
                    int l2rank = 0;
                    if(!(l1==skip_letters[0] || l1==skip_letters[1] || l2==skip_letters[0] || l2==skip_letters[1])){
                        for(int i=0; i< n; i++){
                            if(l1==device_names[i]){
                                l1rank=i;
                            }
                            if(l2==device_names[i]){
                                l2rank=i;
                            }
                            if(l1rank!=0 && l2rank!=0){
                                break;
                            }                                   
                        }

                        max_score1 = max_score1 + device_scoring_matrix[l1rank* n + l1rank];
                        max_score2 = max_score2 + device_scoring_matrix[l2rank* n + l2rank];
                        score += device_scoring_matrix[l1rank*n + l2rank];


                    }

                }
               
                if(max_score1>=max_score2){
                    max_score= max_score1;	
                }else{
                    max_score= max_score2;
                }

            }else{
                for(int i=0; i < seqLength; i++){
                    char l1 = device_combinations[start1+i];
                    char l2 = device_combinations[start2+i];
                    if(!(l1==skip_letters[0] || l1==skip_letters[1] || l2==skip_letters[0] || l2==skip_letters[1])){
                        if(l1==l2){
                            score= score + 1;
                        }
                    }

                }
                max_score = seqLength;   

             }   
            if(max_score == 0){
                device_d_matrix[tid]=1;
            }else{
                device_d_matrix[tid]=1 - (score * 1.0 / max_score);
            }


          } 
        """)

        # --- Define a reference to the __global__ function and call it 1 - (score * 1.0 / max_score);
        DeviceDM = mod.get_function("DeviceDM")
        blockDim = (1, BLOCKSIZE, 1)
        gridDim = (1, combinations / BLOCKSIZE + 1, 1)

        start.record()

        DeviceDM(device_combinations, device_names, np.int32(attributes), np.int32(combinations), np.int32(seqLength),
                 device_scoring_matrix, device_d_matrix, block=blockDim, grid=gridDim)
        end.record()
        end.synchronize()
        secs = start.time_till(end) * 1e-3
        #print("Processing time = %fs" % (secs))
        distance_matrix_list = np.empty_like(host_d_matrix)
        cuda.memcpy_dtoh(distance_matrix_list, device_d_matrix)
        device_d_matrix.free()
        device_combinations.free()
        device_names.free()
        device_scoring_matrix.free()
        final_distance_matrix = distance_matrix_list.tolist()

        for pair in range(combinations):
            dm[pair_combinations[pair][0].id, pair_combinations[pair][1].id] = final_distance_matrix[pair]

        return dm

    def _build_protein_matrix(self, subsmat):
        """Convert matrix from SubsMat format to _Matrix object (PRIVATE)."""
        protein_matrix = _Matrix(self.protein_alphabet)
        for k, v in subsmat.items():
            aa1, aa2 = k
            protein_matrix[aa1, aa2] = v
        return protein_matrix
