from Bio import AlignIO;
from Bio import Phylo;
from NJ_treev2 import NJTree;
from DistanceMatrixCalculatorGPU import DistanceCalculator
import time

class Tree_Generation:
    def PhyloTree(self, dm): # NJ tree generation
        genNJ = NJTree()
        NJ = genNJ.nj(dm)
        return NJ;

    def calculate_distance_matrix(self, type):# distance matrix calculation
        if type == 'DNA':
            matrix_type = 'blastn'
        else:
            matrix_type = 'blosum62'
        calculator = DistanceCalculator(matrix_type)
        aln = AlignIO.read(open('../Datasets/DNA/n500dna.fas'), 'fasta')
        dm = calculator.get_distance(aln)
        return dm

in_t= time.time()
gentree= Tree_Generation()
dis_matrix=gentree.calculate_distance_matrix('DNA')
print("time taken for distance matrix construction= %s"% (time.time()-in_t))
s_time= time.time()
NJ=gentree.PhyloTree(dis_matrix)
Phylo.draw_ascii(NJ)
print("time taken fot nj tree construction= %s"% (time.time()-s_time))
print("total time taken to NJ tree construction= %s"% (time.time()-in_t))