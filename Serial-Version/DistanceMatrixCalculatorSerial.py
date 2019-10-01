from Bio.SubsMat import MatrixInfo
from Bio.Phylo.TreeConstruction import _Matrix, DistanceMatrix
from Bio.Align import MultipleSeqAlignment
import itertools





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

                max_score1 += self.scoring_matrix[l1, l1] # get the distance between two alphabets in score matrix
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
        i=0
        for record in msa:
            record.index= i
            i+=1
        names = [record.id for record in msa]


        dm = DistanceMatrix(names)
        pair_combinations = list(itertools.combinations(msa, 2))




        for pair in range(len(pair_combinations)):
            dm[pair_combinations[pair][0].id, pair_combinations[pair][1].id] = self._pairwise(pair_combinations[pair][0], pair_combinations[pair][1])
        return dm



    def _build_protein_matrix(self, subsmat):
        """Convert matrix from SubsMat format to _Matrix object (PRIVATE)."""
        protein_matrix = _Matrix(self.protein_alphabet)
        for k, v in subsmat.items():
            aa1, aa2 = k
            protein_matrix[aa1, aa2] = v
        return protein_matrix