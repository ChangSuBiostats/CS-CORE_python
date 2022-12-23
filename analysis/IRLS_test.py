'''
==========
Test the Python implementation of CS-CORE IRLS
==========
'''
import numpy as np
import time
from CSCORE.CSCORE_IRLS import CSCORE_IRLS

# Test 1: a pair of independent data
# use ind_gene_pair from the R CS-CORE package

counts = np.genfromtxt('data/ind_gene_pair_counts.csv',
                       delimiter=',', skip_header=1)[:,1:]
seq_depth = np.genfromtxt('data/ind_gene_pair_seq_depth.csv',
                       delimiter=',', skip_header=1)[:,1]

print(CSCORE_IRLS(counts, seq_depth))

# Test 2: B cell
# use B cells' data as generated in https://changsubiostats.github.io/CS-CORE/articles/CSCORE.html
# saved by
# ```{r}
# write.csv(pbmc_B_healthy$nCount_RNA,
#           'data/pbmc_seq_depth.csv')
# pbmc_counts <- GetAssayData(pbmc_B_healthy, slot = 'counts') %>% as.matrix %>% t
# write.csv(pbmc_counts[, genes_selected],
#             'data/pbmc_counts.csv')
# ```

counts = np.genfromtxt('data/pbmc_counts.csv',
                       delimiter=',', skip_header=1)[:, 1:]
seq_depth = np.genfromtxt('data/pbmc_seq_depth.csv',
                       delimiter=',', skip_header=1)[:, 1]

B_cell_result = CSCORE_IRLS(counts, seq_depth)
np.savetxt('data/pbmc_python_est.txt', B_cell_result[0])
np.savetxt('data/pbmc_python_p_value.txt', B_cell_result[1])
