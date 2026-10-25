import gzip
import numpy as np
from src.utils.paths import get_data_dir

def load_gse2034() -> tuple[np.ndarray, np.ndarray]:
    """Load GSE2034 dataset from series matrix.
    
    Returns:
        X: Microarray expression matrix (Samples x Genes)
        y: Bone relapse labels (1 = yes, 0 = no)
    """
    file_path = get_data_dir() / "GSE2034_series_matrix.txt.gz"
    if not file_path.exists():
        file_path = get_data_dir() / "GSE2034_series_matrix.txt"
        if not file_path.exists():
            raise FileNotFoundError("GSE2034 data not found in data/raw/")
            
    labels = []
    data_lines = []
    in_data = False
    
    open_fn = gzip.open if file_path.suffix == ".gz" else open
    mode = "rt" if file_path.suffix == ".gz" else "r"
    
    with open_fn(file_path, mode, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith("!Sample_characteristics_ch1"):
                if "bone relapses" in line:
                    parts = line.split('\t')[1:]
                    for p in parts:
                        p = p.strip('"')
                        if "1=yes" in p:
                            val = int(p.split(":")[-1].strip())
                            labels.append(val)
            elif line.startswith("!series_matrix_table_begin"):
                in_data = True
            elif line.startswith("!series_matrix_table_end"):
                in_data = False
            elif in_data:
                if line.startswith('"ID_REF"'):
                    continue
                parts = line.split('\t')
                gene_expr = [float(x) if x and x != 'null' else np.nan for x in parts[1:]]
                data_lines.append(gene_expr)
                
    if not labels:
        raise ValueError("Could not parse labels from GSE2034 file.")
        
    X = np.array(data_lines).T
    y = np.array(labels)
    
    # Missing values are imputed within the training-only preprocessing pipeline.
        
    return X, y
