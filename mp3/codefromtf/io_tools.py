"""Input and output helpers to load in data.
"""
import numpy as np

def read_dataset(path_to_dataset_folder,index_filename):
    """ Read dataset into numpy arrays with preprocessing included
    Args:
        path_to_dataset_folder(str): path to the folder containing samples and indexing.txt
        index_filename(str): indexing.txt
    Returns:
        A(numpy.ndarray): sample feature matrix A = [[1, x1], 
                                                     [1, x2], 
                                                     [1, x3],
                                                     .......] 
                                where xi is the 16-dimensional feature of each sample
            
        T(numpy.ndarray): class label vector T = [y1, y2, y3, ...] 
                             where yi is +1/-1, the label of each sample 
    """
    with open(path_to_dataset_folder+'/'+index_filename, 'r') as f:
        label_sample_path = f.readlines()
    T = np.array([max(0,float(label_sample_path[i].split(' ')[0])) for i in range(len(label_sample_path))])
    sample_path = [label_sample_path[i].split(' ')[1].replace('\n','') for i in range(len(label_sample_path))]
    
    A = []
    for i in range(len(sample_path)):
        with open(path_to_dataset_folder+'/'+sample_path[i], 'r') as f:
            row_data = f.read().strip().split('  ')
#             print(row_data)
            A.append([1.  if i ==0 else float(row_data[i-1]) for i in range(len(row_data)+1)])
    A = np.array(A)
    
        
    
    return A, T