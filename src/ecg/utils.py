# Any helper functions
def flatten_list(list_of_lists):
    import numpy as np
    return np.concatenate(list_of_lists).tolist() if list_of_lists else []