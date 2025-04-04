import umap
import sklearn.decomposition
import sklearn.cluster
import sklearn.neighbors
from umap.utils import submatrix, average_nn_distance
from scipy.signal import hilbert
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.signal import butter, sosfiltfilt


import numpy as np


def _nhood_search(umap_object, nhood_size):
    if hasattr(umap_object, "_small_data") and umap_object._small_data:
        dmat = sklearn.metrics.pairwise_distances(umap_object._raw_data)
        indices = np.argpartition(dmat, nhood_size)[:, :nhood_size]
        dmat_shortened = submatrix(dmat, indices, nhood_size)
        indices_sorted = np.argsort(dmat_shortened)
        indices = submatrix(indices, indices_sorted, nhood_size)
        dists = submatrix(dmat_shortened, indices_sorted, nhood_size)
    else:
        rng_state = np.empty(3, dtype=np.int64)
        indices, dists = umap_object._knn_search_index.query(
            umap_object._raw_data,
            k=nhood_size,
        )
    return indices, dists


def calculate_local_dimension(model, data=None, n_neighbors=None, threshold=0.7, n_comp=3):
    """
    Calculate the local dimensionality and the first three principal components of the data.

    Parameters:
    model : object
        The model object containing the data and the k-nearest neighbors search index.
    data : array-like, optional
        The data for which to calculate the local dimensionality. If None, the model's raw data is used.
    n_neighbors : int, optional
        The number of neighbors to use in the k-nearest neighbors search. If None, the model's default is used.
    threshold : float, optional
        The threshold for the cumulative explained variance ratio to determine the local dimensionality. Default is 0.7.
    n_comp : int, optional
        The number of principal components to return. Default is 3.

    Returns:
    local_dim : ndarray
        An array containing the local dimensionality for each data point.
    three_comp : ndarray
        An array containing the first three principal components for each data point.
    """

    if data is None:
        data = model._raw_data

    if n_neighbors is None:
        n_neighbors = model.n_neighbors

    highd_indices, highd_dists = _nhood_search(model, n_neighbors)
    local_dim = np.empty(data.shape[0], dtype=np.int64)
    three_comp = np.empty((data.shape[0], n_comp), dtype=np.float64)

    for i in tqdm(range(data.shape[0])):
        d = data[highd_indices[i]]
        pca = PCA()
        compdata = pca.fit_transform(d)
        local_dim[i] = np.where(np.cumsum(pca.explained_variance_ratio_) > threshold)[0][0]

    return local_dim


def get_vicinity_elements(matrix, row, col, radius=1):
    """
    Returns the elements around a given index in a matrix within specified start and end radii.

    Parameters:
    matrix (np.ndarray): The input matrix.
    row (int): The row index of the target element.
    col (int): The column index of the target element.
    radius (int): The radius around the target element to include.

    Returns:
    np.ndarray: The elements around the target index within the specified radii.
    """

    nrow, ncol = matrix.shape

    row_start = max(0, row - 1 * radius)
    row_end = min(nrow, row + 1 * radius + 1)
    col_start = max(0, col - 1 * radius)
    col_end = min(ncol, col + 1 * radius + 1)

    if row_start == 0:
        row_end = 2 * radius + 1
    elif row_end == nrow:
        row_start = row_end - 2 * radius - 1
    elif col_start == 0:
        col_end = 2 * radius + 1
    elif col_end == ncol:
        col_start = col_end - 2 * radius - 1

    if row_start < 0 or row_end > nrow or col_start < 0 or col_end > ncol:
        raise ValueError("The specified radius is too large for the given matrix dimensions.")
    
    return matrix[row_start:row_end, col_start:col_end]


def calculate_envelope(signal):
    """
    Calculate the envelope of a signal using the Hilbert transform.
    
    Parameters:
    signal (numpy.ndarray): Input signal.
    
    Returns:
    numpy.ndarray: Envelope of the input signal.
    """
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    return envelope
    


def calculate_entropy_in_umap(model, data=None, n_neighbors=None, threshold=0.7):
    """
    Calculate the local dimensionality and the first three principal components of the data.

    Parameters:
    model : object
        The model object containing the data and the k-nearest neighbors search index.
    data : array-like, optional
        The data for which to calculate the local dimensionality. If None, the model's raw data is used.
    n_neighbors : int, optional
        The number of neighbors to use in the k-nearest neighbors search. If None, the model's default is used.
    threshold : float, optional
        The threshold for the cumulative explained variance ratio to determine the local dimensionality. Default is 0.7.
    n_comp : int, optional
        The number of principal components to return. Default is 3.

    Returns:
    local_dim : ndarray
        An array containing the local dimensionality for each data point.
    three_comp : ndarray
        An array containing the first three principal components for each data point.
    """

    if data is None:
        data = model._raw_data

    if n_neighbors is None:
        n_neighbors = model.n_neighbors

    highd_indices, highd_dists = _nhood_search(model, n_neighbors)
    entropy = np.empty(data.shape[0])
    eigen_values = np.empty((data.shape[0], n_neighbors), dtype=np.float64)


    for i in tqdm(range(data.shape[0])):
        
        U, s, V = np.linalg.svd(data[highd_indices[i]], full_matrices=False)
        #s = s / np.max(s)
        entropy[i] =calculate_shannon_entropy(s)        
        eigen_values[i] = s

    return entropy
    

def calculate_shannon_entropy(data):
    """
    Calculate the Shannon entropy of a dataset.

    Parameters:
    data (numpy.ndarray): The input data for which to calculate the entropy.

    Returns:
    float: The Shannon entropy of the input data.
    """
    
    # Normalize the data
    data = data / np.sum(data)
    
    # Calculate entropy
    entropy = -np.sum(data * np.log(data + 1e-10))  # Adding a small value to avoid log(0)
    
    return entropy


def calculate_std(model, radii):
    f"""
    Calculate the standard deviation of the given model.

    Parameters:
    model (numpy.ndarray): The model for which to calculate the standard deviation.

    Returns:
    float: The standard deviation of the model.
    """
    
    model_shape_i, model_shape_j = model.shape
    vs_std = np.zeros_like(model)
    
    for i in tqdm(range(model_shape_i)):
        for j in range(model_shape_j):
            
            vicinity_ij = get_vicinity_elements(model, i, j, radii).reshape(-1)
            data_selected = model.reshape(-1)[vicinity_ij]
            vs_std[i, j] = np.std(data_selected)/ np.mean(data_selected)
            
    return vs_std


def calculate_local_entropy_in_model(data, modelshape, radii):
    """
    Calculate the standard deviation of the given model.

    Parameters:
    model (numpy.ndarray): The model for which to calculate the standard deviation.

    Returns:
    float: The standard deviation of the model.
    """

    model_shape_i, model_shape_j = modelshape

    model = np.arange(model_shape_i* model_shape_j).reshape(model_shape_i, model_shape_j)
    local_entropy = np.zeros([model_shape_i, model_shape_j], dtype=np.float64)
    
    for i in tqdm(range(model_shape_i)):
        for j in range(model_shape_j):
            
            vicinity_ij = get_vicinity_elements(model, i, j, radii).reshape(-1)
            data_selected = data[vicinity_ij]
            
            U, s, V = np.linalg.svd(data_selected, full_matrices=False)
            s = s / np.max(s)
            
            local_entropy[i, j] = calculate_shannon_entropy(s)
            
    return local_entropy


def calculate_local_dimensionality_in_model(data, modelshape, radii, threshold):
    """
    Calculate the standard deviation of the given model.

    Parameters:
    model (numpy.ndarray): The model for which to calculate the standard deviation.

    Returns:
    float: The standard deviation of the model.
    """
    
    model_shape_i, model_shape_j = modelshape

    model = np.arange(model_shape_i* model_shape_j).reshape(model_shape_i, model_shape_j)
    local_dim = np.zeros_like(model)

    for i in tqdm(range(model_shape_i)):
        for j in range(model_shape_j):

            vicinity_ij = get_vicinity_elements(model, i, j, radii).reshape(-1)
            data_selected = data[vicinity_ij]
            pca = PCA()
            pca.fit_transform(data_selected)
            local_dim[i, j] = len(np.where(np.cumsum(pca.explained_variance_ratio_) < threshold)[0])
            
    return local_dim


def apply_sosfilter(data, freq, sr, order=10, filter_type='lp'):
    """
    Apply a second-order sections (SOS) filter to the input data.

    Parameters:
    data (numpy.ndarray): The input data to be filtered.
    freq (float or tuple): The frequency for the filter. For 'lp' and 'hp', provide a single float. 
                           For 'bp', provide a tuple (low_freq, high_freq).
    sr (float): The sampling rate of the data.
    order (int): The order of the filter. Default is 10.
    filter_type (str): The type of filter to apply. Options are 'lp' (low-pass), 'hp' (high-pass), 
                       and 'bp' (band-pass). Default is 'lp'.

    Returns:
    numpy.ndarray: The filtered data.
    """
    if filter_type == 'lp':
        sos = butter(order, freq, 'lp', fs=sr, output='sos')
    elif filter_type == 'hp':
        sos = butter(order, freq, 'hp', fs=sr, output='sos')
    elif filter_type == 'bp':
        sos = butter(order, freq, 'bp', fs=sr, output='sos')
    else:
        raise ValueError("Invalid filter_type. Options are 'lp', 'hp', and 'bp'.")
    
    filtered_data = sosfiltfilt(sos, data, axis=-1)

    return filtered_data

