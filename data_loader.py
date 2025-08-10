import math
import random
import numpy as np
import scipy.io as scio
from sklearn.model_selection import KFold
import pandas as pd


def compute_disease_simlarity_matrix(similarity_matrix, top_k):
    """
    Compute the top-k disease similarity matrix

    Parameters:
        similarity_matrix (ndarray): A square matrix representing pairwise disease similarities
        top_k (int): The number of top similar diseases to retain for each disease
    
    Returns:
        ndarray: A matrix where each row retains only the top_k most similar diseases with their similarity scores,
                and other values are set to zero
    """
    num_diseases = similarity_matrix.shape[0]
    result_matrix = np.zeros((num_diseases, num_diseases), dtype=np.float32)

    for i in range(num_diseases):
        # Get the similarity scores with disease i (excluding iteself)
        similarities = [(j, similarity_matrix[i, j]) for j in range(num_diseases) if i != j]

        # Sort the similarities in descending order
        top_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

        # Retain only the top_k similarities
        for j, sim_value in top_similar:
            result_matrix[i, j] = sim_value
    
    return result_matrix


def compute_drug_similarity_matrix(similarity_matrix, top_k):
    """
    Compute the top-k drug similarity matrix

    Parameters:
        similarity_matrix (ndarray): A square matrix representing pairwise drug similarities
        top_k (int): The number of top similar drugs to retain for each drug

    Returns:
        ndarray: A matrix where each row retains only the top_k most similar drugs with their similarity scores,
                and other values are set to zero
    """
    num_drugs = similarity_matrix.shape[0]
    result_matrix = np.zeros((num_drugs, num_drugs), dtype=np.float32)

    for i in range(num_drugs):
        # Filter out self-similarity
        similarities = [(j, similarity_matrix[i, j]) for j in range(num_drugs) if i != j]

        # Sort similarities in descending order
        top_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

        # Retain only the top_k similarities
        for j, sim_value in top_similar:
            result_matrix[i, j] = sim_value

    return result_matrix       


def load_mat_data(file_path):
    """
    Load LRSSL dataset with drug and disease similarity matrices

    Parameters:
        file_path (str): Path to the .mat data file

    Returns:
        Tuple:
            drug_similarity (ndarray): Drug similarity matrix
            disease_similarity (ndarray): Disease similarity matrix
            drug_names (ndarray): Array of drug names
            num_drugs (int): Total number of drugs
            disease_names (ndarray): Array of disease names
            num_diseases (int): Total number of diseases
            interaction_matrix (ndarray): Drug-disease interaction matrix
    """
    data = scio.loadmat(file_path)

    # Extract and convert data types
    drug_similarity = data["drug"].astype(float)
    disease_similarity = data["disease"].astype(float)

    # Process name information
    drug_names = np.ravel(data["Wrname"])
    disease_names = np.ravel(data["Wdname"])

    # Calculate the number of drugs and diseases
    num_drugs = len(drug_names)
    num_diseases = len(disease_names)

    # Extract interaction data
    interaction_matrix = data["didr"]

    return drug_similarity, disease_similarity, drug_names, num_drugs, disease_names, num_diseases, interaction_matrix


def load_lrssl_data(directory_path, use_reduced=True):
    """
    Load LRSSL dataset with drug and disease similarity matrices
    
    Parameters:
        directory_path (str): Path to the data files
        use_reduced (bool): Whether to average drug similarity sources

    Returns:
        Tuple of drug_sim, disease_sim, drug_names, drug_count,
        disease_names, disease_count, and interaction_matrix
    """
    # Load drug similarity data
    sim_chemical = pd.read_csv(f"{directory_path}lrssl_simmat_dc_chemical.txt", sep="\t", index_col=0)
    sim_domain = pd.read_csv(f"{directory_path}lrssl_simmat_dc_domain.txt", sep="\t", index_col=0)
    sim_go = pd.read_csv(f"{directory_path}lrssl_simmat_dc_go.txt", sep="\t", index_col=0)

    # Load disease similarity data
    disease_similarity = pd.read_csv(f"{directory_path}lrssl_simmat_dg.txt", sep="\t", index_col=0)

    # Merge drug similarity sources based on the parameter
    drug_similarity = (sim_chemical + sim_domain + sim_go) / 3 if use_reduced else sim_chemical

    # Load drug-disease interaction matrix
    interaction_matrix = pd.read_csv(f"{directory_path}lrssl_admat_dgc.txt", sep="\t", index_col=0).T

    # Convert to NumPy arrays
    drug_similarity_np = drug_similarity.to_numpy(dtype=np.float32)
    interaction_matrix_np = interaction_matrix.to_numpy(dtype=np.float32)
    disease_similarity_np = disease_similarity.to_numpy(dtype=np.float32).T

    # Extract drug and disease names
    drug_names = drug_similarity.columns.to_numpy().reshape(-1)
    disease_names = disease_similarity.columns.to_numpy().reshape(-1)

    # Count the number of drugs and diseases
    num_drugs = len(drug_names)
    num_diseases = len(disease_names)

    return (
        drug_similarity_np,
        disease_similarity_np,
        drug_names,
        num_drugs,
        disease_names,
        num_diseases,
        interaction_matrix_np
    )


def load_ldataset(data_path):
    """
    Load L-dataset containing drug and disease similarity matrices

    Parameters:
        data_path (str): Path to the dataset files
    
    Return:
        Tuple containing drug similarity matrix, disease similarity matrix,
        drug names, number of drugs, disease names, number of diseases, 
        and the interaction matrix
    """
    # Load disease similarity matrix, drug-disease interaction matrix, drug similarity matrix
    disease_similarity = pd.read_csv(f"{data_path}dis_sim.csv", header=None).to_numpy(dtype=np.float32)
    interaction_matrix = pd.read_csv(f"{data_path}drug_dis.csv", header=None).to_numpy(dtype=np.float32)
    drug_similarity = pd.read_csv(f"{data_path}drug_sim.csv", header=None).to_numpy(dtype=np.float32)

    # Generate drug and disease name indices
    disease_names = np.arange(disease_similarity.shape[0])
    drug_names = np.arange(drug_similarity.shape[0])

    # Calculate the number of drugs and diseases
    num_diseases = disease_similarity.shape[0]
    num_drugs = drug_similarity.shape[0]

    return (
        drug_similarity,
        disease_similarity,
        drug_names,
        num_drugs,
        disease_names,
        num_diseases,
        interaction_matrix.T
    )


def data_preparation(args):
    """
    Prepare data for drug-disease association prediction

    Parameters:
        args: Arguments including dataset name, number of splits, topK values, etc
    
    Return:
        Tuple containing disease similarity matrix, drug similarity matrix,
        true interaction labels, training data, testing data, and positive weight
    """
    dataset_path = f"./dataset/{args.dataset}/"
    assert args.dataset in ["Cdataset", "Fdataset", "Ldataset", "LRSSL"], "Invalid dataset name."

    # Load dataset based on dataset type
    if args.dataset in ["Fdataset", "Cdataset"]:
        drug_sim, disease_sim, drug_names, drug_count, disease_names, disease_count, interactions = load_mat_data(
            f"{dataset_path}{args.dataset}.mat"
        )
    elif args.dataset == "LRSSL":
        drug_sim, disease_sim, drug_names, drug_count, disease_names, disease_count, interactions = load_lrssl_data(
            dataset_path
        )
    else:
        drug_sim, disease_sim, drug_names, drug_count, disease_names, disease_count, interactions = load_ldataset(dataset_path)
    
    # Dataset details
    args.n_diseases, args.n_drugs = interactions.shape

    # K-Fold cross-validation setup
    kfold = KFold(n_splits=args.n_splits, shuffle=True)
    pos_row, pos_col = np.nonzero(interactions)
    neg_row, neg_col = np.nonzero(1 - interactions)

    assert len(pos_row) + len(neg_row) == np.prod(interactions.shape), "Mismatch in positive and negative samples."

    # Split data into training and testing sets
    train_data, test_data = [], []
    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kfold.split(pos_row), kfold.split(neg_row)):
        train_mask = np.zeros_like(interactions, dtype=bool)
        test_mask = np.zeros_like(interactions, dtype=bool)

        # Positive and negative edges for training and testing
        train_pos_edge = np.stack([pos_row[train_pos_idx], pos_col[train_pos_idx]])
        train_neg_edge = np.stack([neg_row[train_neg_idx], neg_col[train_neg_idx]])
        test_pos_edge = np.stack([pos_row[test_pos_idx], pos_col[test_pos_idx]])
        test_neg_edge = np.stack([neg_row[test_neg_idx], neg_col[test_neg_idx]])

        # Combine positive and negative edges
        train_edge = np.concatenate([train_pos_edge, train_neg_edge], axis=1)
        test_edge = np.concatenate([test_pos_edge, test_neg_edge], axis=1)

        # Create masks for training and testing data
        train_mask[train_edge[0], train_edge[1]] = True
        test_mask[test_edge[0], test_edge[1]] = True

        train_data.append(train_mask)
        test_data.append(test_mask)
    
    # Compute similarity matrices
    disease_sim_matrix = compute_disease_simlarity_matrix(disease_sim, args.disease_TopK)
    drug_sim_matrix = compute_drug_similarity_matrix(drug_sim, args.drug_TopK)

    # Calculate positive weight for imbalanced datasets
    pos_num = interactions.sum()
    neg_num = np.prod(interactions.shape) - pos_num
    pos_weight = neg_num / pos_num

    return disease_sim_matrix, drug_sim_matrix, interactions, train_data, test_data, pos_weight


class BatchManager:
    def __init__(self, data, batch_size, mode):
        """
        Initialize BatchManager for training and testing data

        Parameters:
            data (tuple): Contains masks and true labels
            batch_size (int): Number of samples per batch
            mode (str): 'train' or 'test' mode
        """
        disease_input, drug_input, labels = [], [], []

        if mode == "train":
            train_mask, truth_label = data
            adj_matrix = np.zeros_like(truth_label)
            adj_matrix[train_mask] = truth_label[train_mask]
            self.train_adj = adj_matrix

            disease_indices, drug_indices = np.indices(train_mask.shape)
            disease_input = disease_indices.flatten()
            drug_input = drug_indices.flatten()
            labels = np.where(train_mask, truth_label, 0).flatten()
        elif mode == "test":
            test_mask, truth_label = data
            disease_input, drug_input = np.where(test_mask)
            labels = truth_label[test_mask]
        
        # Create batches
        self.batch_data = []
        num_batches = math.ceil(len(disease_input) / batch_size)

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            self.batch_data.append([
                disease_input[start_idx:end_idx],
                drug_input[start_idx:end_idx],
                labels[start_idx:end_idx]
            ])
        
        self.len_data = len(self.batch_data)
    

    def iter_batch(self, shuffle=False):
        """
        Yield batches for model input

        Parameters:
            shuffle (bool): Whether to shuffle the batch data
        
        Yields:
            list: Batch containing disease inputs, drug inputs, and labels
        """
        if shuffle:
            random.shuffle(self.batch_data)
        
        for batch in self.batch_data:
            yield batch


def data_split(args, train_mask, test_mask, original_interactions):
    """
    Split data into training and test batches using BatchManager

    Parameters:
        args: Arguments containing batch_size and other configurations
        train_mask (ndarray): Boolean mask for training data
        test_mask (ndarray): Boolean mask for testing data
        original_interactions (ndarray): Original drug-disease interaction matrix
    
    Returns:
        Tuple containing train and test BatchManager instances
    """
    # Prepare training and testing data tuples
    train_data = (train_mask, original_interactions)
    test_data = (test_mask, original_interactions)

    # Initialize BatchManager for training and testing
    train_manager = BatchManager(train_data, args.batch_size, mode="train")
    test_manager = BatchManager(test_data, args.batch_size, mode="test")

    return train_manager, test_manager