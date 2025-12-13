import json
import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, cohen_kappa_score


DIMENSION_LIST = [
    "realism",
    "deformation",
    "imagination",
    "color_richness",
    "color_contrast",
    "line_combination",
    "line_texture",
    "picture_organization",
    "transformation"
]


MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
NAME = ""
SCORE_INPUT = ""
OUTPUT_FILE_SEQ = ""
OUTPUT_FILE = ""


def load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


# Calculate SC (Spearman) for score consistency
def get_sc(original_scores, current_scores):
    spearman_corr, _ = spearmanr(original_scores, current_scores)
    return spearman_corr


def get_pearson(original_scores, current_scores):
    pearson_corr, _ = pearsonr(original_scores, current_scores)
    return pearson_corr


# Calculate SD (Score Difference)
def get_sd(original_scores, current_scores):
    score_diff = np.abs(original_scores - current_scores)
    avg_sd = score_diff.mean()
    return avg_sd


# def get_auc(original_scores, current_scores):
#     correct_predictions = (original_scores == current_scores).sum()
#     length = len(original_scores)
#     return correct_predictions / length


def get_auc(original_scores, current_scores):
    correct_predictions = (original_scores.round().astype(int).clip(1, 5) == current_scores.round().astype(int).clip(1, 5)).sum()
    length = len(original_scores)
    return correct_predictions / length


def get_mse(original_scores, current_scores):
    return mean_squared_error(original_scores, current_scores)


def get_rmse(original_scores, current_scores):
    return np.sqrt(mean_squared_error(original_scores, current_scores))


def get_qwk(original_scores, current_scores):
    original_scores = original_scores.round().astype(int).clip(1, 5)
    current_scores = current_scores.round().astype(int).clip(1, 5)
    return cohen_kappa_score(original_scores, current_scores, weights='quadratic')


def get_rv_coefficient(original_scores_df, current_scores_df):
    """
    Calculate the RV coefficient between two multivariate datasets.
    
    Parameters:
    original_scores_df: DataFrame with original scores for each dimension
    current_scores_df: DataFrame with current scores for each dimension
    
    Returns:
    float: RV coefficient value
    """
    # Center the data matrices
    orig_centered = original_scores_df - original_scores_df.mean()
    curr_centered = current_scores_df - current_scores_df.mean()
    
    # Calculate the cross-covariance matrix
    cross_cov = orig_centered.T @ curr_centered
    
    # Calculate the covariance matrices for each dataset
    orig_cov = orig_centered.T @ orig_centered
    curr_cov = curr_centered.T @ curr_centered
    
    # Calculate the RV coefficient
    numerator = np.trace(cross_cov.T @ cross_cov)
    denominator = np.sqrt(np.trace(orig_cov.T @ orig_cov) * np.trace(curr_cov.T @ curr_cov))
    
    return numerator / denominator if denominator != 0 else 0


def is_valid_value(value):
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return True


def extract_scores(score_inputs, output_file_scores, dataset_name="ArtEdu", model_name="deepseek-vl2-tiny"):
    score_results = []

    score_all_data = load_json_data(score_inputs)

    target_data = score_all_data[dataset_name][model_name]

    for image_name, score_list in target_data.items():
        for dimension, score_dict in score_list.items():
            if 'current' in score_dict.keys():
                model_score = score_dict['current']
            elif 'predicted' in score_dict.keys():
                model_score = score_dict['predicted']
            elif 'predicted_by_realism' in score_dict.keys():
                model_score = score_dict['predicted_by_realism']
            elif 'predicted_by_deformation' in score_dict.keys():
                model_score = score_dict['predicted_by_deformation']
            elif 'predicted_by_imagination' in score_dict.keys():
                model_score = score_dict['predicted_by_imagination']
            elif 'predicted_by_color_richness' in score_dict.keys():
                model_score = score_dict['predicted_by_color_richness']
            elif 'predicted_by_color_contrast' in score_dict.keys():
                model_score = score_dict['predicted_by_color_contrast']
            elif 'predicted_by_line_combination' in score_dict.keys():
                model_score = score_dict['predicted_by_line_combination']
            elif 'predicted_by_line_texture' in score_dict.keys():
                model_score = score_dict['predicted_by_line_texture']
            elif 'predicted_by_picture_organization' in score_dict.keys():
                model_score = score_dict['predicted_by_picture_organization']
            elif 'predicted_by_transformation' in score_dict.keys():
                model_score = score_dict['predicted_by_transformation']

            infer_score = score_dict['original']

            if is_valid_value(model_score) and is_valid_value(infer_score):
                score_results.append({
                    'image': image_name,
                    'dimension': dimension,
                    'original': infer_score,
                    'current': model_score
                })

    # Save the scores to a DataFrame and output to an Excel file
    scores_df = pd.DataFrame(score_results)
    scores_df.to_csv(output_file_scores, index=False)
    print(f"Original and Current scores have been saved to: {output_file_scores}")


# Process correlations and score differences for each dimension
def calculate_sc_sd(scores_file, save_results="sc_sd_result.csv"):
    # Load the saved scores
    scores_df = pd.read_csv(scores_file)

    # Group by dimension and calculate the SC and SD for each
    sc_sd_results = []

    all_original = []
    all_current = []

    sc_values = []
    pc_values = []
    qwk_values = []    

    for dimension in DIMENSION_LIST:
        dim_scores = scores_df[scores_df['dimension'] == dimension]
        original_scores = dim_scores['original']
        current_scores = dim_scores['current']

        all_original.extend(original_scores)
        all_current.extend(current_scores)

        # Calculate SC (Spearman correlation) and SD (Average Difference)
        sc_value = get_sc(original_scores, current_scores)
        sd_value = get_sd(original_scores, current_scores)
        pc_value = get_pearson(original_scores, current_scores)
        auc_value = get_auc(original_scores, current_scores)

        mse_value = get_mse(original_scores, current_scores)
        rmse_value = get_rmse(original_scores, current_scores)
        qwk_value = get_qwk(original_scores, current_scores)

        sc_values.append(sc_value)
        pc_values.append(pc_value)
        qwk_values.append(qwk_value)

        sc_sd_results.append({
            'dimension': dimension,
            'SC': sc_value,
            'PC': pc_value,
            'SD': sd_value,
            'ACC': auc_value,
            'MSE': mse_value,
            'RMSE': rmse_value,
            'QWK': qwk_value,
        })

    all_original = np.array(all_original)
    all_current = np.array(all_current)

    # overall_sc = get_sc(all_original, all_current)
    # overall_pc = get_pearson(all_original, all_current)
    overall_sd = get_sd(all_original, all_current)
    overall_acc = get_auc(all_original, all_current)
    overall_mse = get_mse(all_original, all_current)
    overall_rmse = get_rmse(all_original, all_current)
    # overall_qwk = get_qwk(all_original, all_current)

    sc_sd_results.append({
        'dimension': 'Overall',
        'SC': np.mean(sc_values),
        'PC': np.mean(pc_values),
        'SD': overall_sd,
        'ACC': overall_acc,
        'MSE': overall_mse,
        'RMSE': overall_rmse,
        'QWK': np.mean(qwk_values),
    })

    # Save the SC and SD results
    sc_sd_df = pd.DataFrame(sc_sd_results)
    sc_sd_df.to_csv(save_results, index=False)
    print(f"SC and SD results have been saved to: {save_results}")


def calculate_result(scores_file, save_results="result.csv"):
    # Load the saved scores
    scores_df = pd.read_csv(scores_file)

    # Group by dimension and calculate the SC and SD for each
    sc_sd_results = []

    original_scores_dict = {}
    current_scores_dict = {}

    all_original = []
    all_current = []

    for dimension in DIMENSION_LIST:
        dim_scores = scores_df[scores_df['dimension'] == dimension]
        original_scores = dim_scores['original']
        current_scores = dim_scores['current']

        all_original.extend(original_scores)
        all_current.extend(current_scores)

        original_scores_dict[dimension] = original_scores.values
        current_scores_dict[dimension] = current_scores.values

        # Calculate SC (Spearman correlation) and SD (Average Difference)
        sc_value = get_sc(original_scores, current_scores)
        sd_value = get_sd(original_scores, current_scores)
        pc_value = get_pearson(original_scores, current_scores)
        auc_value = get_auc(original_scores, current_scores)

        mse_value = get_mse(original_scores, current_scores)
        rmse_value = get_rmse(original_scores, current_scores)
        qwk_value = get_qwk(original_scores, current_scores)

        sc_sd_results.append({
            'dimension': dimension,
            'SC': sc_value,
            'PC': pc_value,
            'SD': sd_value,
            'ACC': auc_value,
            'MSE': mse_value,
            'RMSE': rmse_value,
            'QWK': qwk_value,
        })

    all_original = np.array(all_original)
    all_current = np.array(all_current)

    # overall_sc = get_sc(all_original, all_current)
    # overall_pc = get_pearson(all_original, all_current)
    overall_sd = get_sd(all_original, all_current)
    overall_acc = get_auc(all_original, all_current)
    overall_mse = get_mse(all_original, all_current)
    overall_rmse = get_rmse(all_original, all_current)
    overall_qwk = get_qwk(all_original, all_current)

    original_scores_df = pd.DataFrame(original_scores_dict)
    current_scores_df = pd.DataFrame(current_scores_dict)
    
    # Calculate RV coefficient
    rv_value = get_rv_coefficient(original_scores_df, current_scores_df)    

    sc_sd_results.append({
        'dimension': 'Overall',
        'SC': np.mean(sc_value),
        'PC': np.mean(pc_value),
        'SD': overall_sd,
        'ACC': overall_acc,
        'MSE': overall_mse,
        'RMSE': overall_rmse,
        'QWK': overall_qwk,
    })

    # Save the SC and SD results
    sc_sd_df = pd.DataFrame(sc_sd_results)
    sc_sd_df.to_csv(save_results, index=False)
    print(f"SC and SD results have been saved to: {save_results}")


if __name__ == "__main__":

    # Processing the score data
    extract_scores(score_inputs=SCORE_INPUT, output_file_scores=OUTPUT_FILE_SEQ, model_name=MODEL_NAME)

    # Calculate SC (Spearman) and SD (Average Difference)
    calculate_sc_sd(scores_file=OUTPUT_FILE_SEQ, save_results=OUTPUT_FILE)
