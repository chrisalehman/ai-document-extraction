import pandas as pd
import re


def is_float(string):
    """
    Check if a string represents a floating-point number.

    Args:
        string (str): The string to check.

    Returns:
        bool: True if the string represents a floating-point number, False otherwise.
    """

    pattern = r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$"
    return bool(re.match(pattern, string))


def normalize_value(value):
    """Normalize value for comparison and display.

    Args:
        value (Any): The value to normalize. Can be an int, float, str, or None.

    Returns:
        str: The normalized value as a string. If the value is None or NaN, returns an empty string.
    """
    if isinstance(value, (int, float)) and not pd.isna(value):
        return f"{float(value):.2f}"
    if isinstance(value, str):
        stripped = value.strip()
        if is_float(stripped):
            return f"{float(stripped):.2f}"
        return stripped
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def compare_fields(ground_truth, prediction, comparison_id):
    """Compare fields between ground truth and prediction.

    Args:
        ground_truth (dict): The ground truth data.
        prediction (dict): The predicted data.
        comparison_id (str): An identifier for the comparison.

    Returns:
        list: A list of rows containing the comparison results. Each row is a list with the following elements:
            - comparison_id (str): The comparison identifier.
            - key (str): The field key.
            - pred_norm (str): The normalized predicted value.
            - gt_norm (str): The normalized ground truth value.
            - match (bool): Whether the normalized values match.
    """

    rows = []

    # Compare each field in the ground truth with the prediction
    for key in ground_truth:
        gt_value = ground_truth.get(key, "")
        pred_value = prediction.get(key, "")
        gt_norm = normalize_value(gt_value)
        pred_norm = normalize_value(pred_value)
        match = gt_norm == pred_norm
        rows.append([comparison_id, key, pred_norm, gt_norm, match])

    return rows


def evaluate_accuracy(ground_truth, prediction):
    """
    Evaluate the accuracy of a JSON prediction against the JSON ground truth. Assumes both JSON objects have
    been normalized.

    Args:
        ground_truth (dict): The ground truth JSON object.
        prediction (dict): The predicted JSON object.

    Returns:
        float: The accuracy of the prediction based on the aggregate number of matching values for the matching
        keys. The accuracy is a value between 0.0 and 1.0.
    """

    matches = 0
    total_fields = len(ground_truth)
    for key in ground_truth:
        gt_value = ground_truth[key]
        pred_value = prediction[key]
        if gt_value == pred_value:
            matches += 1

    accuracy = matches / total_fields
    return accuracy
