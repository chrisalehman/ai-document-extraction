import json


def get_metadata(file_path, images_dir):
    """
    Reads a JSONL file and returns a list of metadata tuples, where each tuple contains the image path and the
    corresponding JSON object representing the ground truth document fields for that image.

    Args:
        file_path (str): The path to the JSONL file.
        images_dir (str): The directory containing the images.

    Returns:
        list: A list of (image_path, ground_truth_json) tuples.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:

            example = json.loads(line.strip())
            grouhd_truth_json = json.loads(example["text"])
            file_name = example["file_name"]
            file_path = f"{images_dir}/{file_name}"

            data.append((file_path, grouhd_truth_json))
    return data


def get_text(file_path):
    """ Reads a text file and returns the content as a string."""
    with open(file_path, "r") as file:
        return file.read()
