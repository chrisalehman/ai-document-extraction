def get_w2_form_type(filename):
    """
    Args:
            filename (str): The name of the file to parse.

    Returns:
            str or None: The token after 'input' in the filename, or None if 'input' is not found or filename is invalid.

    Example:
            'W2_Multi_Sample_Data_input_IRS2_clean_10391.png' -> 'IRS2'
    """
    if not filename or "input" not in filename:
        return None
    # Split the filename by '_' and find the token after 'input'
    parts = filename.split("_")
    for i, part in enumerate(parts):
        if part == "input" and i + 1 < len(parts):
            return parts[i + 1]
    return None
