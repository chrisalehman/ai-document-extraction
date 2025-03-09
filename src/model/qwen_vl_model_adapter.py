from qwen_vl_utils import process_vision_info  # type: ignore
import json, torch  # type: ignore
from src.model.evaluator import normalize_value


#### private functions ####


def __prepare_inputs(processor, system_prompt, user_prompt, image_path, label=None):
    # format prompt
    messages = format_prompt(system_prompt, user_prompt, image_path, label)

    # applies chat template to image and prompt
    text = processor.apply_chat_template(
        messages, tokenizer=False, add_generation_prompt=True
    )

    # configure processor
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )

    # sets the correct device
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
    return inputs


def __prepare_batch_inputs(
    processor, system_prompt, user_prompt, image_paths, labels=None
):
    text_list = []
    image_inputs_list = []

    # Process each image path individually
    for i, image_path in enumerate(image_paths):

        # format prompt
        messages = format_prompt(
            system_prompt,
            user_prompt,
            image_path,
            labels[i] if labels is not None else None,
        )

        # Apply chat template to format the image and prompt into a single text string
        text = processor.apply_chat_template(
            messages, tokenizer=False, add_generation_prompt=True
        )
        text_list.append(text)

        # Extract vision information (images) from the messages
        image_inputs, _ = process_vision_info(messages)
        image_inputs_list.append(
            image_inputs[0]
        )  # image_inputs is a list with one image

    # Process the batch of texts and images
    inputs = processor(
        text=text_list,
        images=image_inputs_list,
        padding=True,
        return_tensors="pt",
    )

    # Move inputs to the appropriate device
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
    return inputs


def __run_inference(model, processor, inputs, max_new_tokens):

    with torch.no_grad():  # Disable gradient computation for inference
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


#### public functions ####


def format_prompt(system_prompt, user_prompt, image_path, label=None):
    """
    Formats the prompt by combining system and user prompts with an image path and an optional label.

    Args:
        system_prompt (str): The system-generated prompt.
        user_prompt (str): The user-provided prompt.
        image_path (str): The file path to the image.
        label (str, optional): An optional label for the image. Only used for training.
    Returns:
        list: A list of dictionaries representing the combined prompt input.
    """

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_prompt},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": label}]},
    ]

    # slice if no label provided (i.e., inference instead of training)
    if label is None:
        messages = messages[:2]

    return messages


def process_response(text):
    """
    Extracts and parses a JSON object from a given string.

    This function searches for the first and last curly braces in the input text,
    extracts the substring enclosed by these braces, and attempts to parse it as a JSON object.

    Args:
        text (str): The input string containing the JSON object.

    Returns:
        dict or None: The parsed JSON object if successful, otherwise None.
    """

    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if first_brace == -1 or last_brace == -1 or first_brace >= last_brace:
        return None

    json_string = text[first_brace : last_brace + 1]
    json_object = json.loads(json_string)

    # normalize all values in the dictionary
    for k, v in json_object.items():
        json_object[k] = normalize_value(v)

    return json_object


def run_inference(
    model, processor, system_prompt, user_prompt, image_path, max_new_tokens
):
    """
    Runs inference on the given model using the provided inputs.

    Args:
        model: The model to run inference on.
        processor: The processor to prepare inputs for the model.
        system_prompt (str): The system prompt to guide the model.
        user_prompt (str): The user prompt to guide the model.
        image_path (str): The path to the image file to be processed.
        max_new_tokens (int): The maximum number of new tokens to generate.

    Returns:
        str: The processed response from the model.
    """

    inputs = __prepare_inputs(processor, system_prompt, user_prompt, image_path)
    output_texts = __run_inference(model, processor, inputs, max_new_tokens)
    response = process_response(output_texts[0])

    return response


def run_batch_inference(
    model, processor, system_prompt, user_prompt, image_paths, max_new_tokens
):
    """
    Runs batch inference on a set of images using the specified model and processor.

    Args:
        model: The machine learning model to use for inference.
        processor: The processor to prepare inputs for the model.
        system_prompt (str): The system prompt to guide the model's responses.
        user_prompt (str): The user prompt to guide the model's responses.
        image_paths (list of str): A list of file paths to the images to be processed.
        max_new_tokens (int): The maximum number of new tokens to generate in the response.

    Returns:
        list of str: A list of processed response texts generated by the model.
    """

    batch_inputs = __prepare_batch_inputs(
        processor, system_prompt, user_prompt, image_paths
    )
    output_texts = __run_inference(model, processor, batch_inputs, max_new_tokens)
    responses = [process_response(text) for text in output_texts]

    return responses
