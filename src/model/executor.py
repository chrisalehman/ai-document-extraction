from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from src.utils import w2_dataset
from src.model import qwen_vl_model_adapter, evaluator
import pandas as pd
import math


class Executor:
    def __init__(
        self,
        model: Qwen2_5_VLForConditionalGeneration,
        processor: Qwen2_5_VLProcessor,
        system_prompt: str,
        user_prompt: str,
    ):
        self.model = model
        self.processor = processor
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def execute_inference_test(self, metadata, batch_size, max_new_tokens):
        """
        Run inference on a set of test examples and compare the results to the ground truth.

        Args:
            metadata (list): A list of tuples where each tuple contains the image path and the ground truth JSON.
            batch_size (int): The number of examples to process in each batch.
            max_new_tokens (int): The maximum number of new tokens to generate during inference.

        Returns:
            Pandas dataframe containing the evaluation results.
        """
        print(f"Processing {len(metadata)} examples...")

        all_rows = []
        for i in range(0, len(metadata), batch_size):

            # Get the current batch
            batch_labels = metadata[i : i + batch_size]
            image_paths = [image_path for image_path, _ in batch_labels]

            # Log the batch being processed
            batch_count = (i // batch_size) + 1
            batch_total = math.ceil(len(metadata) / batch_size)
            print(
                f"Processing batch ({batch_count} of {batch_total}); batch size = {batch_size}."
            )

            # Run inference on the batch
            predictions = qwen_vl_model_adapter.run_batch_inference(
                self.model,
                self.processor,
                self.system_prompt,
                self.user_prompt,
                image_paths,
                max_new_tokens,
            )

            # Process each prediction in the batch
            for j, (image_path, ground_truth_json) in enumerate(batch_labels):
                file_name = image_path.split("/")[-1]
                prediction = predictions[j]
                rows = evaluator.compare_fields(ground_truth_json, prediction, i + j)
                form_type = w2_dataset.get_w2_form_type(file_name)
                for row in rows:
                    row.append(form_type)
                all_rows.extend(rows)

        # Display comparison results
        df = pd.DataFrame(
            all_rows,
            columns=[
                "Comparison ID",
                "Field",
                "Predicted Value",
                "Ground Truth Value",
                "Match",
                "Form Type",
            ],
        )

        return df
