def __write_output(f, s):
    """ Writes a string to a file and prints it to the console. """
    print(s, end="")
    f.write(s)


def output_results(df, report_file_path):
    """
    Generates and outputs a report summarizing the results of a comparison.

    The function calculates overall accuracy, field summary, sample summary,
    and form type summary from the provided DataFrame. It also calculates
    mismatch percentages for fields and form types if there are any mismatches.
    The results are written to the specified report file path in markdown format.

    Args:
        df (pandas.DataFrame): DataFrame containing comparison results with columns
                            'Match', 'Field', 'Comparison ID', and 'Form Type'.
        report_file_path (str): Path to the file where the report will be written.

    Returns:
        None
    """
    # Calculate overall accuracy
    overall_accuracy = df["Match"].mean()

    # Field Summary: Sort alphabetically by 'Field'
    field_summary = (
        df.groupby("Field")
        .agg(
            total_comparisons=("Match", "size"),
            matches=("Match", "sum"),
            mismatches=("Match", lambda x: (~x).sum()),
            accuracy=("Match", "mean"),
        )
        .sort_index()
    )

    # Sample Summary: Sort by 'Comparison ID' ascending
    sample_summary = (
        df.groupby("Comparison ID")
        .agg(
            total_fields=("Match", "size"),
            matches=("Match", "sum"),
            mismatches=("Match", lambda x: (~x).sum()),
            accuracy=("Match", "mean"),
        )
        .sort_index()
    )

    # Form Type Summary: Group by 'Form Type' and calculate metrics, sort alphabetically
    form_type_summary = (
        df.groupby("Form Type")
        .agg(
            total_comparisons=("Match", "size"),
            matches=("Match", "sum"),
            mismatches=("Match", lambda x: (~x).sum()),
            accuracy=("Match", "mean"),
        )
        .sort_index()
    )

    # Calculate mismatch percentage per field (optional)
    total_mismatches = (~df["Match"]).sum()
    if total_mismatches > 0:
        field_summary["mismatch_percentage"] = (
            field_summary["mismatches"] / total_mismatches
        ) * 100
    else:
        field_summary["mismatch_percentage"] = 0

    # Calculate mismatch percentage per form type (optional)
    if total_mismatches > 0:
        form_type_summary["mismatch_percentage"] = (
            form_type_summary["mismatches"] / total_mismatches
        ) * 100
    else:
        form_type_summary["mismatch_percentage"] = 0

    # Write output to file and console
    with open(report_file_path, "w") as f:
        __write_output(f, f"**Overall Accuracy**: {overall_accuracy:.2%}\n")
        __write_output(f, "\n**Field Summary**:\n")
        __write_output(f, field_summary.to_markdown() + "\n")
        __write_output(f, "\n**Sample Summary**:\n")
        __write_output(f, sample_summary.to_markdown() + "\n")
        __write_output(f, "\n**Form Type Summary**:\n")
        __write_output(f, form_type_summary.to_markdown() + "\n")


def output_results_by_form_type(df, report_file_path, form_type):
    """
    Generates a detailed report for a specific form type from the given DataFrame and writes it to a file.

    Args:
        df (pandas.DataFrame): The input DataFrame containing comparison data.
        report_file_path (str): The file path where the report will be saved.
        form_type (str): The form type to filter and generate the report for.

    The report includes:
        - Overall accuracy for the specified form type.
        - Field summary with total comparisons, matches, mismatches, accuracy, and mismatch percentage.
        - Sample summary with total fields, matches, mismatches, and accuracy per sample.
        - Detailed comparison with Comparison ID, Field, Predicted Value, Ground Truth Value, and Accuracy %.

    The report is written to the specified file in markdown format.
    """
    ft_df = df[df["Form Type"] == form_type]

    # Calculate overall accuracy for Form_Type
    ft_overall_accuracy = ft_df["Match"].mean()

    # Field Summary for Form Type
    ft_field_summary = (
        ft_df.groupby("Field")
        .agg(
            total_comparisons=("Match", "size"),  # Number of comparisons per field
            matches=("Match", "sum"),  # Number of correct matches
            mismatches=("Match", lambda x: (~x).sum()),  # Number of mismatches
            accuracy=("Match", "mean"),  # Accuracy per field
        )
        .sort_index()  # Sort alphabetically by Field
    )

    # Sample Summary for Form Type
    ft_sample_summary = (
        ft_df.groupby("Comparison ID")
        .agg(
            total_fields=("Match", "size"),  # Number of fields per sample
            matches=("Match", "sum"),  # Number of correct matches
            mismatches=("Match", lambda x: (~x).sum()),  # Number of mismatches
            accuracy=("Match", "mean"),  # Accuracy per sample
        )
        .sort_index()  # Sort by Comparison ID
    )

    # Optional: Calculate mismatch percentage per field for Form Type
    ft_total_mismatches = (~ft_df["Match"]).sum()
    if ft_total_mismatches > 0:
        ft_field_summary["mismatch_percentage"] = (
            ft_field_summary["mismatches"] / ft_total_mismatches
        ) * 100
    else:
        ft_field_summary["mismatch_percentage"] = 0

    # Select the initial relevant columns
    relevant_columns = [
        "Comparison ID",
        "Field",
        "Predicted Value",
        "Ground Truth Value",
        "Match",
    ]
    ft_df = ft_df[relevant_columns]

    # Calculate "Accuracy %" (100% if Match is True, 0% if False)
    ft_df["Accuracy %"] = ft_df["Match"].astype(int) * 100

    # Drop the "Match" column since it’s not requested in the output
    ft_df = ft_df.drop(columns=["Match"])

    # Sort by "Comparison ID" and "Field" for better organization
    ft_df = ft_df.sort_values(by=["Comparison ID", "Field"])

    # Arrange columns in the requested order
    ft_df = ft_df[
        [
            "Comparison ID",
            "Field",
            "Predicted Value",
            "Ground Truth Value",
            "Accuracy %",
        ]
    ]

    # Write output to file only
    with open(report_file_path, "w") as f:
        f.write(f"\n\n**{form_type} Overall Accuracy**: {ft_overall_accuracy:.2%}\n")
        f.write(f"\n**{form_type} Field Summary**:\n")
        f.write(ft_field_summary.to_markdown())
        f.write(f"\n\n**{form_type} Sample Summary**:\n")
        f.write(ft_sample_summary.to_markdown())
        f.write(f"\n\n**{form_type} Detailed Comparison**:\n")
        f.write(ft_df.to_markdown(index=False))
