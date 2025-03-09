# src/utils/env_setup.py
import os
from dotenv import load_dotenv, find_dotenv
import sys
import json


def load_environment_variables(env_file=".env"):
    """
    Load and print all environment variables from the specified .env file.
    
    Args:
        env_file (str): Path to the .env file (default is '.env').
    """

    # load env vars
    env_file_path = find_dotenv(env_file)
    if not env_file_path:
        raise Exception(f"No {env_file} found on the system path.")
    if not load_dotenv(env_file):
        raise Exception(f"Failed to load {env_file_path}.")

    # construct app-only env vars dictionary
    env_vars = {}
    for k,v in os.environ.items():
        if (k.startswith("APP_")):
            env_vars[k] = v
    
    # list loaded env vars
    if env_vars:
        print(f"Loaded application properties from: {env_file_path}")
    else:
        print(f"No new or changed environment variables loaded from {env_file}.")


def setup_environment():
    """Bootstrap the environment for local or Colab execution.
    This function sets up the environment by detecting if the code is running
    in Google Colab or a local environment. It loads the appropriate environment
    variables from an .env file, sets the working directory, and updates the 
    Python path. If running in Colab, it also mounts Google Drive.

    Returns:
        str: The environment type, either 'colab' or 'local'.
    Raises:
        Exception: If the .env file is not found, if the PROJECT_DIR environment
        variable is not set or invalid, or if running in Colab but google.colab 
        is not available.
    """

    # Detect if running in Colab
    is_colab = "google.colab" in sys.modules
    env_file = ".env.colab" if is_colab else ".env.local"
    env = "colab" if is_colab else "local"

    # Load environment variables
    load_environment_variables(env_file)
    
    # Set working directory
    project_dir = os.getenv("APP_PROJECT_DIR")
    if not project_dir or not os.path.exists(project_dir):
        raise Exception(f"APP_PROJECT_DIR not set or invalid: {project_dir}")
    if os.getcwd() != project_dir:
        os.chdir(project_dir)
    print(f"Working directory: {os.getcwd()}")

    # Add src to system path
    sys.path.append(os.path.join(project_dir, "src"))

    # Colab-specific setup (mount Google Drive)
    if is_colab:
        try:
            from google.colab import drive  # pyright: ignore[reportMissingImports]

            drive.mount("/content/drive")
        except ImportError:
            raise Exception("IS_COLAB is true, but google.colab not available")

    return env