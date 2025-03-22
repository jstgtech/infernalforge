# Infernal Forge

Infernal Forge is an open-source inference pipeline designed for generating high-quality images using state-of-the-art machine learning models. This project leverages Hugging Face's `diffusers` library and PyTorch to provide a streamlined and efficient workflow for image generation tasks.

---

## Features

- **Pre-trained Models**: Supports FLUX models hosted on Hugging Face.
- **Customizable Inference**: Easily adjust parameters like image size, inference steps, and random seeds.
- **Reproducibility**: Generate reproducible results with seed logging.
- **Extensibility**: Modular design for easy integration with other pipelines or workflows.
- **Output Directory Management**: Automatically ensures the output directory exists before saving images.
- **Error Handling**: Robust error handling with detailed logging for debugging.
- **Quantization Support**: Leverages GGUF for efficient model quantization, enabling faster inference and reduced memory usage.
- **Logging Utility**: Centralized logging configuration for consistent logging across all modules.

---

## Upcoming Features

Here are some features planned for future releases:

- **API Integration**: Expose the image generation functionality as a RESTful API using frameworks like FastAPI or Flask.
- **Web Interface**: Develop a user-friendly web interface for submitting prompts and downloading generated images.
- **Batch Processing**: Add support for generating multiple images in a single run.
- **Advanced Logging**: Implement structured logging with support for exporting logs to external systems (e.g., ELK stack).
- **Docker Support**: Add Docker configurations for easier deployment and environment consistency.

---

## Pre-requisites

Before setting up the project, ensure you have the following:

1. **Hugging Face Account**:
   - Create an account on [Hugging Face](https://huggingface.co/).
   - Generate an access token by visiting your [Hugging Face Access Tokens page](https://huggingface.co/settings/tokens).
   - Save the token securely, as it will be required to access the models.

2. **Access to FLUX Repositories**:
   - Ensure you have access to one of the following FLUX repositories on Hugging Face:
     - [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
     - [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
   - You will need the repository URL and your Hugging Face access token to download the models.

3. **Python**:
   - Ensure Python 3.12.x is installed on your system. Use one of the following package managers to install Python:
     - **Windows**: Use [Chocolatey](https://chocolatey.org/install) or [Winget](https://winget.run/pkg/Python/Python.3.12)
       ```powershell
       choco install python312
       ```
       ```powershell
       winget install -e --id Python.Python.3.12
       ```
     - **MacOS**: Use [Homebrew](https://brew.sh/)
       ```bash
       brew install python@3.12
       ```
     - **Linux (Ubuntu)**: Use [APT](https://documentation.ubuntu.com/server/how-to/software/package-management/index.html)
       ```bash
       sudo apt install -y python3.12
       ```

4. **PIP**:
   - PIP is the preferred Python package manager and comes pre-installed with Python on most systems.
   - For Ubuntu, install PIP using APT:
     ```bash
     sudo apt install -y python3-pip
     ```

5. **Python Virtual Environment**:
   - Using a Python virtual environment (`venv`) is highly recommended to isolate your Python version and package dependencies from system-wide installations.
   - Virtual environment tools (`venv`) come pre-installed with Python on Windows and MacOS.
   - For Ubuntu, install the `venv` package:
     ```bash
     sudo apt install -y python3.12-venv
     ```

---

## Supported Operating Systems

### Windows
- Windows 10 or 11
- NVIDIA GPU preferred
- WSL2 is highly recommended for this project. Follow the [WSL2 installation guide](https://learn.microsoft.com/en-us/windows/wsl/install) to set it up, using Ubuntu 24.04.

### MacOS
- MacOS 13+ (Ventura or later)
- ARM-based processors (e.g., Apple Silicon) are supported.

### Linux
- Ubuntu 22.04 or 24.04 LTS
- NVIDIA GPU preferred for optimal performance.

---

## Local Setup

Follow these steps to set up the project locally:

1. **Set Up a Virtual Environment**:
   - Create and activate a virtual environment:
     - **MacOS/Linux**:
       ```bash
       python -m venv .venv
       source .venv/bin/activate
       ```
     - **Windows (PowerShell)**:
       ```powershell
       python -m venv .venv
       .venv\Scripts\Activate.ps1
       ```
     - **Windows (Command Prompt)**:
       ```cmd
       python -m venv .venv
       .venv\Scripts\activate.bat
       ```

2. **Install Project Dependencies**:
   - Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the Application**:
   - Execute the main script:
     ```bash
     python main.py
     ```

4. **Deactivate the Virtual Environment**:
   - To leave the virtual environment, run:
     ```bash
     deactivate
     ```

---

## Cleanup Script

The project includes a utility script to clean up the `output/` folder by deleting all files while keeping the folder intact.

### Usage

To clean up the `output/` folder, run the following command:

```bash
python cleanup_output.py
```

### Notes
- The script will log all deleted files.
- If the `output/` folder does not exist or is already empty, the script will notify you.
- The folder itself will not be deleted, only its contents.

---

## Logging

Infernal Forge uses a centralized logging utility to ensure consistent logging across all modules.

### Logging Utility

The `utils/logger.py` module provides a `configure_logger` function to configure and return a logger instance. This ensures that all modules use the same logging format and behavior.

### Log Levels
The following log levels are used:
- **INFO**: General information about the application's progress (e.g., pipeline initialization, image generation).
- **ERROR**: Errors encountered during execution (e.g., pipeline failures, image saving issues).

### Viewing Logs
Logs are printed to the console by default. To view the logs:
1. Run the application:
   ```bash
   python main.py
   ```
2. Check the console output for log messages.

### Customizing Logging
To customize the logging behavior (e.g., log to a file), modify the `configure_logger` function in `utils/logger.py`.

Example:
```python
from utils.logger import configure_logger

logger = configure_logger(
    __name__,
    level=logging.DEBUG,
    log_format="%(asctime)s - %(levelname)s - %(message)s"
)
```

---

### Environment Variables

The following environment variables are used to configure the application:

- **HF_ACCESS_TOKEN**: Your Hugging Face access token.
- **MODEL**: The name of the pre-trained model to use.
- **CKPT_PATH**: The URL to the model checkpoint file.
- **DEFAULT_HEIGHT**: The default height of generated images (default: 256).
- **DEFAULT_WIDTH**: The default width of generated images (default: 256).
- **DEFAULT_NUM_INFERENCE_STEPS**: The default number of inference steps (default: 50).
- **DEFAULT_SEED**: The default random seed for reproducibility (default: 0).

---

## Notes
- This project is designed to work with CUDA 12.4 for GPU acceleration. However, it can still run on CPU if GPU support is unavailable.
- For Windows users, WSL2 with Ubuntu 24.04 is the recommended development environment.

---

## Author

Created by jstgtech

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.