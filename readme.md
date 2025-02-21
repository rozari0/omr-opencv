# OMR Sheet Analyzer

This is a Python-based Optical Mark Recognition (OMR) analyzer that processes scanned OMR sheets and extracts marked answers based on configurable parameters. It supports dynamic layouts defined in a JSON configuration file.

## Features
- Dynamically processes OMR layouts using a JSON configuration.
- Detects marked bubbles and assigns answers.
- Handles multiple marked answers with a configurable threshold.
- Saves results in a CSV file for easy access.

## Requirements
Ensure you have the following dependencies installed:

```sh
pip install opencv-python numpy
```

## Configuration
The OMR sheet layout is defined in `omr_config.json`. Modify this file to match your specific OMR layout, including columns, row counts, and spacing.

## Usage

1. **Prepare the environment**
   - Place your OMR image file (e.g., `image.png`) in the project directory.
   - Ensure the configuration file (`omr_config.json`) is correctly set up.

2. **Run the script**

   ```sh
   python main.py
   ```

3. **Check the results**
   - The extracted answers will be printed in the console.
   - A CSV file (`omr_results.csv`) will be generated containing the results.

## Output
The program outputs a CSV file with the following format:

```csv
Question,Selected Answer
1,A
2,C
3,Multiple Marks
...
```

## Notes
- Modify `image_path` in `omr_analyzer.py` to specify the OMR sheet image.
- Adjust `omr_config.json` to match different OMR layouts.
- Ensure good image quality for accurate results.

## License
This project is open-source and available under the MIT License.
