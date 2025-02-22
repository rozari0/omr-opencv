import cv2
import numpy as np
import json
from typing import List, Tuple, Union, Dict

# Function to preprocess the image (convert to grayscale and threshold it)
def preprocess_image(image_path: str,
                     grayscale_output: str = "grayscale_omr.png",
                     binary_output: str = "binary_omr.png") -> np.ndarray:
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error: Could not load image from path: {image_path}")

    # Save grayscale image
    cv2.imwrite(grayscale_output, img)

    # Apply Gaussian Blur to reduce noise
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Apply binary threshold to create a binary image (invert the colors)
    _, binary_img = cv2.threshold(blurred_img, 40, 255, cv2.THRESH_BINARY_INV)
    
    # Save binary image
    cv2.imwrite(binary_output, binary_img)
    return binary_img

# Function to detect filled bubbles in the OMR sheet
def detect_filled_bubbles(img_thresh: np.ndarray,
                          sections: List[Tuple],
                          img_shape: Tuple[int, int]) -> Dict[str, List[Tuple[Union[int, None], Union[int, str]]]]:
    height, width = img_shape
    filled_bubbles = {}

    # Iterate through all defined sections in the OMR sheet
    for section in sections:
        section_name, x1_perc, x2_perc, y1_perc, y2_perc, options, num_questions, min_fill_ratio = section
        # Convert percentage coordinates to absolute pixel values
        x1 = int(x1_perc / 100 * width)
        x2 = int(x2_perc / 100 * width)
        y1 = int(y1_perc / 100 * height)
        y2 = int(y2_perc / 100 * height)

        # Calculate row height and column width based on section details
        row_height = (y2 - y1) / num_questions
        col_width = (x2 - x1) / len(options)

        # Iterate through all questions and options in the section
        for q in range(num_questions):
            for o, option in enumerate(options):
                # Get bounding box for the current bubble (question-option)
                cx1 = int(x1 + o * col_width)
                cx2 = int(x1 + (o + 1) * col_width)
                cy1 = int(y1 + q * row_height)
                cy2 = int(y1 + (q + 1) * row_height)
                cell = img_thresh[cy1:cy2, cx1:cx2]
                
                # Count total and black pixels in the bubble region
                total_pixels = cell.size
                black_pixels = np.sum(cell == 255)
                
                # Calculate fill ratio and check if it's above the threshold
                fill_ratio = black_pixels / total_pixels
                if fill_ratio > min_fill_ratio:
                    # If filled, add the detected option to the section result
                    filled_bubbles.setdefault(section_name, []).append((q + 1, option))
    
    return filled_bubbles

# Function to draw bounding boxes around detected bubbles on the original image
def draw_detected_areas(image: np.ndarray,
                        filled_bubbles: Dict[str, List[Tuple]],
                        sections: List[Tuple],
                        img_shape: Tuple[int, int]) -> np.ndarray:
    height, width = img_shape

    # Iterate through sections to draw bounding boxes
    for section in sections:
        section_name, x1_perc, x2_perc, y1_perc, y2_perc, options, num_questions, _ = section
        x1 = int(x1_perc / 100 * width)
        x2 = int(x2_perc / 100 * width)
        y1 = int(y1_perc / 100 * height)
        y2 = int(y2_perc / 100 * height)
        row_height = (y2 - y1) / num_questions
        col_width = (x2 - x1) / len(options)

        # Draw bounding boxes for each bubble
        for q in range(num_questions):
            for o, option in enumerate(options):
                cx1 = int(x1 + o * col_width)
                cx2 = int(x1 + (o + 1) * col_width)
                cy1 = int(y1 + q * row_height)
                cy2 = int(y1 + (q + 1) * row_height)
                cv2.rectangle(image, (cx1, cy1), (cx2, cy2), (255, 0, 0), 1)  # Red box for each bubble
                
                # If the bubble is filled, draw a green box
                if (q + 1, option) in filled_bubbles.get(section_name, []):
                    cv2.rectangle(image, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)  # Green box for filled bubble

    return image

# Main function to scan and process the OMR sheet
def scan_omr(image_path: str,
             output_path: str = "scanned_omr.png",
             show: bool = False) -> Dict[str, List[Tuple[Union[int, None], Union[int, str]]]]:
    # Define the sections on the OMR sheet and their properties
    sections = [
        ("MCQ Section 1", 8, 24, 19, 90.5, ["A", "B", "C", "D"], 25, 0.05),
        ("MCQ Section 2", 31, 46.5, 19, 90.5, ["A", "B", "C", "D"], 25, 0.05),
        ("Roll Number", 49, 61.5, 24, 38.5, list(range(1, 7)), 10, 0.3),
        ("Registration", 49, 61.5, 45.5, 60.2, list(range(1, 7)), 10, 0.3),
        ("Subject Code", 67.5, 74, 45.5, 60.2, list(range(1, 4)), 10, 0.25),
        ("Set Code", 67.5, 74, 24, 34, ["Set"], 4, 0.01),
    ]

    # Read the original image
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Error: Unable to read image at path {image_path}")
    
    # Preprocess the image (convert to grayscale and threshold)
    img_thresh = preprocess_image(image_path)
    
    # Detect filled bubbles based on the thresholded image
    filled_bubbles = detect_filled_bubbles(img_thresh, sections, (img_thresh.shape[0], img_thresh.shape[1]))
    
    # Draw the detected bubbles on the original image
    result_img = draw_detected_areas(original_img.copy(), filled_bubbles, sections, (img_thresh.shape[0], img_thresh.shape[1]))
    
    # Save the processed image
    cv2.imwrite(output_path, result_img)
    
    # Optionally display the processed image
    if show:
        cv2.imshow("Scanned OMR", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return filled_bubbles

# Function to merge answers from both MCQ sections
def merge_mcq_sections(filled_bubbles: Dict[str, List[Tuple[Union[int, None], Union[int, str]]]]) -> List[Union[str, int]]:
    # Combine answers from two MCQ sections
    mcq_section_1 = filled_bubbles.get("MCQ Section 1", [])
    mcq_section_2 = filled_bubbles.get("MCQ Section 2", [])
    
    # Adjust question numbers for the second section (starting from 26)
    mcq_section_2_adjusted = [(q + 25, option) for q, option in mcq_section_2]
    
    # Merge the answers
    all_mcqs = mcq_section_1 + mcq_section_2_adjusted
    mcq_answers = [0] * 50  # Assume a total of 50 questions (25 per section)
    for q, option in all_mcqs:
        mcq_answers[q - 1] = option
    return mcq_answers

# Function to process individual sections (Roll Number, Registration, Subject Code)
def process_section(filled_bubbles: Dict[str, List[Tuple]], section_name: str, expected_columns: List[int]) -> List[Union[int, str]]:
    detected = filled_bubbles.get(section_name, [])
    mapping = {}
    # Map detected columns to questions
    for question, col in detected:
        try:
            col_int = int(col)
        except (ValueError, TypeError):
            continue
        if col_int in mapping:
            mapping[col_int] = min(mapping[col_int], question)
        else:
            mapping[col_int] = question
    
    # Generate the result list based on expected columns
    result = []
    for col in expected_columns:
        result.append(mapping.get(col, 'X'))  # 'X' for missing values
    return result

# Function to format the processed section data into a string
def format_section_output(processed_data: List[Union[int, str]]) -> str:
    return ''.join(str(item) for item in processed_data)

# Function to subtract one from each digit in a string
def subtract_one_from_digits(input_string: str) -> str:
    result = []
    for char in input_string:
        if char.isdigit():
            result.append(str(int(char) - 1))
        else:
            result.append(char)
    return ''.join(result)

# Function to generate a structured JSON response from the filled bubbles
def generate_json_response(filled_bubbles: Dict[str, List[Tuple[Union[int, None], Union[int, str]]]]) -> Dict:
    # Process the merged MCQ answers and other sections
    merged_mcq = merge_mcq_sections(filled_bubbles)
    processed_roll = process_section(filled_bubbles, "Roll Number", list(range(1, 7)))
    processed_reg = process_section(filled_bubbles, "Registration", list(range(1, 7)))
    processed_subj = process_section(filled_bubbles, "Subject Code", list(range(1, 4)))

    # Format the processed results into strings
    roll_str = format_section_output(processed_roll)
    reg_str = format_section_output(processed_reg)
    subj_str = format_section_output(processed_subj)

    # Adjust the roll, registration, and subject code
    roll_str_adjusted = subtract_one_from_digits(roll_str)
    reg_str_adjusted = subtract_one_from_digits(reg_str)
    subj_str_adjusted = subtract_one_from_digits(subj_str)

    # Structure the response as a JSON object
    response = {
        "MCQ Answers": merged_mcq,
        "Roll Number": roll_str_adjusted,
        "Registration": reg_str_adjusted,
        "Subject Code": subj_str_adjusted
    }

    return response

# Main execution
if __name__ == "__main__":
    try:
        # Scan and process the OMR sheet, then generate a JSON response
        results = scan_omr("omr.jpg", "scanned_omr.png", show=False)
        response = generate_json_response(results)
        
        # Print the formatted JSON response
        print(json.dumps(response, indent=4))
    except Exception as e:
        # Handle any exceptions that may occur
        print(f"Error: {e}")
