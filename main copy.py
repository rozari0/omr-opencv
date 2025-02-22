import cv2
import numpy as np
import json
import csv

def load_config(config_path="omr_config.json"):
    """Loads OMR layout configuration from a JSON file."""
    with open(config_path, "r") as file:
        return json.load(file)

def load_and_preprocess_image(image_path):
    """Loads, converts to grayscale, applies blur, and thresholds the image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error: Could not load image: {image_path}")
    
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img_thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
    return img_thresh

def find_contours(img_thresh):
    """Finds contours in the thresholded image."""
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_column_boundaries(image_width, columns):
    """Calculates column boundaries dynamically based on JSON configuration."""
    boundaries = []
    start_x = 0
    
    for col in columns:
        col_width = col["col_width"] * image_width
        boundaries.append((start_x, start_x + col_width))
        start_x += col_width + (col["gap_after"] * image_width)
    
    return boundaries

def determine_column(cX, col_boundaries):
    """Determines which column a given x-coordinate belongs to."""
    for col_idx, (start, end) in enumerate(col_boundaries):
        if start <= cX < end:
            return col_idx
    return None

def analyze_omr(image_path, config_path="omr_config.json", top_spacing_percentage=0.0, min_contour_area=200, max_multiple_marks=1):
    """Analyzes the OMR sheet using dynamic configurations and handles multi-marked answers."""
    try:
        config = load_config(config_path)  # Load the OMR layout config
        img_thresh = load_and_preprocess_image(image_path)
        contours = find_contours(img_thresh)

        selected_answers = {}
        options = ['A', 'B', 'C', 'D'][:config["num_options"]]  # Adjust number of options dynamically

        image_height, image_width = img_thresh.shape
        questions_per_column = config["questions_per_column"]
        row_height = (image_height * (1 - top_spacing_percentage)) / questions_per_column

        col_boundaries = get_column_boundaries(image_width, config["columns"])

        # Dictionary to track the number of marked bubbles per question
        question_bubbles = {i: [] for i in range(1, (len(config["columns"]) * questions_per_column) + 1)}

        for contour in contours:
            if cv2.contourArea(contour) < min_contour_area:
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            row = int((cY - (image_height * top_spacing_percentage)) / row_height) + 1
            if row < 1 or row > questions_per_column:
                continue

            col_idx = determine_column(cX, col_boundaries)
            if col_idx is None:
                continue

            question_num = col_idx * questions_per_column + row
            col_start, col_end = col_boundaries[col_idx]
            col_width = col_end - col_start
            segment_width = col_width / (config["num_options"] + 1)
            relative_x = cX - col_start
            option_index = int(relative_x / segment_width) - 1

            # Add the contour to the relevant question
            if 0 <= option_index < len(options):
                question_bubbles[question_num].append(option_index)

        # Now process multi-marked answers
        for question_num, bubbles in question_bubbles.items():
            if len(bubbles) > max_multiple_marks:  # Flag or handle multi-marked answers
                selected_answers[question_num] = "Multiple Marks"
            elif len(bubbles) == 1:
                selected_answers[question_num] = options[bubbles[0]]
            else:
                selected_answers[question_num] = "Not Marked"

        return selected_answers

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def save_results_to_csv(results, filename="omr_results.csv"):
    """Saves extracted OMR answers to a CSV file."""
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Question", "Selected Answer"])
        for q_num in sorted(results.keys()):
            writer.writerow([q_num, results[q_num]])

if __name__ == "__main__":
    image_path = "image.png"  # Replace with your OMR image
    selected_answers = analyze_omr(image_path)

    if selected_answers:
        print("Selected Answers:")
        for q_num in sorted(selected_answers.keys()):
            print(f"Q{q_num}: {selected_answers[q_num]}")
        
        save_results_to_csv(selected_answers)
    else:
        print("Could not analyze OMR sheet.")
