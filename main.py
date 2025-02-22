import cv2
import numpy as np

def preprocess_image(image_path, grayscale_output="grayscale_omr.png", binary_output="binary_omr.png"):
    """Loads an image, converts it to grayscale, applies Gaussian blur, and applies a fixed threshold."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Error: Could not load image from path: {image_path}")

    # Save grayscale image
    cv2.imwrite(grayscale_output, img)

    # Apply Gaussian blur to reduce noise
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply fixed inverse binary thresholding
    _, binary_img = cv2.threshold(blurred_img, 50, 255, cv2.THRESH_BINARY_INV)

    # Save the binary thresholded image
    cv2.imwrite(binary_output, binary_img)

    return binary_img

def detect_filled_bubbles(img_thresh, sections, img_shape):
    """Detects filled bubbles based on predefined sections and unique fill ratio per section."""
    height, width = img_shape
    filled_bubbles = {}

    for section_name, x1_perc, x2_perc, y1_perc, y2_perc, options, num_questions, min_fill_ratio in sections:
        x1, x2 = int(x1_perc / 100 * width), int(x2_perc / 100 * width)
        y1, y2 = int(y1_perc / 100 * height), int(y2_perc / 100 * height)

        row_height = (y2 - y1) / num_questions
        col_width = (x2 - x1) / len(options)

        for q in range(num_questions):
            for o, option in enumerate(options):
                cx1, cx2 = int(x1 + o * col_width), int(x1 + (o + 1) * col_width)
                cy1, cy2 = int(y1 + q * row_height), int(y1 + (q + 1) * row_height)

                cell = img_thresh[cy1:cy2, cx1:cx2]
                total_pixels = cell.size
                black_pixels = np.sum(cell == 255)  # Count filled (white) pixels in the inverted image
                fill_ratio = black_pixels / total_pixels

                if fill_ratio > min_fill_ratio:
                    filled_bubbles.setdefault(section_name, []).append((q + 1, option))

    return filled_bubbles

def draw_detected_areas(image, filled_bubbles, sections, img_shape):
    """Draws bounding boxes around detected filled bubbles and overlays grid lines."""
    height, width = img_shape

    for section_name, x1_perc, x2_perc, y1_perc, y2_perc, options, num_questions, _ in sections:
        x1, x2 = int(x1_perc / 100 * width), int(x2_perc / 100 * width)
        y1, y2 = int(y1_perc / 100 * height), int(y2_perc / 100 * height)

        row_height = (y2 - y1) / num_questions
        col_width = (x2 - x1) / len(options)

        for q in range(num_questions):
            for o, option in enumerate(options):
                cx1, cx2 = int(x1 + o * col_width), int(x1 + (o + 1) * col_width)
                cy1, cy2 = int(y1 + q * row_height), int(y1 + (q + 1) * row_height)

                cv2.rectangle(image, (cx1, cy1), (cx2, cy2), (255, 0, 0), 1)  # Draw grid

                if (q + 1, option) in filled_bubbles.get(section_name, []):
                    cv2.rectangle(image, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)  # Highlight filled bubble

    return image

def scan_omr(image_path, output_path="scanned_omr.png", show=False):
    """Processes an OMR sheet, detects filled bubbles, and saves the scanned result."""
    sections = [
        ("MCQ Section 1", 8, 24, 19, 90.5, ["A", "B", "C", "D"], 25, 0.05),
        ("MCQ Section 2", 31, 46.5, 19, 90.5, ["A", "B", "C", "D"], 25, 0.05),
        ("Roll Number", 49, 61.5, 24, 38.5, list(range(1, 7)), 10, 0.3),
        ("Registration", 49, 61.5, 45.5, 60.2, list(range(1, 7)), 10, 0.3),
        ("Subject Code", 67.5, 74, 45.5, 60.2, list(range(1, 4)), 10, 0.25),
        ("Set Code", 67.5, 74, 24, 34, ["Set"], 4, 0.01),
    ]

    # Load the original image
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Error: Unable to read image at path {image_path}")

    # Preprocess the image
    img_thresh = preprocess_image(image_path)

    # Detect filled bubbles with section-specific fill ratios
    filled_bubbles = detect_filled_bubbles(img_thresh, sections, img_thresh.shape)

    # Draw results
    result_img = draw_detected_areas(original_img, filled_bubbles, sections, img_thresh.shape)

    # Save the processed image
    cv2.imwrite(output_path, result_img)
    print(f"Scanned OMR saved as {output_path}")

    if show:
        cv2.imshow("Scanned OMR", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return filled_bubbles

if __name__ == "__main__":
    try:
        results = scan_omr("omr.jpg", "scanned_omr.png", show=False)
        print("OMR Results:")
        for section, answers in results.items():
            print(f"{section}: {answers}")
    except Exception as e:
        print(f"Error: {e}")
