import cv2


def calculate_box_positions(image_width, image_height):
    """Calculate box positions for different sections of the OMR sheet."""
    sections = [
        ("MCQ Section 1", 8, 24, 19, 90.5, ["A", "B", "C", "D"], 25),
        ("MCQ Section 2", 31, 46.5, 19, 90.5, ["A", "B", "C", "D"], 25),
        ("Roll Number", 49, 61.5, 24, 38.5, list(range(1, 7)), 10),
        ("Registration", 49, 61.5, 45.5, 60.2, list(range(1, 7)), 10),
        ("Subject Code", 67.5, 74, 45.5, 60.2, list(range(1, 4)), 10),
        ("Set Code", 67.5, 74, 24, 34, ["Set"], 4),
    ]

    boxes = []
    for _, x1, x2, y1, y2, cols, rows in sections:
        x1, x2 = int(x1 * image_width / 100), int(x2 * image_width / 100)
        y1, y2 = int(y1 * image_height / 100), int(y2 * image_height / 100)
        col_width, row_height = (x2 - x1) / len(cols), (y2 - y1) / rows

        for col in range(len(cols)):
            for row in range(rows):
                boxes.append(
                    (
                        int(x1 + col * col_width),
                        int(x1 + (col + 1) * col_width),
                        int(y1 + row * row_height),
                        int(y1 + (row + 1) * row_height),
                    )
                )
    return boxes


def draw_rectangles(image):
    """Draw thick blue rectangles around all detected sections."""
    h, w = image.shape[:2]
    for x1, x2, y1, y2 in calculate_box_positions(w, h):
        cv2.rectangle(
            image, (x1, y1), (x2, y2), (255, 0, 0), 4
        )  # Blue color, thickness = 4
    return image


def overlay_rectangles(image_path, output_path="overlay.jpg", show=False):
    """Load an image, overlay rectangles, and save/display the result."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Error: Could not load {image_path}")

    cv2.imwrite(output_path, draw_rectangles(img))
    print(f"Processed image saved as {output_path}")

    if show:
        cv2.imshow("OMR Sheet", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    overlay_rectangles("image.jpg", "overlay_image.png", show=False)
