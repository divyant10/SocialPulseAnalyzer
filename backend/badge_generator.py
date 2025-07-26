from PIL import Image, ImageDraw, ImageFont
import os

def generate_virality_badge(score, username="user"):
    """
    Generates a virality badge PNG based on score.
    Saves it in /frontend/static/badges/{username}_badge.png
    Returns the relative path for HTML use.
    """

    # Define score levels
    if score >= 60:
        level = "High"
        color = "#16a34a"  # green
    elif score >= 20:
        level = "Average"
        color = "#facc15"  # yellow
    else:
        level = "Low"
        color = "#dc2626"  # red

    # Badge size & background
    width, height = 300, 100
    img = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(img)

    # Load font
    try:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        font = ImageFont.truetype(font_path, size=20)
    except:
        font = ImageFont.load_default()

    # Text content
    text = f"{username}'s Virality:\n{level} ({score}%)"
    lines = text.split("\n")

    # Measure each line using textbbox
    line_sizes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_heights = [bbox[3] - bbox[1] for bbox in line_sizes]
    line_widths = [bbox[2] - bbox[0] for bbox in line_sizes]

    total_height = sum(line_heights) + (len(lines) - 1) * 5
    max_width = max(line_widths)

    x = (width - max_width) // 2
    y = (height - total_height) // 2

    # Draw lines centered
    for i, line in enumerate(lines):
        line_bbox = draw.textbbox((0, 0), line, font=font)
        line_w = line_bbox[2] - line_bbox[0]
        draw.text(((width - line_w) // 2, y), line, font=font, fill="white")
        y += line_heights[i] + 5

    # Save the badge
    save_folder = os.path.join("frontend", "static", "badges")
    os.makedirs(save_folder, exist_ok=True)

    file_path = os.path.join(save_folder, f"{username}_badge.png")
    img.save(file_path)

    return f"/static/badges/{username}_badge.png"
