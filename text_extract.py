from google.cloud import vision
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vision-key.json'

def detect_text(path):
    """Detects text in the file and extracts username from first line and full clan name from second line."""

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    if not texts:
        print("No text detected in the image.")
        return None, None

    # Get the full text (first annotation contains all detected text)
    full_text = texts[0].description
    
    # Split text into lines
    lines = [line.strip() for line in full_text.strip().split('\n') if line.strip()]
    
    # Extract first word from first line (username)
    username = None
    if len(lines) > 0:
        first_line_words = lines[0].split()
        if first_line_words:
            username = first_line_words[0].lower()
    
    # Extract full second line (clan name)
    clan_name = None
    if len(lines) > 1:
        clan_name = lines[1]  # Get the entire second line
    
    # Print results
    print("=" * 60)
    print("Extracted Text:")
    print("=" * 60)
    print(f"Full text:\n{full_text}")
    print("=" * 60)
    print(f"\nUsername (line 1): {username}")
    print(f"Clan name (line 2): {clan_name}")
    print("=" * 60)
    
    return username, clan_name

if __name__ == "__main__":
    username, clan_name = detect_text('./dhanika.png')
    
    if username and clan_name:
        print(f"\nResult:")
        print(f"  Username: {username}")
        print(f"  Clan: {clan_name}")
    elif username:
        print(f"\nResult:")
        print(f"  Username: {username}")
        print(f"  Clan: (not found)")
    else:
        print("\nNo text extracted.")