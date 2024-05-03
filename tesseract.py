import cv2
import easyocr

def resize_image(img, width, height):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def preprocess_image_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text(image_path, areas):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image was not loaded.")
        return {}, None

    image = resize_image(image, 3271, 2300)
    reader = easyocr.Reader(['en'])
    results = {}
    for area in areas:
        y1, x1, y2, x2 = area["coords"]
        crop_img = image[y1:y2, x1:x2]
        processed_img = preprocess_image_for_ocr(crop_img)
        result = reader.readtext(processed_img, detail=0)
        extracted_text = ' '.join(result)
        print(f"Extracted text for {area['key']}: {extracted_text}")  # Добавлено логирование
        results[area["key"]] = extracted_text
    return results

def reformat_date(date_str):
    parts = date_str.split('.')
    if len(parts) == 3:
        day, month, year = parts
        return year[-2:] + month + day
    else:
        return "Invalid date format"

def hashing(str1):
    count = 0
    hash_val = 0
    for element in str1:
        if element.isdigit():
            digit = int(element)
        else:
            digit = ord(element.upper()) - 55
        hash_val += digit * [7, 3, 1][count % 3]
        count += 1
    return hash_val % 10

def create_mrz_2(passN, code, birthYYMMDD, gender, endYYMMDD, personN):
    C1 = str(hashing(passN))
    C2 = str(hashing(birthYYMMDD))
    C3 = str(hashing(endYYMMDD))
    C4 = str(hashing(personN))
    C5 = str(hashing(passN + C1 + birthYYMMDD + C2 + endYYMMDD + C3 + personN + "0000000" + C4))

    MRZ2 = passN + C1 + code + birthYYMMDD + C2 + gender + endYYMMDD + C3 + personN
    MRZ2 = MRZ2 + (44-len(MRZ2)-2)*'<' + C4 + C5

    return MRZ2

def main(file_path):
    azerbaijan_areas = [
        {"key": "Passport Number", "coords": [259, 2049, 365, 2727]},
        {"key": "Country Code", "coords": [267, 1349, 364, 1553]},
        {"key": "Date of Birth", "coords": [1060, 933, 1154, 1482]},
        {"key": "Personal Number", "coords": [1057, 1528, 1152, 2203]},
        {"key": "Sex", "coords": [1059, 2285, 1150, 2508]},
        {"key": "Date of Expiry", "coords": [1362, 1547, 1467, 2113]},
        {"key": "MRZ Line-1", "coords": [1714, 48, 1962, 3154]},
        {"key": "MRZ Line-2", "coords": [1982, 124, 2106, 3091]}
    ]
    areas = azerbaijan_areas
    data = extract_text(file_path, areas)

    passport_number = data.get("Passport Number", "")
    code = data.get("Country Code", "")
    dob = reformat_date(data.get("Date of Birth", ""))
    personal_number = data.get("Personal Number", "")
    sex = data.get("Sex", "").strip()
    sex = 'M' if sex == 'KIM' or sex == 'MIM' else 'F'
    expiry = reformat_date(data.get("Date of Expiry", ""))
    mrz_line_2_extracted = data.get("MRZ Line-2", "").replace(' ', '')
    mrz_line_2_extracted=mrz_line_2_extracted[:10]+'AZE'+ mrz_line_2_extracted[13:]

    mrz_line_2_calculated = create_mrz_2(passport_number, code, dob, sex, expiry, personal_number)
    print(f"Extracted MRZ-2: {mrz_line_2_extracted}")
    print(f"Calculated MRZ-2: {mrz_line_2_calculated}")
    # mrz_line_2_calculated = mrz_line_2_calculated.strip().replace('7', 'Z')
    # mrz_line_2_extracted = mrz_line_2_extracted.strip().replace('7', 'Z')

    print("Match:", mrz_line_2_extracted == mrz_line_2_calculated)

    return mrz_line_2_extracted == mrz_line_2_calculated
