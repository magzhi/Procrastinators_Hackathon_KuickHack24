import cv2
import easyocr

def resize_image(img, width, height):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

# def preprocess_image_for_ocr(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     enhanced = clahe.apply(gray)
#     _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return thresh

def correct_gender_in_mrz(mrz_line_2_extracted, expected_gender):
    if mrz_line_2_extracted[20] == '0':
        mrz_line_2_extracted = mrz_line_2_extracted[:20] + expected_gender + mrz_line_2_extracted[21:]
    return mrz_line_2_extracted

def extract_text(image_path, areas):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image was not loaded.")
        return {}, None

    image = resize_image(image, 3271, 2250)
    reader = easyocr.Reader(['en'])
    results = {}
    for area in areas:
        y1, x1, y2, x2 = area["coords"]
        crop_img = image[y1:y2, x1:x2]
        processed_img = crop_img
        result = reader.readtext(processed_img, detail=0)
        extracted_text = ' '.join(result)
        print(f"Extracted text for {area['key']}: {extracted_text}")
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

def main1(image_path):
    azerbaijan_areas = [
        {"key": "Passport Number", "coords": [302, 2080, 368, 2473]},
        {"key": "Country Code", "coords": [300, 1402, 375, 1589]},
        {"key": "Date of Birth", "coords": [830, 991, 886, 1331]},
        {"key": "Personal Number", "coords": [824, 2073, 890, 2595]},
        {"key": "Sex", "coords": [955, 977, 1036, 1152]},
        {"key": "Date of Expiry", "coords": [1549, 982, 1637, 1351]},
        {"key": "MRZ Line-1", "coords": [1817, 88, 1941, 3196]},
        {"key": "MRZ Line-2", "coords": [1995, 112, 2126, 3190]}
    ]
    areas = azerbaijan_areas
    data = extract_text(image_path, areas)

    passport_number = data.get("Passport Number", "")
    code = data.get("Country Code", "")
    dob = reformat_date(data.get("Date of Birth", "")).replace(' ', '')
    personal_number = data.get("Personal Number", "")
    sex = data.get("Sex", "").strip()
    sex = 'M' if sex == 'KIM' or sex[0] == 'M' else 'F'
    expiry = reformat_date(data.get("Date of Expiry", ""))
    mrz_line_2_extracted = data.get("MRZ Line-2", "").replace(' ', '')
    mrz_line_2_extracted = correct_gender_in_mrz(mrz_line_2_extracted, sex)
    mrz_line_2_extracted=mrz_line_2_extracted[:10]+'SRB'+ mrz_line_2_extracted[13:]

    mrz_line_2_calculated = create_mrz_2(passport_number, code, dob, sex, expiry, personal_number)

    print(f"Extracted MRZ-2: {mrz_line_2_extracted}")
    print(f"Calculated MRZ-2: {mrz_line_2_calculated}")

    print("Match:", mrz_line_2_extracted == mrz_line_2_calculated)
    return mrz_line_2_extracted == mrz_line_2_calculated


if __name__ == "__main__":
    main()
