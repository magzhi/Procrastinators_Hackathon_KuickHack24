import random
import get_data

from get_data import *
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

def make_thumbnail(img, size=(400, 400)):
    # открываем изображение
    # определение соотношения сторон
    width, height = img.size
    if width > height:
        ratio = width / size[0]
        new_height = int(height / ratio)
        new_size = (size[0], new_height)
    else:
        ratio = height / size[1]
        new_width = int(width / ratio)
        new_size = (new_width, size[1])
 
    # изменение размера изображения
    img.thumbnail(new_size, Image.LANCZOS)
    return img

def mistake_generator_aze(): 
    correct= [50,50,65,[650,295],[650,350],[650,455],[650,510],[650,723],[1050,723],[655,920],[1050,920],ImageFont.truetype('Helvetica-Bold.ttf', 50),[1428,187],44] 
    for i in range(3): 
        mistake= random.randint(0,len(correct)-1) 
        if type(correct[mistake]) == int: 
            probab= random.randint(0,1) 
            if probab == 0: 
                correct[mistake] += random.randint(-15,-5) 
            else: 
                correct[mistake] += random.randint(5,15) 
        elif type(correct[mistake]) == list: 
            probab= random.randint(0,1) 
            replace= random.randint(0,1) 
            if probab==0: 
                correct[mistake][replace] += random.randint(-15,-8) 
            else: 
                correct[mistake][replace] += random.randint(8,15) 
        else: 
            correct[mistake] = ImageFont.truetype('Helvetica.ttf', 50) 
    return correct 
 
 
def imageMergerFakeSpelling(countryID,file_name,path,name,surname):
    bgID = random.randint(0,99)
    temp = ""
    if bgID > 9:
        temp = str(bgID)
    else:
        temp = "0" + str(bgID)
    search = "srb_passport/" + temp + ".jpg"
    background = Image.open(search)
    foreground = Image.open("srbE.png")
    background.paste(foreground, (0, 0), foreground)

    params= mistake_generator_aze()

    I1 = ImageDraw.Draw(background)
    innerFont = ImageFont.truetype('Helvetica.ttf', params[0])
    boldFont = ImageFont.truetype('Helvetica-Bold.ttf', params[1])
    mrzFont = ImageFont.truetype('OCRBRegular.ttf', params[2])

    start, end, birthd = random_dates()
    passN = passport_number()
    personN = personal_number()

    I1.text(params[3], surname.upper(), font=innerFont, fill=(255,0,0))
    I1.text(params[4], surname.upper(), font=innerFont, fill=(255,0,0))
    I1.text(params[5], name.upper(), font=innerFont, fill=(255,0,0))
    I1.text(params[6], name.upper(), font=innerFont, fill=(255,0,0))
    I1.text(params[7], birthd, font=innerFont, fill=(255,0,0))
    I1.text(params[8], personN, font=innerFont, fill=(255,0,0))
    I1.text(params[9], start, font=innerFont, fill=(255,0,0))
    I1.text(params[10], end, font=params[11], fill=(255,0,0))
    I1.text(params[12], passN, font=innerFont, fill=(255,0,0))

    # 16.09.1994
    # 1234567890
    MRZ1 = surname.upper() + "<<" + name.upper()
    MRZ1count = 44-len(MRZ1)-5
    while MRZ1count > 0:
        MRZ1 += "<"
        MRZ1count -= 1
    birthYYMMDD = birthd[8] + birthd[9] + birthd[3] + birthd[4] + birthd[0] + birthd[1]
    endYYMMDD = end[8] + end[9] + end[3] + end[4] + end[0] + end[1]

    gender = "M" if bgID%10 in [0,1,2,3,4] else "F"

    C1 = str(hashing("C" + str(passN)))
    C2 = str(hashing(birthYYMMDD))
    C3 = str(hashing(endYYMMDD))
    C4 = str(hashing(str(personN)))
    C5 = str(hashing("C" + str(passN)+C1+birthYYMMDD+C2+endYYMMDD+C3+str(personN)+"0000000"+C4))

    MRZ2 = str(passN) + C1 + "   " + birthYYMMDD + C2 + gender + endYYMMDD + C3 + str(personN) + "<<<<<<<" + C4 + C5


    x = 330
    y = 1231
    for element in MRZ1:
        I1.text((x,y),str(element),font=mrzFont,fill=(255,0,0))
        x += params[13]
    x = 148
    y = 1333
    for element in MRZ2:
        I1.text((x,y),str(element),font=mrzFont,fill=(255,0,0))
        x += 44



    background = background.convert('RGB')
    background.save(f"{path}/{file_name}.png")
 

def imageMergerFakeBacked(countryID,file_name,path,name,surname):
    bgID = random.randint(0,99)
    temp = ""
    if bgID > 9:
        temp = str(bgID)
    else:
        temp = "0" + str(bgID)
    search = "aze_passport/" + temp + ".jpg"
    background = Image.open(search)
    foreground = Image.open(f"aze{random.randint(1,10)}.png")
    background.paste(foreground, (0, 0), foreground)

    I1 = ImageDraw.Draw(background)
    innerFont = ImageFont.truetype('Helvetica.ttf', 50)
    boldFont = ImageFont.truetype('Helvetica-Bold.ttf', 50)
    mrzFont = ImageFont.truetype('OCRBRegular.ttf', 65)

    start, end, birthd = random_dates()
    passN = passport_number()
    personN = personal_number()

    I1.text((650,295), surname.upper(), font=innerFont, fill=(255,0,0))
    I1.text((650,350), surname.upper(), font=innerFont, fill=(255,0,0))
    I1.text((650,455), name.upper(), font=innerFont, fill=(255,0,0))
    I1.text((650,510), name.upper(), font=innerFont, fill=(255,0,0))
    I1.text((650,723), birthd, font=innerFont, fill=(255,0,0))
    I1.text((1050,723), personN, font=innerFont, fill=(255,0,0))
    I1.text((655,920), start, font=innerFont, fill=(255,0,0))
    I1.text((1050,920), end, font=boldFont, fill=(255,0,0))
    I1.text((1428,187), passN, font=innerFont, fill=(255,0,0))

    # 16.09.1994
    # 1234567890
    MRZ1 = surname.upper() + "<<" + name.upper()
    MRZ1count = 44-len(MRZ1)-5
    while MRZ1count > 0:
        MRZ1 += "<"
        MRZ1count -= 1
    birthYYMMDD = birthd[8] + birthd[9] + birthd[3] + birthd[4] + birthd[0] + birthd[1]
    endYYMMDD = end[8] + end[9] + end[3] + end[4] + end[0] + end[1]

    gender = "M" if bgID%10 in [0,1,2,3,4] else "F"

    C1 = str(hashing("C" + str(passN)))
    C2 = str(hashing(birthYYMMDD))
    C3 = str(hashing(endYYMMDD))
    C4 = str(hashing(str(personN)))
    C5 = str(hashing("C" + str(passN)+C1+birthYYMMDD+C2+endYYMMDD+C3+str(personN)+"0000000"+C4))

    MRZ2 = str(passN) + C1 + "   " + birthYYMMDD + C2 + gender + endYYMMDD + C3 + str(personN) + "<<<<<<<" + C4 + C5


    x = 330
    y = 1231
    for element in MRZ1:
        I1.text((x,y),str(element),font=mrzFont,fill=(255,0,0))
        x += 44
    x = 148
    y = 1333
    for element in MRZ2:
        I1.text((x,y),str(element),font=mrzFont,fill=(255,0,0))
        x += 44
    

    background = background.convert('RGB')
    background.save(f"{path}/{file_name}.png")


def imageMergerFakeExpired(countryID,file_name,path,name,surname):
    bgID = random.randint(0,99)
    temp = ""
    if bgID > 9:
        temp = str(bgID)
    else:
        temp = "0" + str(bgID)
    search = "aze_passport/" + temp + ".jpg"
    background = Image.open(search)
    foreground = Image.open(f"aze{random.randint(1,5)}.png")
    background.paste(foreground, (0, 0), foreground)

    I1 = ImageDraw.Draw(background)
    innerFont = ImageFont.truetype('Helvetica.ttf', 50)
    boldFont = ImageFont.truetype('Helvetica-Bold.ttf', 50)
    mrzFont = ImageFont.truetype('OCRBRegular.ttf', 65)

    name, surname = random_name()
    start, end, birthd = random_dates_expired()
    passN = passport_number()
    personN = personal_number()

    I1.text((650,295), surname.upper(), font=innerFont, fill=(255,0,0))
    I1.text((650,350), surname.upper(), font=innerFont, fill=(255,0,0))
    I1.text((650,455), name.upper(), font=innerFont, fill=(255,0,0))
    I1.text((650,510), name.upper(), font=innerFont, fill=(255,0,0))
    I1.text((650,723), birthd, font=innerFont, fill=(255,0,0))
    I1.text((1050,723), personN, font=innerFont, fill=(255,0,0))
    I1.text((655,920), start, font=innerFont, fill=(255,0,0))
    I1.text((1050,920), end, font=boldFont, fill=(255,0,0))
    I1.text((1428,187), passN, font=innerFont, fill=(255,0,0))

    # 16.09.1994
    # 1234567890
    MRZ1 = surname.upper() + "<<" + name.upper()
    MRZ1count = 44-len(MRZ1)-5
    while MRZ1count > 0:
        MRZ1 += "<"
        MRZ1count -= 1
    birthYYMMDD = birthd[8] + birthd[9] + birthd[3] + birthd[4] + birthd[0] + birthd[1]
    endYYMMDD = end[8] + end[9] + end[3] + end[4] + end[0] + end[1]

    gender = "M" if bgID%10 in [0,1,2,3,4] else "F"

    C1 = str(hashing("C" + str(passN)))
    C2 = str(hashing(birthYYMMDD))
    C3 = str(hashing(endYYMMDD))
    C4 = str(hashing(str(personN)))
    C5 = str(hashing("C" + str(passN)+C1+birthYYMMDD+C2+endYYMMDD+C3+str(personN)+"0000000"+C4))

    MRZ2 = str(passN) + C1 + "   " + birthYYMMDD + C2 + gender + endYYMMDD + C3 + str(personN) + "<<<<<<<" + C4 + C5


    x = 330
    y = 1231
    for element in MRZ1:
        I1.text((x,y),str(element),font=mrzFont,fill=(255,0,0))
        x += 44
    x = 148
    y = 1333
    for element in MRZ2:
        I1.text((x,y),str(element),font=mrzFont,fill=(255,0,0))
        x += 44



    background = background.convert('RGB')
    background.save(f"{path}/{file_name}.png")


def imageMergerFakeHashing(countryID,file_name,path,name,surname):
    bgID = random.randint(0,99)
    temp = ""
    if bgID > 9:
        temp = str(bgID)
    else:
        temp = "0" + str(bgID)
    search = "aze_passport/" + temp + ".jpg"
    background = Image.open(search)
    foreground = Image.open(f"aze{random.randint(1,5)}.png")
    background.paste(foreground, (0, 0), foreground)

    I1 = ImageDraw.Draw(background)
    innerFont = ImageFont.truetype('Helvetica.ttf', 50)
    boldFont = ImageFont.truetype('Helvetica-Bold.ttf', 50)
    mrzFont = ImageFont.truetype('OCRBRegular.ttf', 65)

    start, end, birthd = random_dates_expired()
    passN = passport_number()
    personN = personal_number()

    I1.text((650,295), surname.upper(), font=innerFont, fill=(255,0,0))
    I1.text((650,350), surname.upper(), font=innerFont, fill=(255,0,0))
    I1.text((650,455), name.upper(), font=innerFont, fill=(255,0,0))
    I1.text((650,510), name.upper(), font=innerFont, fill=(255,0,0))
    I1.text((650,723), birthd, font=innerFont, fill=(255,0,0))
    I1.text((1050,723), personN, font=innerFont, fill=(255,0,0))
    I1.text((655,920), start, font=innerFont, fill=(255,0,0))
    I1.text((1050,920), end, font=boldFont, fill=(255,0,0))
    I1.text((1428,187), passN, font=innerFont, fill=(255,0,0))

    # 16.09.1994
    # 1234567890
    MRZ1 = surname.upper() + "<<" + name.upper()
    MRZ1count = 44-len(MRZ1)-5
    while MRZ1count > 0:
        MRZ1 += "<"
        MRZ1count -= 1
    birthYYMMDD = birthd[8] + birthd[9] + birthd[3] + birthd[4] + birthd[0] + birthd[1]
    endYYMMDD = end[8] + end[9] + end[3] + end[4] + end[0] + end[1]

    gender = "M" if bgID%10 in [0,1,2,3,4] else "F"

    C1 = str(hashing_mistake("C" + str(passN)))
    C2 = str(hashing_mistake(birthYYMMDD))
    C3 = str(hashing_mistake(endYYMMDD))
    C4 = str(hashing_mistake(str(personN)))
    C5 = str(hashing_mistake("C" + str(passN)+C1+birthYYMMDD+C2+endYYMMDD+C3+str(personN)+"0000000"+C4))

    MRZ2 = str(passN) + C1 + "   " + birthYYMMDD + C2 + gender + endYYMMDD + C3 + str(personN) + "<<<<<<<" + C4 + C5


    x = 330
    y = 1231
    for element in MRZ1:
        I1.text((x,y),str(element),font=mrzFont,fill=(255,0,0))
        x += 44
    x = 148
    y = 1333
    for element in MRZ2:
        I1.text((x,y),str(element),font=mrzFont,fill=(255,0,0))
        x += 44



    background = background.convert('RGB')
    background.save(f"{path}/{file_name}.png")



def imageMergerReal(countryID,file_name,path,name,surname):
    bgID = random.randint(0,99)
    temp = ""
    if bgID > 9:
        temp = str(bgID)
    else:
        temp = "0" + str(bgID)
    search = "aze_passport/" + temp + ".jpg"
    background = Image.open(search)
    foreground = Image.open("azeE.png")
    background.paste(foreground, (0, 0), foreground)

    I1 = ImageDraw.Draw(background)
    innerFont = ImageFont.truetype('Helvetica.ttf', 50)
    boldFont = ImageFont.truetype('Helvetica-Bold.ttf', 50)
    mrzFont = ImageFont.truetype('OCRBRegular.ttf', 65)

    start, end, birthd = random_dates()
    passN = passport_number()
    personN = personal_number()

    I1.text((650,295), surname.upper(), font=innerFont, fill=(255,0,0))
    I1.text((650,350), surname.upper(), font=innerFont, fill=(255,0,0))
    I1.text((650,455), name.upper(), font=innerFont, fill=(255,0,0))
    I1.text((650,510), name.upper(), font=innerFont, fill=(255,0,0))
    I1.text((650,723), birthd, font=innerFont, fill=(255,0,0))
    I1.text((1050,723), personN, font=innerFont, fill=(255,0,0))
    I1.text((655,920), start, font=innerFont, fill=(255,0,0))
    I1.text((1050,920), end, font=boldFont, fill=(255,0,0))
    I1.text((1428,187), passN, font=innerFont, fill=(255,0,0))

    # 16.09.1994
    # 1234567890
    MRZ1 = surname.upper() + "<<" + name.upper()
    MRZ1count = 44-len(MRZ1)-5
    while MRZ1count > 0:
        MRZ1 += "<"
        MRZ1count -= 1
    birthYYMMDD = birthd[8] + birthd[9] + birthd[3] + birthd[4] + birthd[0] + birthd[1]
    endYYMMDD = end[8] + end[9] + end[3] + end[4] + end[0] + end[1]

    gender = "M" if bgID%10 in [0,1,2,3,4] else "F"

    C1 = str(hashing("C" + str(passN)))
    C2 = str(hashing(birthYYMMDD))
    C3 = str(hashing(endYYMMDD))
    C4 = str(hashing(str(personN)))
    C5 = str(hashing("C" + str(passN)+C1+birthYYMMDD+C2+endYYMMDD+C3+str(personN)+"0000000"+C4))

    MRZ2 = str(passN) + C1 + "   " + birthYYMMDD + C2 + gender + endYYMMDD + C3 + str(personN) + "<<<<<<<" + C4 + C5


    x = 330
    y = 1231
    for element in MRZ1:
        I1.text((x,y),str(element),font=mrzFont,fill=(255,0,0))
        x += 44
    x = 148
    y = 1333
    for element in MRZ2:
        I1.text((x,y),str(element),font=mrzFont,fill=(255,0,0))
        x += 44



    background = background.convert('RGB')
    background.save(f"{path}/{file_name}.png")

def hashing(str1):
    count = 0
    hash = 0
    for element in str1:
        if count % 3 == 0:
            if element.isdigit():
                hash += int(element)*7
            else:
                hash += (ord(element)-55)*7
        elif count % 3 == 1:
            if element.isdigit():
                hash += int(element)*3
            else:
                hash += (ord(element)-55)*3
        elif count % 3 == 2:
            if element.isdigit():
                hash += int(element)*1
            else:
                hash += (ord(element)-55)*1
        count += 1
    return hash % 10

def create_mrz_2(passN,code,birthYYMMDD,gender,endYYMMDD,personN):
    C1 = str(hashing("C" + str(passN)))
    C2 = str(hashing(birthYYMMDD))
    C3 = str(hashing(endYYMMDD))
    C4 = str(hashing(str(personN)))
    C5 = str(hashing("C" + str(passN)+C1+birthYYMMDD+C2+endYYMMDD+C3+str(personN)+"0000000"+C4))

    MRZ2 = 'C'+str(passN) + C1 + code + birthYYMMDD + C2 + gender + endYYMMDD + C3 + str(personN) + "<<<<<<<" + C4 + C5

    return MRZ2


def hashing_mistake(str1):
    count = 0
    hash = 0
    for element in str1:
        if count % 3 == 0:
            if element.isdigit():
                hash += int(element)*random.randint(1,7)
            else:
                hash += (ord(element)-55)*random.randint(7,13)
        elif count % 3 == 1:
            if element.isdigit():
                hash += int(element)*random.randint(3,9)
            else:
                hash += (ord(element)-55)*random.randint(3,9)
        elif count % 3 == 2:
            if element.isdigit():
                hash += int(element)*random.randint(1,9)
            else:
                hash += (ord(element)-55)* random.randint(1,9)
        count += 1
    return hash % 10

name,surname= random_name()

