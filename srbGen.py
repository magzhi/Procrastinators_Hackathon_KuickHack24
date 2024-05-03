import random
from datetime import datetime, timedelta, date
from string import ascii_letters, whitespace
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from get_data import *

good_chars = (ascii_letters + whitespace).encode()
junk_chars = bytearray(set(range(0x100)) - set(good_chars))

def passport_number():
    int_part = random.randint(100000000, 999999999)
    return str(int_part)


def personal_number():
    int_part = random.randint(1000000000000, 9999999999999)
    return str(int_part)


def clean(text):
    return text.encode('ascii', 'ignore').translate(None, junk_chars).decode()


def imageMergerReal(name,surname,path,i):
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

    I1 = ImageDraw.Draw(background)
    innerFont = ImageFont.truetype('ARIAL.TTF', 44)
    mrzFont = ImageFont.truetype('OCRBRegular.ttf', 68)

    start, end, birthd = random_dates()
    passN = passport_number()
    personN = personal_number()

    I1.text((662,281), surname.upper(), font=innerFont, fill=(0,0,0))
    I1.text((662,373), name.upper(), font=innerFont, fill=(0,0,0))
    I1.text((662,545), birthd, font=innerFont, fill=(0,0,0))
    I1.text((1390,545), personN, font=innerFont, fill=(0,0,0))
    I1.text((662,940), start, font=innerFont, fill=(0,0,0))
    I1.text((662,1038), end, font=innerFont, fill=(0,0,0))
    I1.text((1390,200), passN, font=innerFont, fill=(0,0,0))

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


    C1 = str(hashing(str(passN)))
    C2 = str(hashing(birthYYMMDD))
    C3 = str(hashing(endYYMMDD))
    C4 = str(hashing(str(personN)))
    C5 = str(hashing(str(passN)+C1+birthYYMMDD+C2+endYYMMDD+C3+str(personN)+"0"+C4))

    MRZ2 = str(passN) + C1 + "   " + birthYYMMDD + C2 + gender + endYYMMDD + C3 + str(personN) + "<" + C4 + C5


    x = 314
    y = 1223
    for element in MRZ1:
        I1.text((x,y),str(element),font=mrzFont,fill=(0,0,0))
        x += 46
    x = 85
    y = 1340
    for element in MRZ2:
        I1.text((x,y),str(element),font=mrzFont,fill=(0,0,0))
        x += 46


    background = background.convert('RGB')
    background.save(f"{path}\{i}.png")

def imageMergerSpelling(name,surname,path,i):
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

    params= mistake_generator_srb()

    I1 = ImageDraw.Draw(background)
    innerFont = ImageFont.truetype('ARIAL.TTF', params[0])
    mrzFont = ImageFont.truetype('OCRBRegular.ttf', params[1])

    start, end, birthd = random_dates()
    passN = passport_number()
    personN = personal_number()

    I1.text(params[2], surname.upper(), font=innerFont, fill=(0,0,0))
    I1.text(params[3], name.upper(), font=innerFont, fill=(0,0,0))
    I1.text(params[4], birthd, font=innerFont, fill=(0,0,0))
    I1.text(params[5], personN, font=innerFont, fill=(0,0,0))
    I1.text(params[6], start, font=innerFont, fill=(0,0,0))
    I1.text(params[7], end, font=innerFont, fill=(0,0,0))
    I1.text(params[8], passN, font=innerFont, fill=(0,0,0))

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


    C1 = str(hashing(str(passN)))
    C2 = str(hashing(birthYYMMDD))
    C3 = str(hashing(endYYMMDD))
    C4 = str(hashing(str(personN)))
    C5 = str(hashing(str(passN)+C1+birthYYMMDD+C2+endYYMMDD+C3+str(personN)+"0"+C4))

    MRZ2 = str(passN) + C1 + "   " + birthYYMMDD + C2 + gender + endYYMMDD + C3 + str(personN) + "<" + C4 + C5


    x = 314
    y = 1223
    for element in MRZ1:
        I1.text((x,y),str(element),font=mrzFont,fill=(0,0,0))
        x += params[9]
    x = 85
    y = 1340
    for element in MRZ2:
        I1.text((x,y),str(element),font=mrzFont,fill=(0,0,0))
        x += params[9]


    background = background.convert('RGB')
    background.save(f"{path}\{i}.png")


def imageMergerBackGround(name,surname,path,i):
    bgID = random.randint(0,99)
    temp = ""
    if bgID > 9:
        temp = str(bgID)
    else:
        temp = "0" + str(bgID)
    search = "srb_passport/" + temp + ".jpg"
    background = Image.open(search)
    foreground = Image.open(f"srb{random.randint(1,10)}.png")
    background.paste(foreground, (0, 0), foreground)

    I1 = ImageDraw.Draw(background)
    innerFont = ImageFont.truetype('ARIAL.TTF', 44)
    mrzFont = ImageFont.truetype('OCRBRegular.ttf', 68)

    start, end, birthd = random_dates()
    passN = passport_number()
    personN = personal_number()

    I1.text((662,281), surname.upper(), font=innerFont, fill=(0,0,0))
    I1.text((662,373), name.upper(), font=innerFont, fill=(0,0,0))
    I1.text((662,545), birthd, font=innerFont, fill=(0,0,0))
    I1.text((1390,545), personN, font=innerFont, fill=(0,0,0))
    I1.text((662,940), start, font=innerFont, fill=(0,0,0))
    I1.text((662,1038), end, font=innerFont, fill=(0,0,0))
    I1.text((1390,200), passN, font=innerFont, fill=(0,0,0))

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


    C1 = str(hashing(str(passN)))
    C2 = str(hashing(birthYYMMDD))
    C3 = str(hashing(endYYMMDD))
    C4 = str(hashing(str(personN)))
    C5 = str(hashing(str(passN)+C1+birthYYMMDD+C2+endYYMMDD+C3+str(personN)+"0"+C4))

    MRZ2 = str(passN) + C1 + "   " + birthYYMMDD + C2 + gender + endYYMMDD + C3 + str(personN) + "<" + C4 + C5


    x = 314
    y = 1223
    for element in MRZ1:
        I1.text((x,y),str(element),font=mrzFont,fill=(0,0,0))
        x += 46
    x = 85
    y = 1340
    for element in MRZ2:
        I1.text((x,y),str(element),font=mrzFont,fill=(0,0,0))
        x += 46

    background = background.convert('RGB')
    background.save(f"{path}\{i}.png")


def mistake_generator_srb(): 
    correct= [44,68,[662,281], [662,373], [662,545], [1390,545],[662,940], [662,1038],[1390,200] ,46] 
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


name,surname= random_name()
for i in range(200):
    imageMergerSpelling(random.choice(name),random.choice(surname),r"C:\Users\srlk0\Desktop\generator\dataset_srb\test\misspeling",i)
    imageMergerReal(random.choice(name),random.choice(surname),r"C:\Users\srlk0\Desktop\generator\dataset_srb\test\correct",i)
    imageMergerBackGround(random.choice(name),random.choice(surname),r"C:\Users\srlk0\Desktop\generator\dataset_srb\test\hashing",i)
    imageMergerSpelling(random.choice(name),random.choice(surname),r"C:\Users\srlk0\Desktop\generator\dataset_srb\val\misspeling",i)
    imageMergerReal(random.choice(name),random.choice(surname),r"C:\Users\srlk0\Desktop\generator\dataset_srb\val\correct",i)
    imageMergerBackGround(random.choice(name),random.choice(surname),r"C:\Users\srlk0\Desktop\generator\dataset_srb\val\hashing",i)

