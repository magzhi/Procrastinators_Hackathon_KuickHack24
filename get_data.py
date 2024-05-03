import random
from datetime import datetime, timedelta,date
from string import ascii_letters, whitespace

good_chars = (ascii_letters + whitespace).encode()
junk_chars = bytearray(set(range(0x100)) - set(good_chars))
def clean(text):
    return text.encode('ascii', 'ignore').translate(None, junk_chars).decode()


def random_dates():
    start_birthday=datetime(1963, 1, 1)
    end_birthday=datetime(2008,1,1)
    birthday = datetime.date(start_birthday + timedelta(days=random.randint(0, (end_birthday- start_birthday).days)))
    birthday = str(birthday).split('-')
    # Начинаем с первого января 2020 года
    start_date = datetime(2015, 1, 2)
    # Ограничиваемся одним годом – до 1 января 2021 года
    end_date = datetime(2024, 1, 2)
    start_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    start_date = datetime.date(start_date)
    start_date_massive = str(start_date).split('-')
    if int(start_date_massive[2]) == 1:
        if int(start_date_massive[1]) == 1:
            end_date = date(int(start_date_massive[0]) + 9, 12, 30)
        else:
            end_date = date(int(start_date_massive[0]) + 10, int(start_date_massive[1]) - 1, 27)
    else:
        end_date = date(int(start_date_massive[0]) + 10, int(start_date_massive[1]), int(start_date_massive[2]) - 1)
    end_date_massive = str(end_date).split('-')
    return '.'.join(start_date_massive[::-1]),'.'.join(end_date_massive[::-1]), '.'.join(birthday[::-1])


def random_dates_expired():
    start_birthday=datetime(1963, 1, 1)
    end_birthday=datetime(2008,1,1)
    birthday = datetime.date(start_birthday + timedelta(days=random.randint(0, (end_birthday- start_birthday).days)))
    birthday = str(birthday).split('-')
    # Начинаем с первого января 2020 года
    start_date = datetime(2005, 1, 2)
    # Ограничиваемся одним годом – до 1 января 2021 года
    end_date = datetime(2012, 1, 2)
    start_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    start_date = datetime.date(start_date)
    start_date_massive = str(start_date).split('-')
    end_date=date(random.randint(2023,2024), random.randint(1,12), random.randint(1,27))
    end_date_massive = str(end_date).split('-')
    return '.'.join(start_date_massive[::-1]),'.'.join(end_date_massive[::-1]), '.'.join(birthday[::-1])

def random_name():
    with open('allname.txt','r',encoding='utf-8') as f:
        all_names= f.readlines()
    with open('allsurname.txt','r',encoding='utf-8') as f:
        all_surname= f.readlines()
    names=[]
    surnames=[]

    all_names=list(map(lambda x:x.strip('\n'),all_names))
    all_surname=list(map(lambda x:x.strip('\n'),all_surname))
    
    for i in range(len(all_names)):
        flag=True
        for j in all_names[i]:
            if j not in ascii_letters:
                flag=False
                break
        if flag and all_names[i].strip('\n') != '' and len(all_names[i].split())==1:
            names.append(all_names[i].strip('\n'))
    for i in range(len(all_surname)):
        flag=True
        for j in all_surname[i]:
            if j not in ascii_letters:
                flag=False
                break
        if flag and all_surname[i].strip('\n') != '' :
            surnames.append(all_surname[i].strip('\n'))
    ans=[]
    return names,surnames


def passport_number():
    int_part= random.randint(10000000,99999999)
    return str(int_part)

def personal_number():
    variants= []
    for i in range(65,91):
        if i == 79:
            continue
        variants.append(chr(i))
    for i in range(1,10):
        variants.append(i)
    ans=''
    for i in range(7):
        ans = ans + str(random.choice(variants))
    return ans

