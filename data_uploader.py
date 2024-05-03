from PIL import Image
 
def make_thumbnail(img, size=(128, 128)):
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
    img.show()
    return img

