from PIL import Image, ImageFilter, ImageOps

img_path = '/Users/albertyu/Documents/Projects/cv/utils/img/img.png'
img_save_path = '/Users/albertyu/Documents/Projects/cv/utils/img/img_sketch.png'

# load your image 
img = Image.open(img_path)

def dodge(a, b, alpha):
    return min(int(a*255/(256-b*alpha)), 255)

def draw(img, blur=2, alpha=1.0, img_save_path=img_save_path):
    '''
    comment 
    '''
    img1 = img.convert('L') # grey scale
    img2 = img1.copy()
    img2 = ImageOps.invert(img2)

    for i in range(blur):
        img2 = img2.filter(ImageFilter.BLUR)

    w, h = img1.size 

    for i in range(w):
        for j in range(h):
            a = img1.getpixel((i, j))
            b = img2.getpixel((i, j))
            img1.putpixel((i, j), dodge(a, b, alpha))

    img1.show()
    img1.save(img_save_path)


draw(img)
