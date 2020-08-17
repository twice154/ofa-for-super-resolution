from PIL import Image
for i in range(1, 601):
    img_path = "./readysetgo_120fps/%04d.png" % (i)
    im = Image.open(img_path)
    width, height = im.size   # Get dimensions

    new_width = 448
    new_height = 448

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    save_img_path = "./readysetgo_120fps_448center/%04d.png" % (i)
    im.save(save_img_path)
    print("Image #" + str(i) + " saved.")