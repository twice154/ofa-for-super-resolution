from PIL import Image
for i in range(1, 601):
    img_path = "./readysetgo_120fps_448center/%04d.png" % (i)
    im = Image.open(img_path)
    width, height = im.size   # Get dimensions

    new_width = int(0.25 * width)
    new_height = int(0.25 * width)

    # 4x downscaling
    im = im.resize((new_width, new_height), Image.BICUBIC)
    save_img_path = "./readysetgo_120fps_448center_4xBicubic/%04d.png" % (i)
    im.save(save_img_path)
    print("Image #" + str(i) + " saved.")