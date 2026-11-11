"""
Generates rich sample images for zero-shot object detection testing:
- Living room scene (couch, plant, lamp, coffee table, mug)
- Street traffic scene (car, bus, bicycle, pedestrian, traffic light)
- Work desk scene (laptop, keyboard, mouse, headphones, notepad)
- Animal scene (cat, dog, bird, bowl)
"""

import os
from PIL import Image, ImageDraw, ImageFont


def create_samples(output_dir: str = "samples"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Office / Desk Scene
    desk_img = Image.new("RGB", (800, 600), color=(240, 238, 233))
    draw = ImageDraw.Draw(desk_img)
    # Desk wooden surface
    draw.rectangle([0, 180, 800, 600], fill=(215, 185, 150))
    # Wall background
    draw.rectangle([0, 0, 800, 180], fill=(235, 238, 242))

    # Laptop
    draw.rectangle([250, 220, 550, 420], fill=(50, 55, 65))
    draw.rectangle([265, 235, 535, 395], fill=(30, 120, 200)) # Screen
    draw.rectangle([200, 420, 600, 470], fill=(80, 85, 95)) # Base/Keyboard

    # Coffee Mug
    draw.ellipse([640, 280, 720, 360], fill=(220, 50, 50))
    draw.ellipse([650, 290, 710, 350], fill=(100, 50, 30)) # Coffee inside

    # Notebook
    draw.rectangle([80, 320, 200, 480], fill=(245, 245, 245), outline=(150, 150, 150), width=2)
    # Pen
    draw.rectangle([210, 340, 222, 460], fill=(30, 30, 30))

    # Mouse
    draw.ellipse([620, 400, 680, 480], fill=(40, 40, 40))

    desk_img.save(os.path.join(output_dir, "office_desk.jpg"), quality=95)

    # 2. Street Traffic Scene
    street_img = Image.new("RGB", (800, 600), color=(135, 206, 235)) # Sky
    draw = ImageDraw.Draw(street_img)
    # Road
    draw.rectangle([0, 320, 800, 600], fill=(60, 64, 70))
    # Road markings
    for x in range(20, 800, 120):
        draw.rectangle([x, 450, x + 60, 465], fill=(240, 240, 240))

    # Buildings in background
    draw.rectangle([30, 120, 220, 320], fill=(180, 160, 150))
    draw.rectangle([250, 80, 480, 320], fill=(150, 170, 190))
    draw.rectangle([520, 140, 750, 320], fill=(200, 190, 170))

    # Car 1 (Red sedan)
    draw.rectangle([100, 390, 320, 480], fill=(220, 40, 40))
    draw.rectangle([140, 340, 280, 390], fill=(200, 30, 30)) # Cabin
    draw.ellipse([130, 460, 180, 510], fill=(20, 20, 20)) # Wheel 1
    draw.ellipse([240, 460, 290, 510], fill=(20, 20, 20)) # Wheel 2

    # Car 2 (Blue SUV)
    draw.rectangle([440, 370, 720, 490], fill=(40, 100, 220))
    draw.rectangle([480, 310, 680, 370], fill=(35, 85, 190))
    draw.ellipse([480, 465, 540, 525], fill=(20, 20, 20))
    draw.ellipse([620, 465, 680, 525], fill=(20, 20, 20))

    # Traffic light
    draw.rectangle([740, 160, 770, 270], fill=(30, 30, 30))
    draw.ellipse([745, 170, 765, 190], fill=(240, 40, 40)) # Red
    draw.ellipse([745, 200, 765, 220], fill=(240, 200, 30)) # Yellow
    draw.ellipse([745, 230, 765, 250], fill=(40, 200, 40)) # Green

    street_img.save(os.path.join(output_dir, "street_traffic.jpg"), quality=95)

    # 3. Living Room & Pet Scene
    living_img = Image.new("RGB", (800, 600), color=(245, 240, 235))
    draw = ImageDraw.Draw(living_img)
    # Floor
    draw.rectangle([0, 380, 800, 600], fill=(160, 110, 80)) # Hardwood

    # Couch / Sofa
    draw.rectangle([100, 260, 500, 450], fill=(70, 120, 140))
    draw.rectangle([120, 320, 480, 440], fill=(55, 100, 120)) # Cushions

    # Dog sitting on the right
    draw.ellipse([580, 380, 700, 480], fill=(180, 120, 60)) # Body
    draw.ellipse([640, 320, 720, 400], fill=(190, 130, 70)) # Head
    draw.ellipse([650, 310, 675, 345], fill=(130, 80, 40)) # Ear

    # House plant in pot
    draw.rectangle([30, 320, 80, 420], fill=(190, 90, 50)) # Pot
    draw.ellipse([10, 210, 100, 330], fill=(40, 140, 60)) # Foliage

    # Wall artwork
    draw.rectangle([200, 60, 400, 180], fill=(255, 255, 255), outline=(60, 60, 60), width=4)
    draw.ellipse([250, 90, 350, 150], fill=(240, 180, 40))

    living_img.save(os.path.join(output_dir, "living_room.jpg"), quality=95)
    print(f"Sample images created successfully in {output_dir}/")


if __name__ == "__main__":
    create_samples()
