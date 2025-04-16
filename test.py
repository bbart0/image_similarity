import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
from skimage.metrics import structural_similarity as ssim
import lpips
import numpy as np
import torch

ssim_score =0
psnr_score=0
lpips_score=0
uiqi_score =0


def select_image():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
    return path

def load_and_resize(image_path, size=(300, 300)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be loaded.")
    img = cv2.resize(img, size) # images must be of the same size to be compared for eg ssim
    return img

def compute_similarity(img1, img2):
    global ssim_score, psnr_score, lpips_score

    # img1 img2 are by default loaded in color by imread
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    ssim_score, _ = ssim(gray1, gray2, full=True)
    psnr_score = cv2.PSNR(img1, img2) # psnr in opencv only compares the brightness (??? cpp implementation says something different),
    # r (max pixel value) left as the default 255

    loss_fn_alex = lpips.LPIPS(net='alex')
    normalized_img1 = (img1.astype(np.float32) / 127.5) - 1.0 # why numpy? opencv internally uses numpy arrays
    normalized_img2 = (img2.astype(np.float32) / 127.5) - 1.0
    lpips_score = loss_fn_alex(torch.from_numpy(normalized_img1).permute(2, 0, 1), torch.from_numpy(normalized_img2).permute(2, 0, 1)).item()


def convert_to_tk_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    return ImageTk.PhotoImage(pil_img)

def pick_and_compare():
    path1 = select_image()
    if not path1:
        return
    path2 = select_image()
    if not path2:
        return

    img1 = load_and_resize(path1)
    img2 = load_and_resize(path2)
    compute_similarity(img1, img2)

    img1_tk = convert_to_tk_image(img1)
    img2_tk = convert_to_tk_image(img2)

    image_label1.configure(image=img1_tk)
    image_label1.image = img1_tk

    image_label2.configure(image=img2_tk)
    image_label2.image = img2_tk

    ssim_label = Label(window,  font=("Arial", 11), background='black', fg='white')
    ssim_label.config(text=f"Similarity Score (SSIM): {ssim_score:.4f}")
    ssim_label.pack(pady=10)

    ssim_label = Label(window,  font=("Arial", 11), background='black', fg='white')
    ssim_label.config(text=f"Peak Signal-to-Noise Ratio (PSNR): {psnr_score:.4f}")
    ssim_label.pack(pady=10)

    ssim_label = Label(window,  font=("Arial", 11), background='black', fg='white')
    ssim_label.config(text=f"Lpips: {lpips_score:.4f}")
    ssim_label.pack(pady=10)


window = tk.Tk()
window.title("Image Similarity Checker")
window.configure(background='black')

score_label = Label(window, text="Select two images to compare", font=("Arial", 14), background='black', fg='white')
score_label.pack(pady=10)

frame = tk.Frame(window, bg='black')
frame.pack()

image_label1 = Label(frame)
image_label1.pack(side="left", padx=10)

image_label2 = Label(frame)
image_label2.pack(side="right", padx=10)

compare_button = Button(window, text="Select Images and Compare", command=pick_and_compare, font=("Arial", 12), bg='#404040', fg='white')
compare_button.pack(pady=10)

window.mainloop()
