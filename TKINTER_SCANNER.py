import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk
import cv2
import numpy as np


# --- Globals ---
img = None
edge_img = None
display_img = None
tk_img = None
width = height = 0
scanning = False
animation_data = []  # stores list of contour point sequences
current_contour_index = 0
current_point_index = 0


# --- ML-style Nearest Neighbor Traversal using Extreme Points ---
def nearest_neighbor_order(contours):
    """Sort contours using nearest-neighbor traversal based on contour extreme points."""
    if not contours:
        return []

    contour_data = []
    for contour in contours:
        # Extract extreme points (leftmost, rightmost, topmost, bottommost)
        topmost = tuple(contour[contour[:, :, 1].argmin()][0])
        contour_data.append((contour, [ topmost,]))

    n = len(contour_data)
    visited = np.zeros(n, dtype=bool)
    order = []

    # Start with contour having the smallest (x + y) of its leftmost point (top-left most contour)
    start_idx = np.argmin([p[1][0][0] + p[1][0][1] for p in contour_data])
    current_idx = start_idx

    for _ in range(n):
        visited[current_idx] = True
        order.append(contour_data[current_idx])

        unvisited = np.where(~visited)[0]
        if len(unvisited) == 0:
            break

        # Compute distance to next contour based on nearest extreme point distance
        current_extremes = np.array(contour_data[current_idx][1])
        dists = []
        for j in unvisited:
            next_extremes = np.array(contour_data[j][1])
            # pairwise distances between extreme points
            dist_matrix = np.linalg.norm(
                current_extremes[:, None, :] - next_extremes[None, :, :], axis=2
            )
            dists.append(np.min(dist_matrix))

        current_idx = unvisited[np.argmin(dists)]

    return [c[0] for c in order]


# --- Image Loading ---
def open_image():
    global edge_img, display_img, tk_img, width, height, animation_data, scanning

    file_path = filedialog.askopenfilename(
        filetypes=[("JPEG Files", "*.jpg;*.jpeg"), ("All Files", "*.*")]
    )
    if not file_path:
        return

    image = cv2.imread(file_path)
    resized_img = cv2.resize(image, (1600, 1200))
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Detect contours with hierarchy
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        messagebox.showwarning("No Contours", "No edges detected in this image.")
        return

    hierarchy = hierarchy[0]
    external_contours = []

    # Only external contours (no parent)
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 30:
            continue
        if hierarchy[i][3] == -1:  # no parent → external
            external_contours.append(contour)

    # ML-style nearest neighbor ordering using extreme points
    ordered_contours = nearest_neighbor_order(external_contours)

    # Build animation data (scaled for Tkinter display)
    animation_data.clear()
    for contour in ordered_contours:
        contour_points = []
        for p in contour:
            x, y = p[0]
            x_scaled = int(x * 800 / 1600)
            y_scaled = int(y * 600 / 1200)
            contour_points.append((x_scaled, y_scaled))
        animation_data.append(contour_points)

    # Create black-white edge image for display background
    edge_canvas = np.zeros((1200, 1600, 3), dtype=np.uint8)
    for contour in ordered_contours:
        cv2.drawContours(edge_canvas, [contour], -1, (255, 255, 255), 1)

    img_rgb = cv2.cvtColor(edge_canvas, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((800, 600))
    display_img = img_pil.copy()
    edge_img = img_pil.copy()
    width, height = display_img.size

    tk_img = ImageTk.PhotoImage(display_img)
    canvas.create_image(0, 0, anchor="nw", image=tk_img)

    scanning = False
    status_label.config(
        text=f"External contours detected: {len(animation_data)} | Ordered by Extreme-Point Nearest-Neighbor ✅"
    )


# --- Animation ---
def start_animation():
    global scanning, current_contour_index, current_point_index
    if edge_img is None:
        messagebox.showwarning("No Image", "Please open an image first.")
        return
    if scanning:
        return

    scanning = True
    current_contour_index = 0
    current_point_index = 0
    status_label.config(text="Animating red dots along external contours...")
    animate_next()


def animate_next():
    global scanning, current_contour_index, current_point_index, tk_img

    if not scanning or current_contour_index >= len(animation_data):
        scanning = False
        status_label.config(text="Animation complete ✅")
        return

    frame = display_img.copy()
    draw = ImageDraw.Draw(frame)

    # Draw completed contours fully
    for i in range(current_contour_index):
        for (x, y) in animation_data[i]:
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="red")

    # Draw current contour up to current_point_index
    contour_points = animation_data[current_contour_index]
    for i in range(current_point_index):
        x, y = contour_points[i]
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="red")

    # Update Tkinter canvas
    tk_img = ImageTk.PhotoImage(frame)
    canvas.create_image(0, 0, anchor="nw", image=tk_img)

    # Move forward along contour
    current_point_index += 8  # number of dots per frame (smaller = slower)
    if current_point_index >= len(contour_points):
        current_point_index = 0
        current_contour_index += 1

    if scanning:
        root.after(1, animate_next)  # delay per frame (larger = slower)


def stop_animation():
    global scanning
    scanning = False
    status_label.config(text="Animation stopped ⛔")


# --- GUI Setup ---
root = tk.Tk()
root.title("External Contour Animation (Extreme-Point Nearest-Neighbor)")

frame = tk.Frame(root)
frame.pack(pady=10)

open_btn = tk.Button(frame, text="Open JPG", command=open_image)
open_btn.grid(row=0, column=0, padx=5)

start_btn = tk.Button(frame, text="Start Animation", command=start_animation)
start_btn.grid(row=0, column=1, padx=5)

stop_btn = tk.Button(frame, text="Stop", command=stop_animation)
stop_btn.grid(row=0, column=2, padx=5)

canvas = tk.Canvas(root, width=800, height=600, bg="black")
canvas.pack(padx=10, pady=10)

status_label = tk.Label(root, text="No image loaded.")
status_label.pack(pady=5)

root.mainloop()
