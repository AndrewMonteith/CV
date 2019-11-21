import os

import cv2
from os.path import join

from densedisparitydistance import calc_disparity_map, calc_depth
from preprocessing import prp_dist_calc_input, prp_obj_detection_input
from yoloimg import yolo_detect

master_path_to_dataset = "C:\\Users\\Hp\\Downloads\\TTBB - Bad"


# image: image detection performed on
# class_name: string name of detected object_detection
# left, top, right, bottom: rectangle parameters for detection
# colour: to draw detection rectangle in

def draw_box(image, class_name, distance, left, top, width, height):
    right = left + width
    bottom = top + height

    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), (255, 178, 50), 3)

    # construct label
    label = '%s:%.2f' % (class_name, distance)

    # Display the label at the top of the bounding box
    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, label_size[1])
    cv2.rectangle(image, (left, top - round(1.5 * label_size[1])),
                  (left + round(1.5 * label_size[0]), top + base_line), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


def annotate_image(left_img, objects, distances):
    for i, (obj_class, box) in enumerate(objects):
        if distances[i] == -1:
            continue

        draw_box(left_img, obj_class, distances[i], *box)


def process_image(l_img, gr_img):
    prp_obj_detection_input(l_img)

    gl_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
    prp_dist_calc_input(gl_img, gr_img)

    # array of (<class-match>, <box>)
    objects = yolo_detect(l_img)

    disparity_map = calc_disparity_map(gl_img, gr_img)

    distances = [calc_depth(disparity_map, box) for (_, box) in objects]

    return objects, distances


def run_simulation():
    l_imgs_folder = join(master_path_to_dataset, "left-images")
    r_imgs_folder = join(master_path_to_dataset, "right-images")
    processed_imgs_path = join(master_path_to_dataset, "processed")

    # Useful NB: Images are passed by reference
    for l_img_name in os.listdir(l_imgs_folder):
        print("--------- Processing ", l_img_name)
        r_img_name = l_img_name.replace("L", "R")

        l_img, r_img = cv2.imread(join(l_imgs_folder, l_img_name)), cv2.imread(join(r_imgs_folder, r_img_name), 0)

        objects, distances = process_image(l_img.copy(), r_img)

        annotate_image(l_img, objects, distances)

        cv2.imwrite(join(processed_imgs_path, l_img_name), l_img)


if __name__ == "__main__":
    run_simulation()
