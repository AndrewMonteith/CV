import os

import cv2
from os.path import join

from disparitymaps import calc_depth, calc_disparity_map
from preprocessing import prp_dist_calc_input, prp_obj_detection_input
from yoloimg import yolo_detect

# As request by the program specification.
master_path_to_dataset = "C:\\Users\\Hp\\Downloads\\TTBB - Performance"


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
        if distances[i] == 100_000_000:
            continue

        draw_box(left_img, obj_class, distances[i], *box)


def process_image(l_img, gr_img):
    prp_obj_detection_input(l_img, gr_img)

    gl_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
    prp_dist_calc_input(gl_img, gr_img)

    # array of (<class-match>, <box>)
    objects = yolo_detect(l_img)

    disparity_map = calc_disparity_map(gl_img, gr_img)

    distances = [calc_depth(disparity_map, box) for (_, box) in objects]

    return objects, distances, disparity_map


def false_colour(image):
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = image.astype('uint8')
    return cv2.applyColorMap(image, cv2.COLORMAP_BONE)


def run_simulation():
    l_imgs_folder = join(master_path_to_dataset, "left-images")
    r_imgs_folder = join(master_path_to_dataset, "right-images")

    processed_colour = join(master_path_to_dataset, "processed-color")
    processed_disp_map = join(master_path_to_dataset, "processed-disp-map")

    cv2.namedWindow("Colour Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Disparity Map", cv2.WINDOW_NORMAL)

    # Useful NB: Images are passed by reference
    for l_img_name in os.listdir(l_imgs_folder):
        r_img_name = l_img_name.replace("L", "R")

        l_img, r_img = cv2.imread(join(l_imgs_folder, l_img_name)), cv2.imread(join(r_imgs_folder, r_img_name), 0)

        objects, distances, disparity_map = process_image(l_img.copy(), r_img)

        print(disparity_map[:, 0])

        annotate_image(l_img, objects, distances)

        min_index = distances.index(min(distances))
        closest_object_class = objects[min_index][0]
        closest_object_dist = distances[min_index]

        print(l_img_name)
        print(f"{r_img_name} : {closest_object_class} ({closest_object_dist}m)")

        cv2.resizeWindow("Colour Image", l_img.shape[1], l_img.shape[0])
        cv2.resizeWindow("Disparity Map", disparity_map.shape[1], disparity_map.shape[0])

        cv2.imshow("Colour Image", l_img)
        cv2.imshow("Disparity Map", false_colour(disparity_map))

        cv2.waitKey(40)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_simulation()
