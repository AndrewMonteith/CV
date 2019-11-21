import ntpath
import os

import cv2

from densedisparitydistance import calculate_distance
from yoloimg import yolo_detect


def does_folder_exist(path, folder_name):
    folder_path = os.path.join(path, folder_name)

    return os.path.exists(folder_path) and os.path.isdir(folder_path)


master_path_to_dataset = "C:\\Users\\Hp\\Downloads\\TTBB-shorter"


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


def get_folder_paths():
    left_images_path = os.path.join(master_path_to_dataset, "left-images")
    right_images_path = os.path.join(master_path_to_dataset, "right-images")

    return left_images_path, right_images_path


def setup_simulation_environment():
    """
        Make sure the right folders are there
        Creates an output directory for all processed images
    """
    l_imgs_path, r_imgs_path = get_folder_paths()

    if not (os.path.isdir(l_imgs_path) and os.path.isdir(r_imgs_path)):
        raise Exception("Failed to run simulation, folder did not contain left-images and right-images folder")

    processed_img_dir = os.path.join(master_path_to_dataset, "processed")
    if os.path.exists(processed_img_dir):
        for file in os.listdir(processed_img_dir):
            os.remove(os.path.join(processed_img_dir, file))
    else:
        os.mkdir(processed_img_dir)

    return processed_img_dir


def all_image_pairs():
    l_imgs_path, r_imgs_path = get_folder_paths()

    for l_img_name in os.listdir(l_imgs_path):
        r_img_name = l_img_name.replace("L", "R")

        l_img_path = os.path.join(l_imgs_path, l_img_name)
        r_img_path = os.path.join(r_imgs_path, r_img_name)

        yield l_img_path, r_img_path


def run_simulation(preprocess_image, detect_objects, estimate_distance):
    """
    Runs a simulation pass over the images contained in master_path_to_data
    :param preprocess_image: Function that takes a pair of images and performs any
                          preprocessing on them to make object detection easier
    :param detect_objects: Function that takes colour image and detects
                             any dynamic objects in them
    :param estimate_distance: Function that takes a pair of images and objects
                                we detected in them and attempts to determine the
                                distance between them
    """

    output_folder = setup_simulation_environment()

    def process_image(l_img, r_img):
        preprocessed_l_img = preprocess_image(l_img)

        objects = detect_objects(preprocessed_l_img)

        distances = [estimate_distance(l_img, r_img, box) for (_, box) in objects]

        annotate_image(l_img, objects, distances)

        return l_img

    for l_img_path, r_img_path in all_image_pairs():
        print("Processing image ", l_img_path)
        l_img, r_img = cv2.imread(l_img_path), cv2.imread(r_img_path)

        annotated_image = process_image(l_img, r_img)

        cv2.imwrite(os.path.join(output_folder, ntpath.basename(l_img_path)), annotated_image)


def no_preprocessing(left_img):
    return left_img


if __name__ == "__main__":
    run_simulation(no_preprocessing, yolo_detect, calculate_distance)

# Code to a file of images into a video
# img_arr = os.listdir(os.path.join(master_path_to_dataset, "left-images"))
# img_array = []
# size = None
# for filename in img_arr:
#     img = cv2.imread(os.path.join(master_path_to_dataset, "left-images", filename))
#     height, width, layers = img.shape
#     size = (width, height)
#     img_array.append(img)
# out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
#
# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()
