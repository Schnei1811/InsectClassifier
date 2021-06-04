import cv2
import os
import numpy as np
import argparse
import multiprocessing as mp
from glob import glob
from tqdm import tqdm


def crop_image(img_path):

    source_image = cv2.imread(img_path)
    grayscale_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

    rows_min = grayscale_image.min(axis=0)
    cols_min = grayscale_image.min(axis=1)

    min_x = np.where(cols_min < 100)[0][0]
    max_x = np.where(cols_min < 100)[0][-1]
    min_y = np.where(rows_min < 100)[0][0]
    max_y = np.where(rows_min < 100)[0][-1]

    source_edge_img = source_image[min_x:max_x, min_y:max_y]

    # cv2.imwrite(os.path.join(cropped_img_dir, img_path.split("\\")[-1] + "_test4.jpg"), source_edge_img)

    center_row_index = source_edge_img.shape[1] // 2
    center_col_index = source_edge_img.shape[0] // 2

    edge_diff = np.abs((center_row_index - center_col_index))

    if center_row_index < center_col_index:
        square_source_edge_img = source_edge_img[:, edge_diff:-edge_diff]
    elif center_col_index < center_row_index:
        square_source_edge_img = source_edge_img[edge_diff:-edge_diff, :]
    else:
        square_source_edge_img = source_edge_img

    square_gray_scale_img = cv2.cvtColor(square_source_edge_img, cv2.COLOR_BGR2GRAY)

    center_row = square_gray_scale_img[square_gray_scale_img.shape[0] // 2]
    center_col = square_gray_scale_img[square_gray_scale_img.shape[1] // 2]

    min_x = np.where(center_row > 215)[0][0]
    max_x = np.where(center_row > 215)[0][-1]
    min_y = np.where(center_col > 215)[0][0]
    max_y = np.where(center_col > 215)[0][-1]

    diff_min_x = int(min_x)
    diff_max_x = int(center_row.shape[0] - max_x)
    diff_min_y = int(min_y)
    diff_max_y = int(center_col.shape[0] - max_y)

    diff_max = max(diff_min_x, diff_max_x, diff_min_y, diff_max_y) + 50

    # print(diff_max, diff_min_x, diff_max_x, diff_min_y, diff_max_y)

    # test = square_gray_scale_img[:, square_gray_scale_img.shape[1] // 2 - 50:square_gray_scale_img.shape[1] // 2 + 50]
    # cv2.imwrite(os.path.join(cropped_img_dir, img_path.split("\\")[-1] + "_test0.jpg"), test)

    # cv2.imwrite(os.path.join(cropped_img_dir, img_path.split("\\")[-1] + "_test1.jpg"), square_source_edge_img)

    square_source_edge_img = square_source_edge_img[diff_max:-diff_max, diff_max:-diff_max]

    # cv2.imwrite(os.path.join(cropped_img_dir, img_path.split("\\")[-1] + "_test2.jpg"), square_source_edge_img)

    rad = (square_source_edge_img.shape[0]) // 2

    # create a mask
    mask = np.full((square_source_edge_img.shape[0], square_source_edge_img.shape[1]), 0, dtype=np.uint8)
    # create circle mask, center, radius, fill color, size of the border
    cv2.circle(mask, (rad, rad), rad, (255, 255, 255), -1)

    # get only the inside pixels
    fg = cv2.bitwise_or(square_source_edge_img, square_source_edge_img, mask=mask)

    mask = cv2.bitwise_not(mask)
    background = np.full(square_source_edge_img.shape, 255, dtype=np.uint8)
    bk = cv2.bitwise_or(background, background, mask=mask)
    final = cv2.bitwise_or(fg, bk)
    cv2.imwrite(os.path.join(cropped_img_dir, img_path.split("\\")[-1]), final)

def watershed(img_dir_list, args):

    ext = args.extension
    threshold = args.threshold

    black_white_img_dir = f"interim_images\\{args.data_name}\\Black_White_div_{args.div}\\"
    boxes_img_dir = f"interim_images\\{args.data_name}\\Boxes_div_{args.div}\\"
    contours_img_dir = f"interim_images\\{args.data_name}\\Contours_div_{args.div}\\"
    grayscale_img_dir = f"interim_images\\{args.data_name}\\Greyscale_div_{args.div}\\"
    mask_img_dir = f"interim_images\\{args.data_name}\\Mask_div_{args.div}\\"
    alpha_img_dir = f"interim_images\\{args.data_name}\\Alpha_div_{args.div}\\"
    density_dir = f"interim_images\\{args.data_name}\\Density_div_{args.div}\\"

    for i, img_path in enumerate(tqdm(img_dir_list)):

        img_name = img_path.split("\\")[-1]
        cropped_img_path = img_path

        source_image_name = "_".join(img_path.split("\\")[-1].split("_")[:2]) + "_" + str(i)

        # if mixed == True:
        #     source_image_name = img_path.split("\\")[-1].split(".")[0] + "_" + str(i)

        # Load the image
        source_image = cv2.imread(img_path)

        # Convert image to grayscale
        grayscale_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

        # Save the image
        cv2.imwrite(os.path.join(grayscale_img_dir, source_image_name + "_grayscale" + ext), grayscale_image)

        # convert image to black & white
        converted_grayscale_image = np.where(grayscale_image < threshold, 0, 255)

        density_image = np.where(grayscale_image < threshold, 255, 0)
        density = int(np.sum(density_image) / 255)

        print(img_name, density)

        cv2.imwrite(os.path.join(black_white_img_dir, source_image_name + "_converted_grayscale" + ext), converted_grayscale_image)        
        cv2.imwrite(os.path.join(density_dir, source_image_name + f"_density_{density}" + ext), density_image)

        converted_grayscale_image = cv2.imread(os.path.join(black_white_img_dir, source_image_name + "_converted_grayscale" + ext))

        # Remove gaussian noise
        denoised_image = cv2.GaussianBlur(converted_grayscale_image, (3, 3), 0)

        # Detect edges
        # 160 and 210 : min and max thresholds. Look at the saved image after tweakings, in order to find the right values.
        # edges = cv2.Canny(denoised_image, 160, 210)
        edges = cv2.Canny(denoised_image, 160, 210)

        # Save the image
        cv2.imwrite(os.path.join(contours_img_dir, source_image_name + "_edges" + ext), edges)

        # Use erode and dilate to remove unwanted edges and close gaps of some edges
        # Again, tweak the kernel values as needed

        # Erode will make the edges thinner. If the kernel size is big, some edges will be removed.
        # (1,1) will erode a little, (2,2) will erode more, (5,5) will erode even more...
        kernel = np.ones((1, 1), np.uint8)
        eroded_edges = cv2.erode(edges, kernel, iterations=10)

        # dilate will smooth the edges
        # (1,1) will dilate a little, (2,2) will dilate more, (5,5) will dilate even more...
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(eroded_edges, kernel, iterations=1)

        # Find contours
        # Use a copy of the image: findContours alters the image
        dilated_edges_copy = dilated_edges.copy()
        ret, thresh = cv2.threshold(dilated_edges_copy, 127, 200, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # We could have used v2.RETR_EXTERNAL and CHAIN_APPROX_NONE too

        # Create a list containing only the contours parents from the hierarchy returned by findContours
        hierarchy_parents_only = [x[3] for x in hierarchy[0]]

        # print("Number of contours found: ", len(contours))
        # print("Number of hierarchies found: ", len(hierarchy_parents_only))

        # Now we will filter the contours. We select only the ones we need.
        selected_contours = list()
        selected_hierarchy = list()
        min_area = 100

        for index, contour in enumerate(contours):
            # Keep only contours having no parent (remove overlapping contours
            if hierarchy_parents_only[index] == -1:
                # Keep only contours having an area greater than "min_area"
                area = cv2.contourArea(contour)
                if area > min_area:
                    selected_contours.append(contour)
                    selected_hierarchy.append(hierarchy[0][index])

        # print("Number of selected contours: ", len(selected_contours))
        # print("Number of selected hierarchies : ", len(selected_hierarchy))

        # Draw all contours on the source image (usefull for debugging, but change color (0, 0, 0) to something else if the background is black too).
        # -1 means drawing all contours, "(0, 255, 0)" for contours in green color, "3" is the thickness of the contours
        source_image_with_contours = cv2.drawContours(source_image, selected_contours, -1, (0, 0, 0), 3)

        # Save the image
        cv2.imwrite(os.path.join(contours_img_dir, source_image_name + "_with_contours" + ext), source_image_with_contours)

        img_boxes = source_image

        print(f"Processing: {img_path}. {len(selected_contours)} Found")

        # Now, extract each image
        for index, contour in enumerate(tqdm(selected_contours)):
            # Create mask where white is what we want, black otherwise
            mask = np.zeros_like(grayscale_image)

            # Draw filled contour in mask
            cv2.drawContours(mask, selected_contours, index, 255, -1)

            # Mask everything but the object we want to extract
            masked_image = cv2.bitwise_and(source_image, source_image, mask=mask)
            # print(source_image_name + "_out_{}".format(index) + ext)

            cv2.imwrite(os.path.join(mask_img_dir, source_image_name + "_out_{}.".format(index) + ext), masked_image)

            # Determine the bounding box (minimum rectangle in which the object can fit)
            (y, x) = np.where(mask == 255)
            (top_y, top_x) = (np.min(y), np.min(x))
            (bottom_y, bottom_x) = (np.max(y), np.max(x))

            img_boxes = cv2.rectangle(img_boxes, (top_x, top_y), (bottom_x, bottom_y), (0, 0, 255), 3)

            # Crop image (extract)
            og_source_image = cv2.imread(img_path)
            # import ipdb;ipdb.set_trace()
            extracted_image = og_source_image[top_y:bottom_y + 1, top_x:bottom_x + 1]
            alpha_image = cv2.cvtColor(extracted_image, cv2.COLOR_RGB2RGBA)

            for i, x in enumerate(extracted_image):
                for k, y in enumerate(x):
                    if extracted_image[i][k][0] > 200 and extracted_image[i][k][1] > 200 and extracted_image[i][k][2] > 200:
                        alpha_image[i][k][3] = 0

            extracted_white_check_image = cv2.cvtColor(extracted_image, cv2.COLOR_BGR2GRAY)

            white_check = np.where(extracted_white_check_image == 255, 1, 0)

            pixels = white_check.shape[0] * white_check.shape[1]
            white_sum = sum(sum(white_check))

            # Write to file
            # Image name for writing to file

            # import ipdb;ipdb.set_trace()

            if (white_sum / pixels) > 0.05 and np.mean(extracted_white_check_image) > 200:
                global edge_iter
                cropped_image_path = os.path.join(extracted_images_dir, "edge_edge_" + str(edge_iter) + ext)
                cv2.imwrite(cropped_image_path, extracted_image)
                edge_iter += 1
            elif np.abs(top_y - bottom_y + 1) < 10 or np.abs(top_x - bottom_x + 1) < 10:
                print("passing", os.path.join(extracted_images_dir, source_image_name + "_" + str(index) + ext))
                pass
            else:
                cropped_image_path = os.path.join(extracted_images_dir, source_image_name + "_" + str(index) + ext)
                cv2.imwrite(cropped_image_path, extracted_image)
                alpha_image_path = os.path.join(alpha_img_dir, source_image_name + "_" + str(index) + ext)
                cv2.imwrite(alpha_image_path, alpha_image)

            # print(white_sum/pixels, np.mean(extracted_white_check_image), img_path)

        cv2.imwrite(os.path.join(boxes_img_dir, source_image_name + "_with_boxes" + ext), img_boxes)

        # except:
        #     print("ERROR", os.path.join(contours_img_dir, source_image_name + ext))
        #     try:
        #         os.remove(os.path.join(contours_img_dir, source_image_name + ext))
        #     except:
        #         print("Error removing ", os.path.join(contours_img_dir, source_image_name + ext))



edge_iter = 0
div = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="alus")
    parser.add_argument("--threshold", type=int, default=200)
    parser.add_argument("--div", type=int, default=1)
    parser.add_argument("--extension", type=str, default=".jpg")
    parser.add_argument("--density", type=bool, default=True)
    parser.add_argument("--multiprocess", type=bool, default=False)
    args = parser.parse_args()   

    img_dir = f"images\\{args.data_name}"
    test_images = f"test_images\\{args.data_name}"
    extracted_images_dir = f"extracted\\{args.data_name}\\"

    black_white_img_dir = f"interim_images\\{args.data_name}\\Black_White_div_{args.div}\\"
    boxes_img_dir = f"interim_images\\{args.data_name}\\Boxes_div_{args.div}\\"
    contours_img_dir = f"interim_images\\{args.data_name}\\Contours_div_{args.div}\\"
    grayscale_img_dir = f"interim_images\\{args.data_name}\\Greyscale_div_{args.div}\\"
    mask_img_dir = f"interim_images\\{args.data_name}\\Mask_div_{args.div}\\"
    alpha_img_dir = f"interim_images\\{args.data_name}\\Alpha_div_{args.div}\\"
    density_dir = f"interim_images\\{args.data_name}\\Density_div_{args.div}\\"

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    img_dir_list = glob(f"{img_dir}\\*")

    if len(img_dir_list) == 0:
        print(f"Image Dir {data_name} empty. Fill with images")
        quit()

    if not os.path.exists(black_white_img_dir):
        os.makedirs(black_white_img_dir)
    
    if not os.path.exists(boxes_img_dir):
        os.makedirs(boxes_img_dir)

    if not os.path.exists(contours_img_dir):
        os.makedirs(contours_img_dir)

    if not os.path.exists(extracted_images_dir):
        os.makedirs(extracted_images_dir)

    if not os.path.exists(grayscale_img_dir):
        os.makedirs(grayscale_img_dir)

    if not os.path.exists(mask_img_dir):
        os.makedirs(mask_img_dir)

    if not os.path.exists(alpha_img_dir):
        os.makedirs(alpha_img_dir)

    if not os.path.exists(density_dir):
        os.makedirs(density_dir)

    if (args.multiprocess == True):
        mp_list_size = int(len(img_dir_list) / (mp.cpu_count() - 2))
        mp_img_list = [img_dir_list[i:i + mp_list_size] for i in range(0, len(img_dir_list), mp_list_size)]
        # print("Image Directory: ", args.image_dir, "\nNum Images: ", len(img_dir_list))

        for i, mp_list_chunk in enumerate(mp_img_list):
            print("Initializing Worker ", i, len(mp_list_chunk))
            p = mp.Process(target=watershed, args=(mp_list_chunk, args))
            p.start()
    else:
        watershed(img_dir_list, args)
