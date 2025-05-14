# libraries
import pandas as pd
import os
import sys

# manually adding the module path of lightglue

# sys.path.append('dataset/resources/lightglue-feature-extraction/LightGlue')
# import lightglue
from dataset.resources.lightglue_feature_extraction.LightGlue.lightglue import LightGlue, SIFT
from dataset.resources.lightglue_feature_extraction.LightGlue.lightglue.utils import load_image

import torch
import numpy as np
from scipy.stats import wasserstein_distance # Will need this if we use wasserstein
import re
import time
import pickle
from IPython.utils import io
from datetime import datetime
import subprocess
import os
# making the directors to run locally
MAIN_DIRECTORY = os.path.abspath("./dataset")
os.makedirs(MAIN_DIRECTORY, exist_ok=True)
os.chdir(MAIN_DIRECTORY)

PREDICTION_FOLDER = os.path.join(MAIN_DIRECTORY, "predict")
ORIGINAL_IMAGES_FOLDER = os.path.join(MAIN_DIRECTORY)
IMAGE_NOT_FOUND_PLACEHOLDER = os.path.join(MAIN_DIRECTORY, "resources/turtle-not-found.png")

TIMESTAMP_STRING = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

SEGMENTED_IMAGES_FOLDER = os.path.join(MAIN_DIRECTORY, "results", "segmented_images", TIMESTAMP_STRING)
RESULTS_IMAGES_FOLDER = os.path.join(MAIN_DIRECTORY, "results", "final_re-id_results")

LIGHTGLUE_RESOURCES_FOLDER = os.path.join(MAIN_DIRECTORY, "resources", "lightglue_feature_extraction", "LightGlue")
DETECTION_RESOURCES_FOLDER = os.path.join(MAIN_DIRECTORY, "resources", " detection-segmentation-rotation")

# if the directors don't exist to avoid errors
os.makedirs(PREDICTION_FOLDER, exist_ok=True)
os.makedirs(SEGMENTED_IMAGES_FOLDER, exist_ok=True)
os.makedirs(RESULTS_IMAGES_FOLDER, exist_ok=True)
os.makedirs(LIGHTGLUE_RESOURCES_FOLDER, exist_ok=True)
os.makedirs(DETECTION_RESOURCES_FOLDER, exist_ok=True)


def get_processed_images_names_and_directions():
    df = pd.read_csv(SEGMENTED_IMAGES_FOLDER.replace("extraction", "") + "file_name_equivalence.csv")
    processed_image_names = df.segmented.tolist()
    predicted_directions = ["R" if "[R]" in n else "L" for n in processed_image_names]
    return df.original.tolist(), processed_image_names, predicted_directions

# Load the scripts for the Object detection, Segmentation, and Rotation and
def run_main_script(original_images_path, segmented_images_path):
    # Define the path to the main.py script
    main_script_path = os.path.join("dataset/resources/detection-segmentation-rotation", 'main.py')
    print("abc", main_script_path)
    # Define the command-line arguments
    year = ""  # You can modify this as needed
    root_dir = original_images_path
    save_dir = segmented_images_path
    extraction_flag = "--extraction"  # This flag will be added if needed

    # Construct the command as a list of arguments
    command = [
        "python3", "detection-segmentation-rotation/main.py",
        "--year", year,
        "--root-dir", root_dir,
        "--save-dir", save_dir,
        extraction_flag
    ]
    
    try:
      result = subprocess.run(command, check=True, capture_output=True, text=True)
      print("Command succeeded!")
      print("Output:", result.stdout)

    except subprocess.CalledProcessError as e:
      print("Command failed!")
      print("Return code:", e.returncode)
      print("Error message:", e.stderr)
      print("Command that failed:", e.cmd)

# Example usage

def pre_process_images(original_images_path=PREDICTION_FOLDER, segmented_images_path=SEGMENTED_IMAGES_FOLDER):
    os.makedirs(segmented_images_path, exist_ok=True)
    with io.capture_output() as captured:
#         %cd $DETECTION_RESOURCES_FOLDER

        # Use the %run magic command to execute the script
        run_main_script(original_images_path, segmented_images_path)
#         %run main.py --year "" --root-dir $original_images_path --save-dir $segmented_images_path --extraction

        global SEGMENTED_IMAGES_FOLDER
        SEGMENTED_IMAGES_FOLDER = segmented_images_path + "/extraction"

#         %cd $MAIN_DIRECTORY

    return get_processed_images_names_and_directions()



from pathlib import Path

LIGHTGLUE_LIBRARY_FOLDER = LIGHTGLUE_RESOURCES_FOLDER

# %cd $LIGHTGLUE_LIBRARY_FOLDER
os.chdir(LIGHTGLUE_LIBRARY_FOLDER)

# if Path.cwd().name != "LightGlue":
#     !git clone --quiet https://github.com/cvg/LightGlue/
#     %cd LightGlue
#     !pip install --progress-bar off --quiet -e .



torch.set_grad_enabled(False)

# Use GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'


def label_from_path(path):
  '''
    Extract year, number, and side (if available) from the filename

    Args:
      path (str) : path to extract label from

    Returns:
      _ (str) : ID of format [YY]-[Turtle Nr] [L/R]
  '''

  match = re.search(r'(\d{2})-(\d+)', path)

  if match:
      year, number = match.groups()

      # Check for capital 'L' or 'R' in the filename
      side_match = re.search(r'[LR]', path)

      if side_match:
          side = side_match.group()
      else:
          # If capital 'L' or 'R' not found, check for lowercase 'l' or 'r'
          side_match = re.search(r'[lr]', path)
          side = side_match.group().upper() if side_match else ''

      # Create a new filename based on the specified format
      return f"{year}-{number} {side}"

def feature_dict_to_device(f_dict, device):
  '''
    Sends a dicitonary with tensors as values to a certain device

    Args:
      f_dict: Dictionary with tensors as values
      device: Device to send the dictionary to

    Returns:
      f_dict (dict): Same dictionary on different device
  '''
  for key in f_dict:
    temp_dict = {}

    for f_key in f_dict[key]:
      temp_dict[f_key] = f_dict[key][f_key].to(device)
    f_dict[key] = temp_dict

  return f_dict

def divide_list(original_list, N):
  '''
  Divides a given list into sublists of size N approximately.

  Args:
    original_list (list): The list to be divided.
    N (int): Size of sublists.

  Returns:
    list: List of sublists where each sublist, except the last one, has N elements.
          The last sublist may have fewer elements if the length of original_list is not divisible by N.
  '''

  num_sublists = len(original_list) // N
  remainder = len(original_list) % N

  sublists = [original_list[i * N:(i + 1) * N] for i in range(num_sublists)]

  if remainder > 0:
      sublists.append(original_list[-remainder:])

  return sublists

from itertools import compress

def batch_by_dict(feature_dict, labels, max_batch_size=32):
  '''
  Batches a given feature dictionary according to Keypoint length and a given batch size
  Within a single batch, every feature has the same number of keypoints. The batches are of
  the given size, unless not possible

  Args:
    feature_dict : dictionary of features: Key - path from data root, Value - keypoints + descriptors

  Returns:
    batch_list (list) : list of dicitonaries, where every dictionary is a single batch
    label_list (list) : re-ordered list of lables according to the batching process
  '''
  kpt_len = np.array([len(feature['keypoints'][0]) for feature in feature_dict.values()])

  feature_list = feature_dict.values()
  label_list = []
  batch_list = []
  for n_kpt in set(kpt_len):
    sub_feat = list(compress(feature_list, kpt_len == n_kpt))
    sub_labels = list(compress(labels, kpt_len == n_kpt))

    # if size of subset > batch size, split into smaller batches
    bs_sub_feat = divide_list(sub_feat, max_batch_size)
    bs_sub_labels = divide_list(sub_labels, max_batch_size)

    for bs_feat, bs_labels in zip(bs_sub_feat, bs_sub_labels):
      new_dict = {}
      for key in bs_feat[0].keys():
        new_dict[key] = []

      for feat in bs_feat:
        for key in feat.keys():
          new_dict[key].append(feat[key])

      for key in new_dict:
        new_dict[key] = torch.concat(new_dict[key])

      batch_list.append(new_dict)
      label_list += bs_labels

  return batch_list, label_list

import numpy as np
fpath_base = LIGHTGLUE_RESOURCES_FOLDER + "/LightGlue_Implementations/WassersteinDistributions/"
def import_distributions(fpath_base,method='superpoint'):
  # Import pre-saved score distributions for non-matches (x0) and matches (x1)
  if method == 'disk':
    f0 = 'x0_disk.npy'
    f1 = 'x1_disk.npy'
  elif method == 'superpoint':
    f0 = 'x0_superpoint.npy'
    f1 = 'x1_superpoint.npy'
  elif method == 'aliked':
    f0 = 'x0_aliked.npy'
    f1 = 'x1_aliked.npy'
  elif method == 'sift':
    f0 = '20240106_Update/x0_sift.npy'
    f1 = 'x1_sift.npy'

  fp0 = fpath_base + f0
  fp1 = fpath_base + f1

  x0 = np.load(fp0)
  x1 = np.load(fp1)
  return x0, x1


def matching_score(match_out, method="score10", x0=[], batch_index=0):
  '''
  Given a LightGlue output for a matching pair, return the score for the current pair

  Args:
    match-out (dict): output of LightGlue
    method (str): the method of extraction
      'score10/25/50': Sum of the highest 10/25/50 confidence scores for matched keypoints
      'wasserstein_h0': The scipy function wasserstein_distance
      'wasserstein_h1': The scipy function wasserstein_distance, but ammended to be 1-wasserstein
    x0 (list): the distribution that we are comparing to
  '''

  match_scores = match_out["scores"][batch_index]
  im0_scores = match_out["matching_scores0"][batch_index]

  # top 10 matched keypoint scores summed
  if method == "score10":
    return match_scores.sort(descending=True)[0][:min(len(match_scores), 10)].sum()

  # top 25 matched keypoint scores summed
  elif method == "score25":
    return match_scores.sort(descending=True)[0][:min(len(match_scores), 25)].sum()

  # top 50 matched keypoint scores summed
  elif method == "score50":
    return match_scores.sort(descending=True)[0][:min(len(match_scores), 50)].sum()

  # ------------
  # add Wasserstein_h0 here
  elif method == "wassersteinH0":
    return wasserstein_distance(x0, im0_scores.to("cpu"))

  # add Wasserstein_h1 here - this compares to a matched distribution (i.e. small wasserstein = match)
  elif method == "wassersteinH1":
    return 1 - wasserstein_distance(x0, im0_scores.to("cpu"))
  # ------------

  else:
    print("Unviable method argument. Defaulting to total number of matches.")
    return match_scores.sort(descending=True)[0][:min(len(match_scores), 10)].sum()


def predict_topN(score, labels, topN=5, metric='Highest'):
  '''
  Returns the top N predicted (unique) turtle IDs based on an array of scores

  Args:
    score - array of scores for current image against whole reference set
    labels - list of turtle IDs corresponding to each score
    topN - top N predictions to return
    metric - what metric you are using (most use the largest score, apart from wassersteinH1

  Returns:
    _ (list): top N unique turtle IDs
    _ (list): scores associated with each turtle ID in the top N
  '''

  score, labels = np.array(score), np.array(labels)
  indices = np.argsort(score) #  default asc
  if metric == 'wassersteinH1':
    score_dict = {label: s for label, s in zip(labels[indices], score[indices])} # only minimum scores for each label
    reverse = False # ascending order

  else:
    score_dict = {label: s for label, s in zip(labels[indices], score[indices])} # only maximum scores for each label
    reverse = True # descending order

  sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=reverse) # sort by score
  topN_items = sorted_items[:topN]

  return [item[0] for item in topN_items], [item[1] for item in topN_items]


def predict_NewTurtle(dist, val, thresh):
  '''
  Based on the maximum h0 score this predicts the probability that the turtle is not a match

  Inputs
  dist - The model distribution (LIGHTGLUE_RESOURCES_FOLDER + "/LightGlue_Implementations/NewTurtle_H1scores.sav")
  val - The maximum H0 score from all the comparisons
  thresh - A threshold where if the p value is greater then true

  Outputs
  p - The probability that the turtle is a new turtle (not part of the dataset). 1 - the integral of the distribution.
  pred_New - Boolean on whether the turtle is new or not
  '''
  p = 1 - dist.integrate_box_1d(0, val)
  pred_New = p > thresh
  return p, pred_New

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# #Configs
sift_extractor = SIFT(max_num_keypoints=1024).eval().to(device)
lightglue_config = {"extractor" :  "sift",
                    "n_kpts" : 1024,
                    "n_layers" : 9,
                    "metric" : "wassersteinH0",
                    "novelty_thresh" : 0.74}
feature_extractor = lightglue_config["extractor"]
n_keypoints = lightglue_config["n_kpts"]
lg_n_layers = lightglue_config["n_layers"]

light_glue = LightGlue(features=feature_extractor,
                       n_layers=lg_n_layers,
                       width_confidence=-1, # disable for batching
                       depth_confidence=-1).eval().to(device)
fpath_base_W = LIGHTGLUE_RESOURCES_FOLDER + "/LightGlue_Implementations/WassersteinDistributions/"
novel_path = LIGHTGLUE_RESOURCES_FOLDER + "/LightGlue_Implementations/"
save_path = LIGHTGLUE_RESOURCES_FOLDER + "/LightGlue_Keypoints/"

def predict(new_impath, face_side, matcher=light_glue, extractor=sift_extractor, config=lightglue_config, topN=5):
  '''
  Perform inference on a single image for face recognition or similarity matching.

  Args:
  - new_impath (str): File path to the new image.
  - face_side (str): Side of the face ('L' for left, 'R' for right).
  - matcher: Matcher object for matching.
  - extractor: Feature extractor object for extracting features.
  - config (dict): Configuration dictionary containing various settings.
  - topN (int, optional): Number of top matches to consider. Defaults to 5.

  Returns:
  - list: Predicted labels for the new image.
  - list: Predicted scores for matches.
  - bool: Indicates whether the image is considered new.
  - float: Probability of the image being new.
  - float: Time taken for image loading and feature extraction.
  - float: Time taken for matching.
  '''

  train_dict_L = torch.load(f"{save_path}totalL_{config['extractor']}_{config['n_kpts']}_highRes.pth", map_location=torch.device(device))
  train_dict_R = torch.load(f"{save_path}totalR_{config['extractor']}_{config['n_kpts']}_highRes.pth", map_location=torch.device(device))
  # print(train_dict_L)
  # print("\n\n\n\n\n\n\n")
  # to device
  train_dict_L = feature_dict_to_device(train_dict_L, device)
  train_dict_R = feature_dict_to_device(train_dict_R, device)

  # reference set labels
  turtles_in_trainL = [label_from_path(im_path) for im_path in train_dict_L]
  turtles_in_trainR = [label_from_path(im_path) for im_path in train_dict_R]

  # define batches and sort labels accordingly
  batch_list_L, turtles_in_trainL = batch_by_dict(train_dict_L, turtles_in_trainL, max_batch_size=64)
  batch_list_R, turtles_in_trainR = batch_by_dict(train_dict_R, turtles_in_trainR, max_batch_size=64)

  # import reference distributions
  x0, x1 = import_distributions(fpath_base_W, method=config['extractor'])

  # import #1-score PDF for novelty detection
  novelty_dist = pickle.load(open(novel_path + 'NoveltyPDF.sav', 'rb'))

  # load image and extract features
  t_s = time.time()
  image = load_image(new_impath).to(device)
  nb_feat0 = extractor.extract(image)
  t_e = time.time()
  time_load_extract = t_e - t_s

  # define scoring function
  metric = config["metric"]
  if metric == 'wassersteinH0':
    scoring_function = lambda out, i: matching_score(out, method=metric, x0=x0, batch_index=i)
  elif metric == 'wassersteinH1':
    scoring_function = lambda out, i: matching_score(out, method=metric, x0=x1, batch_index=i)
  else:
    scoring_function = lambda out, i: matching_score(out, method=metric, batch_index=i)

  # only search left/right reference set
  batch_list, turtles_in_train =  (batch_list_L, turtles_in_trainL) if face_side == "L" else (batch_list_R, turtles_in_trainR)

  scores = []
  nr_matches = []
  # match
  t_s = time.time()
  for batch in batch_list:
    batch_size = batch['keypoints'].shape[0]

    feat1 = batch # get feature dictionary

    # reformat feat0 to fit batch
    feat0 = {}
    for key in nb_feat0:
      feat0[key] = torch.concat([nb_feat0[key]] * batch_size)

    matches = light_glue({"image0": feat0, "image1": feat1})

    for i in range(batch_size):
      # obtain scores
      score = scoring_function(matches, i)
      scores.append(score.item())

      # nr of matched keypoints
      nr_matches.append(len(matches["matches"][i]))

  # predict
  # top 5 matches
  predicted_label, predicted_scores = predict_topN(scores, turtles_in_train, metric=metric)

  # new turtle condition
  novelty_thresh = config["novelty_thresh"]
  p_new, is_new = predict_NewTurtle(novelty_dist, predicted_scores[0], novelty_thresh)

  t_e = time.time()
  match_time = t_e - t_s

  return predicted_label, predicted_scores, is_new, p_new, time_load_extract, match_time

"""## Putting it all together"""

import os
import glob
import matplotlib.pyplot as plt
from PIL import Image, ExifTags

def correct_image_orientation(img):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = dict(img._getexif().items())

        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # No Exif orientation data, no need to correct orientation
        pass

    return img

def find_image_path(image_name):
    year_path = os.path.join(ORIGINAL_IMAGES_FOLDER, f"20{image_name[0:2]}")
    search_pattern = os.path.join(year_path, f"{image_name}*")
    found_images = glob.glob(search_pattern)
    if found_images:
      return found_images[0]  # Return the first match
    return None

def display_prediction(input_image_path, labels, scores, is_new, p_new):
    plt.figure(figsize=(20, 10))

    # Display input image with corrected orientation
    plt.subplot(1, len(labels) + 1, 1)
    with Image.open(input_image_path) as input_image:
        input_image = correct_image_orientation(input_image)
        plt.imshow(input_image)
        plt.title(f"Input Image: {os.path.basename(input_image_path)}\nIs new: {is_new} ({p_new:.2f})")
        plt.axis('off')

    # Display top N matches
    for i, (title, confidence) in enumerate(zip(labels, scores)):
        plt.subplot(1, len(labels) + 1, i + 2)
        image_path = find_image_path(title)
        plt.title(f"{title}\nConf: {confidence:.2f}")
        plt.axis('off')
        if image_path:
            with Image.open(image_path) as image:
                image = correct_image_orientation(image)
                plt.imshow(image)
        else:
            with Image.open(IMAGE_NOT_FOUND_PLACEHOLDER) as image:
                plt.imshow(image)

    plt.show()

def collect_results(results, original_img_name, direction, labels, scores, is_new, p_new):
    results["img_name"].append(original_img_name)
    results["face_side"].append(direction)
    results["match_1"].append(labels[0])
    results["score_1"].append(scores[0])
    results["match_2"].append(labels[1])
    results["score_2"].append(scores[1])
    results["match_3"].append(labels[2])
    results["score_3"].append(scores[2])
    results["match_4"].append(labels[3])
    results["score_4"].append(scores[3])
    results["match_5"].append(labels[4])
    results["score_5"].append(scores[4])
    results["predicted_new"].append(is_new)
    results["score_new"].append(p_new)

def predict_folder():
    start = time.time()
    print("Ready to predict images in the 'predict' folder.")
    print("Pre-processing images...")
    original_image_names, processed_image_names, predicted_directions = pre_process_images()
    print(f"{len(processed_image_names)} images preprocessed in {(time.time()-start):.0f}s")
    print("\n")
    results = {"img_name": [], "face_side": [],
               "match_1": [], "score_1": [], "match_2": [], "score_2": [],
               "match_3": [], "score_3": [], "match_4": [], "score_4": [],
               "match_5": [], "score_5": [], "predicted_new": [], "score_new": []}
    for original_img_name, processed_img_name, direction in zip(original_image_names, processed_image_names, predicted_directions):
        s = time.time()
        print(f"Running prediction for image '{original_img_name}'...")
        # Run pre-processing (object detection, segmentation, rotation, face side prediction)
        processed_img_path = os.path.join(SEGMENTED_IMAGES_FOLDER, processed_img_name)
        # Run feature extraction and find matches
        labels, scores, is_new, p_new, time_load_extract, match_time = predict(processed_img_path, direction)
        # Display results
        original_img_path = os.path.join(PREDICTION_FOLDER, original_img_name)
        print(labels)
        print(scores)
        print(is_new)
        display_prediction(original_img_path, labels, scores, is_new, p_new)
        print(f"Total prediction time for image '{original_img_name}': {(time.time()-s):.0f}s")
        print("\n")
        collect_results(results, original_img_name, direction, labels, scores, is_new, p_new)

    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(RESULTS_IMAGES_FOLDER + "/" + TIMESTAMP_STRING + ".csv")


def insert_new_keypoints(yr,t_id,side,keypoints,dict_db,fname,config):
  '''
  The plan for this function is to add the keypoints of the new image into the correct database.
  Then save the updated dictionary
  '''

  #Update the dictionary
  d_name = f"{yr}/{t_id} {side}"
  dict_db[d_name] = keypoints
  #Save the database
  torch.save(dict_db, fname)

def load_keypoints(db_path,side,config):
  #Load the database
  fname = f"{db_path}total{side}_{config['extractor']}_{config['n_kpts']}_highRes.pth"
  dict_db = torch.load(fname)
  print(dict_db)
  print("\n\n\n\n\n\n\n")
  return dict_db, fname

def get_new_keypoints(ImageName, side, SEGMENTED_IMAGES_FOLDER, device):
  #Load the Segmented Image
  image_path = f"{SEGMENTED_IMAGES_FOLDER}/{ImageName} [{side}].jpg"
  #Establish the extractor
  extractor = SIFT(max_num_keypoints=1024).eval().to(device)

  image = load_image(image_path).to(device)
  feat = extractor.extract(image)

  return feat

def get_new_and_update_keypoint_db(yr,t_id,side,ImageName, SEGMENTED_IMAGES_FOLDER, device, db_path, config):
  dict_db, fname = load_keypoints(db_path,side,config)
  #Update the dictionary
  d_name = f"{yr}/{t_id} {side}"

  #Check if key exists
  key_exists = 0
  if d_name in dict_db.keys():
    key_exists = 1

  if key_exists == 1:
    user_input = 0
    while user_input == 0:
      overwrite = input('Warning, key already exists in the dictionary. Do you want to overwrite it, yes(y) or no(n)?')
      print(overwrite)
      if overwrite == 'y':
        #Get keypoints
        keypoints = get_new_keypoints(ImageName, side, SEGMENTED_IMAGES_FOLDER, device)
        #Inset the keypoints
        insert_new_keypoints(yr, t_id, side, keypoints, db_path, fname, config)

        dict_db[d_name] = keypoints
        #Save the database
        torch.save(dict_db, fname)
        user_input = 1
      elif overwrite == 'n':
        print("No selected, not updaing the database")
        user_input = 1
      else:
        print("Input not recognised please type 'y' for yes, and 'n' for no")
  else:
    #Continue as normal
    keypoints = get_new_keypoints(ImageName, side, SEGMENTED_IMAGES_FOLDER, device)
    insert_new_keypoints(yr,t_id,side,keypoints,db_path,config)



# final application, just predict from prediction folder
predict_folder()



Input_Image_Name = '24-001R' # @param {type:"string"}
Year_Input_Photo_Taken = '2024' # @param {type:"string"}
Turtle_ID = '24-001' # @param {type:"string"}
Turtle_Side = 'Right' # @param ["Left", "Right"]

#Format the forms to make sure
predict_image = Input_Image_Name
# Turtle dictionary
yr = Year_Input_Photo_Taken
t_id = Turtle_ID
if Turtle_Side == "Left":
  side = 'L'
elif Turtle_Side == "Right":
  side = 'R'

d_name = f"{yr}/{t_id} {side}"
# print(f"Starting adding image {predict_image} as {d_name} to the database")
# get_new_and_update_keypoint_db(yr,t_id,side,predict_image, SEGMENTED_IMAGES_FOLDER, device, save_path, lightglue_config)