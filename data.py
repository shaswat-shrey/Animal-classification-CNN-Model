import os
import random
from shutil import copyfile

try: 
    os.makedirs('tmp', exist_ok=True)
    os.makedirs('tmp/cats-v-dogs', exist_ok=True)
    os.makedirs('tmp/cats-v-dogs/training', exist_ok=True)
    os.makedirs('tmp/cats-v-dogs/validation', exist_ok=True)
    os.makedirs('tmp/cats-v-dogs/test', exist_ok=True)
    os.makedirs('tmp/cats-v-dogs/training/cats', exist_ok=True)
    os.makedirs('tmp/cats-v-dogs/training/dogs', exist_ok=True)
    os.makedirs('tmp/cats-v-dogs/validation/cats', exist_ok=True)
    os.makedirs('tmp/cats-v-dogs/validation/dogs', exist_ok=True)
    os.makedirs('tmp/cats-v-dogs/test/cats', exist_ok=True)
    os.makedirs('tmp/cats-v-dogs/test/dogs', exist_ok=True)
except:
    print("Error to make the directories")
    
CAT_DIR = "/Users/shaswatshrey/Downloads/kagglecatsanddogs_3367a/PetImages/Cat"
DOG_DIR = "/Users/shaswatshrey/Downloads/kagglecatsanddogs_3367a/PetImages/Dog"


TRAINING_DIR = "tmp/cats-v-dogs/training/"
VALIDATION_DIR = "tmp/cats-v-dogs/validation/"

TRAINING_CATS = os.path.join(TRAINING_DIR, "cats/")
VALIDATION_CATS = os.path.join(VALIDATION_DIR, "cats/")

TRAINING_DOGS = os.path.join(TRAINING_DIR, "dogs/")
VALIDATION_DOGS = os.path.join(VALIDATION_DIR, "dogs/")

INCLUDE_TEST = True

print(len(os.listdir('tmp/cats-v-dogs/training/cats')))
print(len(os.listdir('tmp/cats-v-dogs/training/dogs')))

print(len(os.listdir('tmp/cats-v-dogs/validation/cats')))
print(len(os.listdir('tmp/cats-v-dogs/validation/dogs')))

print(len(os.listdir('tmp/cats-v-dogs/test/cats')))
print(len(os.listdir('tmp/cats-v-dogs/test/dogs')))

def splitData(main_dir, training_dir, validation_dir, test_dir:None, include_test_split=True, split_ratio=0.9):
    """
    Splits the data into train validation and test sets (optional)

    Args:
    main_dir (string): path containing the images
    training_dir (string): path to be used for training
    validation_dir (string): path to be used for validation
    test_dir (string): path to be used for test
    include_test_split (boolean): whether to include a test split or not
    split_size (float): size of the dataset to be used for training

    """
    files = []
    for file in os.listdir(main_dir):
        if os.path.getsize(os.path.join(main_dir, file)):
            files.append(file)

    shuffled_files = random.sample(files, len(files))

    split = int(split_ratio * len(shuffled_files))
    train = shuffled_files[:split]
    split_valid_test = int(split + (len(shuffled_files) - split) / 2)

    if include_test_split:
        validation = shuffled_files[split:split_valid_test]
        test = shuffled_files[split_valid_test:]
    else:
        validation = shuffled_files[split:]

    for element in train:
        copyfile (
            os.path.join(main_dir, element),
            os.path.join(training_dir, element)
        )

    for element in validation:
        copyfile (
            os.path.join(main_dir, element),
            os.path.join(validation_dir, element)
        )

    if include_test_split:
        for element in test:
            copyfile (
                os.path.join(main_dir, element),
                os.path.join(test_dir, element)
            )

    print('Task is succesfull, Code line: 93')

splitData(CAT_DIR, 'tmp/cats-v-dogs/training/cats', 'tmp/cats-v-dogs/validation/cats', 'tmp/cats-v-dogs/test/cats', INCLUDE_TEST, 0.9) 
splitData(DOG_DIR, 'tmp/cats-v-dogs/training/dogs', 'tmp/cats-v-dogs/validation/dogs', 'tmp/cats-v-dogs/test/dogs', INCLUDE_TEST, 0.9) 