import os
import tensorflow as tf
from keras import optimizers
from keras import applications
from keras.utils import multi_gpu_model
import wandb
import model as GP

# Importing MultiLabelImageDataGenerator from multi_label_keras_image.py
from multi_label_keras_image import MultiLabelImageDataGenerator

# Initialize W&B
wandb.init(project="lca_cnn", entity="kiriti-v")

# Define dataset paths and model parameters
test_data_dir = '/data/laser_detection/work_dirs/RED_cascade_rcnn_r50_fpn_20e_coco/all_crop_box_by_name_112/'
img_width, img_height = 448, 448
batch_size = 14
class_num = 5

# Initializing the data generator for the test set without augmentation, as it's for evaluation
test_datagen = MultiLabelImageDataGenerator(
    rescale=1. / 255,  # Rescaling factor to match the training phase normalization
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode=None,  # Updated to None since we handle labels in the generator
    shuffle=False  # Data order should be consistent for evaluation, hence no shuffling
)

def dual_input_generator(single_input_generator):
    for input_batch in single_input_generator:
        # Duplicate the input for both model inputs
        yield [input_batch, input_batch]

# Wrap the original generator to duplicate its output for both inputs of the model
dual_generator = dual_input_generator(test_generator)

def compile_model():
    # Setup the base models with weights pre-trained on ImageNet
    base_model1 = applications.InceptionV3(include_top=False, weights='imagenet',
                                                    input_shape=(img_width, img_height, 3))
    base_model2 = applications.InceptionV3(include_top=False, weights='imagenet',
                                                    input_shape=(img_width, img_height, 3))

    # Assuming the model-building function is defined as in the provided context
    model = GP.build_global_attention_pooling_model_cascade_attention([base_model1, base_model2], class_num)

    # Compiling the model with the specified loss functions and metrics
    model.compile(optimizer=optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=None, decay=0.0),
                  loss=['categorical_crossentropy'] * 4 + ['mean_squared_error'] * 2 + [GP.rank_loss,
                                                                                        GP.cross_network_similarity_loss,
                                                                                        GP.rank_loss, GP.rank_loss],
                  metrics=['accuracy'])
    return model

# Path to the folder containing saved models
model_folder = '/data/laser_detection/work_dirs/LCA_CNN/all_crop_box_by_name_112_0/'

# Loop through each saved model in the folder and evaluate
for model_file in os.listdir(model_folder):
    if model_file.endswith('.hdf5'):
        print(f"Evaluating {model_file}...")
        model_path = os.path.join(model_folder, model_file)
        model = compile_model()
        model.load_weights(model_path)

        # Evaluate the model using the dual input generator
        evaluation_metrics = model.evaluate_generator(dual_generator, steps=len(test_generator))

        # Log evaluation metrics for each model to W&B
        wandb.log({"model_file": model_file, "evaluation_metrics": evaluation_metrics})

print("Evaluation complete. Metrics logged to W&B.")