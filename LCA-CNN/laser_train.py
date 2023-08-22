import operator
import os
import pickle

use_gpu_num = '0'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu_num

from keras import applications
from keras import optimizers
import tensorflow as tf
import my_callbacks

from keras.utils import multi_gpu_model

import math

import model as GP
from multi_label_keras_image import MultiLabelImageDataGenerator

###################################
import keras.backend as KB

# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
# config.gpu_options.allow_growth = True

# Create a session with the above options specified.
KB.tensorflow_backend.set_session(tf.Session(config=config))

is_gpu = False
gpu_count = len(KB.tensorflow_backend._get_available_gpus())
if gpu_count != 0:
    is_gpu = True
if gpu_count > 1:
    tf.device('/cpu:0')

if not os.path.isdir('weights' + str(use_gpu_num)):
    os.mkdir('weights' + str(use_gpu_num))

###################################

save_path = 'fullmodel_inception_v3.h5'
save_weight_path = 'fullmodel_inception_v3_weight.h5'

# dimensions of our images.
img_width, img_height = 448, 448

train_data_dir = 'your_roteted_training_data' #'/data/laser_detection/work_dirs/RED_cascade_rcnn_r50_fpn_20e_coco/data_split/train_new_rotate/'
# train_data_dir = 'train'
test_data_dir = 'your_testing_data' #'/data/laser_detection/work_dirs/RED_cascade_rcnn_r50_fpn_20e_coco/data_split/val_new/'

epochs = 100
epochs_pre = 10
batch_size = 14
seed = 2

class_num = len([x for x in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_dir, x))])
print('class_num:' + str(class_num))


def train_gatp_two_stream():
    # batch_size = 14
    # build the network
    modelvgg = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    modelvgg2 = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

    # modelvgg.load_weights('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
    modelvgg2.load_weights('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
    print('Model loaded.')

    model = GP.build_global_attention_pooling_model_cascade_attention([modelvgg, modelvgg2], class_num)

    single_model = model
    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)

    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy',
                        'categorical_crossentropy',
                        'mean_squared_error', 'mean_squared_error',
                        GP.rank_loss,
                        GP.cross_network_similarity_loss,
                        GP.rank_loss,
                        GP.rank_loss,
                        ],
                  # optimizer=optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=False),
                  optimizer=optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.000001),
                  metrics={'output_1': ['accuracy', 'top_k_categorical_accuracy'],
                           'output_2': ['accuracy', 'top_k_categorical_accuracy'],
                           'output_3': ['accuracy', 'top_k_categorical_accuracy'],
                           # 'output_4': ['accuracy', 'top_k_categorical_accuracy'],
                           'output_5': ['accuracy', 'top_k_categorical_accuracy'],
                           })

    # prepare data augmentation configuration
    train_datagen = MultiLabelImageDataGenerator(
        input_num=2,
        label_num=4,
        constraint_num=2,
        extra_constraints=[2, 2, 2, 2]
    )

    test_datagen = MultiLabelImageDataGenerator(
        input_num=2,
        label_num=4,
        constraint_num=2,
        extra_constraints=[2, 2, 2, 2]
    )

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        interpolation='bilinear',
        shuffle=True)
    validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False,
        interpolation='bilinear',
        class_mode='categorical')
    file_path = "/data/laser_detection/work_dirs/LCA_CNN/all_crop_box_by_name_112_" + use_gpu_num + \
                "/weights-gatp-two-stream-inception_v3-{epoch:03d}-{val_output_5_accuracy:.4f}.hdf5"
    checkpoint = my_callbacks.ModelCheckpoint(file_path, single_model, monitor='val_output_5_accuracy',
                                              verbose=1, save_best_only=False, save_weights_only=True, mode='max')
                                              #verbose=1, save_best_only=True, save_weights_only=True, mode='max')
    callbacks_list = [checkpoint]

    # fine-tune the model
    model.fit_generator(
        train_generator,
        steps_per_epoch=(train_generator.n // batch_size),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=math.ceil(float(validation_generator.n) // batch_size),
        callbacks=callbacks_list,
    )

    result = model.evaluate_generator(
        validation_generator
    )
    print(result)

    # model.save(save_path)
    # model.save_weights(save_weight_path)


def val():
    # batch_size = 14
    # build the network
    modelvgg = applications.InceptionV3(weights=None, include_top=False, input_shape=(img_width, img_height, 3))
    modelvgg2 = applications.InceptionV3(weights=None, include_top=False, input_shape=(img_width, img_height, 3))

    print('Model loaded.')

    model = GP.build_global_attention_pooling_model_cascade_attention([modelvgg, modelvgg2], class_num)

    model.load_weights(
        '/data/laser_detection/work_dirs/LCA_CNN/weights_new0/weights-gatp-two-stream-inception_v3-031-0.9103.hdf5')

    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)

    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy',
                        'categorical_crossentropy',
                        'mean_squared_error', 'mean_squared_error',
                        GP.rank_loss,
                        GP.cross_network_similarity_loss,
                        GP.rank_loss,
                        GP.rank_loss,
                        ],
                  # optimizer=optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=False),
                  optimizer=optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.000001),
                  metrics={'output_1': ['accuracy', 'top_k_categorical_accuracy'],
                           'output_2': ['accuracy', 'top_k_categorical_accuracy'],
                           'output_3': ['accuracy', 'top_k_categorical_accuracy'],
                           #'output_4': ['accuracy', 'top_k_categorical_accuracy'],
                           'output_5': ['accuracy', 'top_k_categorical_accuracy'],
                           })

    test_datagen = MultiLabelImageDataGenerator(
        input_num=2,
        label_num=4,
        constraint_num=2,
        extra_constraints=[2, 2, 2, 2]
    )

    validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False,
        interpolation='bilinear',
        class_mode='categorical')

    result = model.evaluate_generator(
        validation_generator
    )
    print(len(result), result)

    # model.save(save_path)
    # model.save_weights(save_weight_path)

def extract(model_path):
    modelvgg = applications.InceptionV3(weights=None, include_top=False, input_shape=(img_width, img_height, 3))
    modelvgg2 = applications.InceptionV3(weights=None, include_top=False, input_shape=(img_width, img_height, 3))

    print('Model loaded.')

    model = GP.build_global_attention_pooling_model_cascade_attention([modelvgg, modelvgg2], class_num)

    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy',
                        'categorical_crossentropy',
                        'mean_squared_error', 'mean_squared_error',
                        GP.rank_loss,
                        GP.cross_network_similarity_loss,
                        GP.rank_loss,
                        GP.rank_loss,
                        ],
                  # optimizer=optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=False),
                  optimizer=optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.000001),
                  metrics={'output_1': ['accuracy', 'top_k_categorical_accuracy'],
                           'output_2': ['accuracy', 'top_k_categorical_accuracy'],
                           'output_3': ['accuracy', 'top_k_categorical_accuracy'],
                           # 'output_4': ['accuracy', 'top_k_categorical_accuracy'],
                           'output_5': ['accuracy', 'top_k_categorical_accuracy'],
                           })
    #model.load_weights('/data/laser_detection/work_dirs/LCA_CNN/weights_new0/weights-gatp-two-stream-inception_v3-031-0.9103.hdf5')
    model.load_weights(model_path)#'/data/laser_detection/work_dirs/LCA_CNN/all_crop_box_by_name202301280/')
    test_datagen = MultiLabelImageDataGenerator(
        input_num=2,
        label_num=4,
        constraint_num=2,
        extra_constraints=[2, 2, 2, 2]
    )

    validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=1,#batch_size,
        shuffle=False,
        interpolation='bilinear',
        class_mode='categorical')
    assert(len(validation_generator.filenames) == len(validation_generator))
    validation_generator.reset()
    fname2cls = {}
    for i, file_name in enumerate(validation_generator.filenames):
        v = validation_generator[i]
        prob = model.predict(v[0])[3]
        output = prob.argmax(-1)[0]
        #print(file_name, prob.shape, prob[:, output])
        fname2cls[os.path.basename(file_name)] = (output, prob[:, output][0])#np.argmax(output)
    print(fname2cls)
    #for k in fname2cls:
    #    print(k, fname2cls[k])
    with open('112_%s.pkl' % os.path.basename(model_path), 'wb') as f:
        pickle.dump(fname2cls, f)

    result = model.evaluate_generator(
        validation_generator
    )
    print(len(result), result)







if __name__ == '__main__':
    train_gatp_two_stream()
    #val()
    #model_foler = '/data/laser_detection/work_dirs/LCA_CNN/all_crop_box_by_name_112_0/'
    #models = os.listdir(model_foler)
    #for m in models:
    #    if m.endswith('.hdf5'):# and '095' in m:
    #        extract(model_foler + m)
