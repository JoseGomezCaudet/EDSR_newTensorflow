import tensorflow as tf
import os
import cv2
import numpy as np
import math
import data_utils
from skimage import io
import edsr
from PIL import Image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
import matplotlib.pyplot as plt


class run:
    def __init__(self, config, ckpt_path, scale, batch, epochs, B, F, lr, load_flag, meanBGR):
        self.config = config
        self.ckpt_path = ckpt_path
        self.scale = scale
        self.batch = batch
        self.epochs = epochs
        self.B = B
        self.F = F
        self.lr = lr
        self.load_flag = load_flag
        self.mean = meanBGR
        self.model = None

    def train(self, imagefolder, validfolder, edsrObj = None):

        # Create training dataset
        train_image_paths = data_utils.getpaths(imagefolder)
        train_dataset = tf.data.Dataset.from_generator(
            generator=data_utils.make_dataset,
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])),
            args=[train_image_paths, self.scale, self.mean]
        ).padded_batch(self.batch, padded_shapes=([None, None, 3], [None, None, 3]))

        # Create validation dataset
        val_image_paths = data_utils.getpaths(validfolder)
        val_dataset = tf.data.Dataset.from_generator(
            generator=data_utils.make_val_dataset,
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])),
            args=[val_image_paths, self.scale, self.mean]
        ).padded_batch(1, padded_shapes=([None, None, 3], [None, None, 3]))

        # Edsr model
        print("\nRunning EDSR.")
        edsrObj = edsr.Edsr(self.B, self.F, self.scale)
        optimizer = Adam(learning_rate=self.lr)

        train_loss_metric = Mean()
        val_psnr_metric = Mean()
        val_ssim_metric = Mean()

        checkpoint_dir = os.path.join(self.ckpt_path, "edsr_ckpt")
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=edsrObj)

        # Check for checkpoint directory and existence of checkpoint
        if os.path.exists(checkpoint_dir) and tf.train.latest_checkpoint(checkpoint_dir):
            if self.load_flag:
                checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
                print("\nLoaded checkpoint.")
            else:
                print("Checkpoint found, but not loading as per 'load_flag'.")
        else:
            print("No checkpoint found or 'load_flag' not set. Training from scratch.")


        print("Training...")
        for epoch in range(1, self.epochs + 1):
            print(f"Start of Epoch {epoch}")

            # Iterate over the batches of the dataset.
            for step, (LR, HR) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    out = edsrObj(LR, training=True)
                    loss = edsrObj.loss(HR, out)
                    grads = tape.gradient(loss, edsrObj.trainable_weights)

                optimizer.apply_gradients(zip(grads, edsrObj.trainable_weights))

                train_loss_metric(loss)

                if step % 1000 == 0:
                    print(f"Step {step}: loss = {train_loss_metric.result().numpy()}, lr = {optimizer.learning_rate.numpy()}")
                    train_loss_metric.reset_states()

        self.model = edsrObj

    def upscale(self, path):
        """
        Upscales an image via model. This loads a checkpoint, not a .pb file.
        """
        fullimg = cv2.imread(path, 3)
        floatimg = fullimg.astype(np.float32) - self.mean
        LR_input_ = floatimg.reshape(1, floatimg.shape[0], floatimg.shape[1], 3)

        print("\nUpscale image by a factor of {}:\n".format(self.scale))
        
         # load the model
        edsr_model = edsr.Edsr(self.B, self.F, self.scale) 
        checkpoint_dir = self.ckpt_path + "edsr_ckpt"
        checkpoint = tf.train.Checkpoint(model=edsr_model)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

        HR_output = edsr_model(LR_input_)
        Y = HR_output.numpy()[0]
        HR_image = (Y + self.mean).clip(min=0, max=255)
        HR_image = HR_image.astype(np.uint8)

        bicubic_image = cv2.resize(fullimg, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)

        """
        This is commented to be run in a Jupyter notebook
        cv2.imshow('Original image', fullimg)
        cv2.imshow('EDSR upscaled image', HR_image)
        cv2.imshow('Bicubic upscaled image', bicubic_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""
        
        
        # Show original image
        plt.imshow(cv2.cvtColor(fullimg, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.show()

        # Show upscaled image
        plt.imshow(cv2.cvtColor(HR_image, cv2.COLOR_BGR2RGB))
        plt.title('EDSR Upscaled Image')
        plt.show()

        # Show bicubic upscaled image
        plt.imshow(cv2.cvtColor(bicubic_image, cv2.COLOR_BGR2RGB))
        plt.title('Bicubic Upscaled Image')
        plt.show()

    def load_pb(self, path_to_saved_model):
        model = tf.saved_model.load(path_to_saved_model)
        return model

    """
    def upscaleFromPb(self, path):
        
        Upscale single image by desired model. This loads a .pb file.
        
        # Read model
        pbPath = "./models/EDSR_x{}.pb".format(self.scale)

        # Get graph
        graph = self.load_pb(pbPath)

        fullimg = cv2.imread(path, 3)
        floatimg = fullimg.astype(np.float32) - self.mean
        LR_input_ = floatimg.reshape(1, floatimg.shape[0], floatimg.shape[1], 3)

        LR_tensor = graph.get_tensor_by_name("IteratorGetNext:0")
        HR_tensor = graph.get_tensor_by_name("NHWC_output:0")

        with tf.Session(graph=graph) as sess:
            print("Loading pb...")
            output = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input_})
            Y = output[0]
            HR_image = (Y + self.mean).clip(min=0, max=255)
            HR_image = (HR_image).astype(np.uint8)

            bicubic_image = cv2.resize(fullimg, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)

            cv2.imshow('Original image', fullimg)
            cv2.imshow('EDSR upscaled image', HR_image)
            cv2.imshow('Bicubic upscaled image', bicubic_image)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        sess.close()"""

    def export(self):
        print("Exporting model...")

        export_dir = "./models/"
        if not os.path.exists(export_dir):
                os.makedirs(export_dir)

        export_file = "EDSR_x{}.pb".format(self.scale)
        export_path = export_dir + export_file
        tf.saved_model.save(self.model, export_path)

    def psnr(self, img1, img2):
        mse = tf.reduce_mean(tf.square(img1 - img2))
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * tf.math.log(PIXEL_MAX / tf.sqrt(mse)) / tf.math.log(10.0)