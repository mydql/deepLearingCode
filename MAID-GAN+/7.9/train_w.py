import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
import cv2
import datetime
import math



import  os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


### ljy改5：限制显存
# gpus = tf.config.experimental.list_physical_devices('GPU')  # 获取GPU列表
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
#     # 失效： tf.config.experimental.set_per_process_memory_fraction(0.25)
#     # 第一个参数为原则哪块GPU，只有一块则是gpu[0],后面的memory_limt是限制的显存大小，单位为M
#     tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*12)])

from model import Sample1,Generator,Discriminator_w
from data import read_data,merge

tf.random.set_seed(234)

x_train = read_data('train.h5')
# x_train = x_train[0:1280]
x_train = x_train[0:10001]

x_validation = read_data('train.h5')
x_validation = x_validation[10001:11001]

x_test = read_data('test_baby.h5')

x_train = tf.convert_to_tensor(x_train,dtype=tf.float32)/127.5 - 1.
x_validation = tf.convert_to_tensor(x_validation,dtype = tf.float32)/127.5 - 1.
x_test = tf.convert_to_tensor(x_test,dtype = tf.float32)/127.5 - 1.


train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(10000).batch(128)

validation_db = tf.data.Dataset.from_tensor_slices(x_validation)
validation_db = validation_db.batch(128)


test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(16)





def celoss_ones(logits):
    # [b, 1]

    logits = tf.squeeze(logits)
    loss = tf.reduce_mean(tf.square(logits - tf.ones_like(logits)))
    return tf.reduce_mean(loss)



def celoss_zeros(logits):
    logits = tf.squeeze(logits)
    loss = tf.reduce_mean(tf.square(logits-tf.zeros_like(logits)))
    return tf.reduce_mean(loss)


def gradient_penalty(discriminator, batch_x, fake_image):

    batchsz = batch_x.shape[0]

    # [b, h, w, c]
    t = tf.random.uniform([batchsz, 1, 1, 1])
    # [b, 1, 1, 1] => [b, h, w, c]
    t = tf.broadcast_to(t, batch_x.shape)

    interplate = t * batch_x + (1 - t) * fake_image

    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplote_logits = discriminator(interplate, training=True)
    grads = tape.gradient(d_interplote_logits, interplate)

    # grads:[b, h, w, c] => [b, -1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1) #[b]
    gp = tf.reduce_mean( (gp-1)**2 )

    return gp


def d_loss_fn(sample,model1,discriminator,batch_x,training):
    # 1. treat real image as real
    # 2. treat generated image as fake
    real_input1 = sample(batch_x)
    fake_image,_ = model1(real_input1, training)
    fake_input1 = sample(fake_image)
    real_inputs = (real_input1,batch_x)
    fake_inputs = (fake_input1,fake_image)
    d_real_logits = discriminator(real_inputs)
    d_fake_logits = discriminator(fake_inputs)


    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)
    #gp = gradient_penalty(discriminator, out, fake_image)

    loss = (d_loss_real + d_loss_fake)/2.

    return loss


def g_loss_fn(sample,model1,discriminator,batch_x, training):
    input1 = sample(batch_x)
    fake_image,_= model1(input1, training)
    input1 = sample(fake_image)
    inputs = (input1,fake_image)
    d_fake_logits = discriminator(inputs)
    loss2 = celoss_ones(d_fake_logits)
    fake_image = (fake_image+1)/2.
    batch_x = (batch_x+1)/2.
    loss1 = tf.reduce_mean(tf.abs(batch_x-fake_image))
    mse = tf.reduce_mean(tf.square(batch_x-fake_image))




    return loss1+0.001*loss2,mse,loss1,loss2


def VGGloss(true,fake):
    mod = VGG16(include_top=False,weights='imagenet')
    true = tf.concat([true,true,true],axis = -1)
    fake = tf.concat([fake,fake,fake],axis = -1)
    #print(true_out.shape)
    out = tf.reduce_mean(tf.square(mod(true)-mod(fake)))
    return out


def main(train = 0):
    #超参数：
    learning_rate = 0.0001
    #training = True
    best_loss = 1.
    best_loss_fc = 1
    train_best_loss = 1.
    train_best_loss_fc = 1

    sample = Sample1()

    model1 = Generator()

    discriminator = Discriminator_w()





    optimizer_g =optimizers.RMSprop(learning_rate = learning_rate)
    optimizer_d = optimizers.RMSprop(learning_rate=0.0001)


    if train == 1:

        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = 'logs/' + current_time
        summary_writer = tf.summary.create_file_writer(log_dir)

        for epoch in range(1):
            for step, x in enumerate(train_db):
                with tf.GradientTape() as type:
                    loss_d = d_loss_fn(sample,model1,discriminator,x,training = True)
                    print('epoch:',epoch,loss_d)
                grads_d = type.gradient(loss_d,discriminator.trainable_variables)
                optimizer_d.apply_gradients(zip(grads_d, discriminator.trainable_variables))


        for epoch in range(301):
            g_mean_loss = []
            d_mean_loss = []
            mean_loss1 = []
            mean_loss2 = []
            for step, x in enumerate(train_db):
                with tf.GradientTape() as type:
                    loss_d = d_loss_fn(sample, model1, discriminator, x, training=True)
                    d_mean_loss.append(loss_d)

                grads_d = type.gradient(loss_d, discriminator.trainable_variables)
                optimizer_d.apply_gradients(zip(grads_d, discriminator.trainable_variables))

                with tf.GradientTape() as type1:
                    loss_g, mse, loss1, loss2 = g_loss_fn(sample, model1, discriminator, x, training=True)
                    g_mean_loss.append(loss_g)
                    mean_loss1.append(loss1)
                    mean_loss2.append(loss2)
                grads_g = type1.gradient(loss_g, sample.trainable_variables + model1.trainable_variables)

                optimizer_g.apply_gradients(zip(grads_g, sample.trainable_variables + model1.trainable_variables))

            g_mean_loss = tf.reduce_mean(g_mean_loss)
            d_mean_loss = tf.reduce_mean(d_mean_loss)
            mean_loss1 = tf.reduce_mean(mean_loss1)
            mean_loss2 = tf.reduce_mean(mean_loss2)

            print('epoch:', epoch, 'loss-g:', float(g_mean_loss), 'loss1:', float(mean_loss1), 'loss2:', float(mean_loss2), 'loss-d:', float(d_mean_loss))
            with summary_writer.as_default():
                tf.summary.scalar('loss-g', float(g_mean_loss), step=epoch)
                tf.summary.scalar('loss1', float(mean_loss1), step=epoch)
                tf.summary.scalar('loss2', float(mean_loss2), step=epoch)
                tf.summary.scalar('loss-d', float(d_mean_loss), step=epoch)

            sample.save_weights(r'./last/sample/sample.ckpt')
            model1.save_weights(r'./last/model1/model1.ckpt')
            discriminator.save_weights(r'./last/discriminator/discriminator.ckpt')

            if epoch % 5 == 0:
                validation_loss = []
                for x in validation_db:
                    out = sample(x)
                    out,_ = model1(out)
                    out = (out+1)/2.
                    x = (x+1)/2.
                    loss = tf.reduce_mean(tf.square(out - x))
                    validation_loss.append(loss)
                loss = tf.reduce_mean(validation_loss)
                psnr = (10 * (math.log10(1.0 / loss)))
                if float(loss) < best_loss:
                    best_loss = float(loss)
                    sample.save_weights(r'./save_weights/sample/sample.ckpt')
                    model1.save_weights(r'./save_weights/model1/model1.ckpt')
                    discriminator.save_weights(r'./save_weights/discriminator/discriminator.ckpt')
                out = out*255.
                print('validation_loss:', float(loss), 'psnr:', psnr)



    else:
        sample.load_weights(r'./save_weights/sample/sample.ckpt')
        model1.load_weights(r'./save_weights/model1/model1.ckpt')


        for x in test_db:
            out = sample(x)
            out, _ = model1(out, training=False)
            out = (out + 1) / 2.
            x = (x + 1) / 2.
            out = merge(out,[4,4])
            x  = merge(x,[4,4])

            loss = tf.reduce_mean(tf.square(out - x))
            psnr = (10 * (math.log10(1.0 / loss)))
            ssim = tf.reduce_mean(tf.image.ssim(out, x, max_val=1))

            out = out * 255.
            x = x * 255.


        cv2.imwrite('./test_recon/baby_{}_{}.bmp'.format('%.4f'%psnr,'%.4f'%ssim), tf.squeeze(out).numpy())

        cv2.imwrite('test_recon/x_baby.png', tf.squeeze(x).numpy())


if __name__ == '__main__':
    main(train= 0)




