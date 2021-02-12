from collections import OrderedDict
from functools import reduce
import numpy as np
import tensorflow as tf
from operator import mul
import vgg


class StyleApplier:
    """
    Klasa nakładająca styl z parametru style na podany obrazek
    """

    def __init__(self):
        """
        Inicjalizacja parametrów startowych.
        ustawiane są warstwy obrazka oraz stylu przekazanego na początku programu
        """
        self.style_loss = 0
        self.content_loss = 0
        self.content_layers = ('relu4_2', 'relu5_2')
        self.style_layers = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
        self.style_layers_weights = {}

    def get_loss_vals(self, loss_store):
        """
        Zwraca wartości strat
        """
        return OrderedDict((key, val.eval()) for key, val in loss_store.items())

    def stylize(self, network, content, styles, iterations, content_weight, content_weight_blend, style_weight,
                style_layer_weight_exp, style_blend_weights, tv_weight, learning_rate, beta1, beta2, epsilon, pooling):
        """
        Nałożenie stylu na obraz
        Metoda jest wywoływana iteracyjnie, obliczane są straty i wagi, a potem do rodzica jest przekazywany
        tuple z iteratorem i tablicą obrazu oraz, jeśli to ostatnia iteracja, z obliczonymi stratami

        :rtype: iterator[tuple[int,image]]
        """
        self.style_features = [{} for _ in styles]
        self.content_features = {}
        self.style_shapes = [(1,) + style.shape for style in styles]
        self.shape = (1,) + content.shape
        self.vgg_weights, vgg_mean_pixel = vgg.load_net(network)
        self.layer_weight = 1.0
        for style_layer in self.style_layers:
            self.style_layers_weights[style_layer] = self.layer_weight
            self.layer_weight *= style_layer_weight_exp

        self.calculate_sum_weight()
        self.calculate_content_feature(pooling, content, vgg_mean_pixel)
        self.calculate_style_feature(styles, pooling, vgg_mean_pixel)

        # Użycie propagacji wstecznej na stylizowanym obrazie
        with tf.Graph().as_default():
            initial = tf.random_normal(self.shape) * 0.256
            self.image = tf.Variable(initial)
            self.net = vgg.net_preloaded(self.vgg_weights, self.image, pooling)

            self.calculate_content_loss(content_weight_blend, content_weight)
            self.calculate_style_loss(styles, style_weight, style_blend_weights)
            self.denoise_image(tv_weight)
            self.calculate_total_loss()

            # konfiguracja optymalizatora
            train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(self.loss)

            # optymalizacja
            best_loss = float('inf')
            best = None
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for i in range(iterations):
                    if i > 0:
                        print('%4d/%4d' % (i + 1, iterations))
                    else:
                        print('%4d/%4d' % (i + 1, iterations))
                    train_step.run()

                    last_step = (i == iterations - 1)
                    if last_step:
                        loss_vals = self.get_loss_vals(self.loss_store)
                    else:
                        loss_vals = None

                    if last_step:
                        this_loss = self.loss.eval()
                        if this_loss < best_loss:
                            best_loss = this_loss
                            best = self.image.eval()

                        img_out = vgg.unprocess(best.reshape(self.shape[1:]), vgg_mean_pixel)
                    else:
                        img_out = None

                    yield i + 1 if last_step else i, img_out, loss_vals

    def calculate_sum_weight(self):
        """
        obliczanie sumy ważonej warstw stylu
       """
        layer_weights_sum = 0
        for style_layer in self.style_layers:
            layer_weights_sum += self.style_layers_weights[style_layer]
        for style_layer in self.style_layers:
            self.style_layers_weights[style_layer] /= layer_weights_sum

    def calculate_content_feature(self, pooling, content, vgg_mean_pixel):
        """
        Obliczanie właściwości obrazu
        """
        g = tf.Graph()
        with g.as_default(), g.device('/cpu:0'), tf.Session():
            self.image = tf.placeholder('float', shape=self.shape)
            self.net = vgg.net_preloaded(self.vgg_weights, self.image, pooling)
            content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])
            for layer in self.content_layers:
                self.content_features[layer] = self.net[layer].eval(feed_dict={self.image: content_pre})

    def calculate_style_feature(self, styles, pooling, vgg_mean_pixel):
        """
        Obliczanie właściwości stylu
        """
        for i in range(len(styles)):
            g = tf.Graph()
            with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
                self.image = tf.placeholder('float', shape=self.style_shapes[i])
                self.net = vgg.net_preloaded(self.vgg_weights, self.image, pooling)
                style_pre = np.array([vgg.preprocess(styles[i], vgg_mean_pixel)])
                for layer in self.style_layers:
                    features = self.net[layer].eval(feed_dict={self.image: style_pre})
                    features = np.reshape(features, (-1, features.shape[3]))
                    gram = np.matmul(features.T, features) / features.size
                    self.style_features[i][layer] = gram

    def calculate_content_loss(self, content_weight_blend, content_weight):
        """
        Obliczanie strat dla obrazu
        """
        content_layers_weights = {'relu4_2': content_weight_blend, 'relu5_2': 1.0 - content_weight_blend}

        content_losses = []
        for content_layer in self.content_layers:
            content_losses.append(content_layers_weights[content_layer] * content_weight * (2 * tf.nn.l2_loss(
                self.net[content_layer] - self.content_features[content_layer]) / self.content_features[
                                                                                                content_layer].size))
        self.content_loss += reduce(tf.add, content_losses)

    def calculate_style_loss(self, styles, style_weight, style_blend_weights):
        """
        Obliczanie strat dla stylu
        """
        for i in range(len(styles)):
            style_losses = []
            for style_layer in self.style_layers:
                layer = self.net[style_layer]
                _, height, width, number = map(lambda i: i.value, layer.get_shape())
                size = height * width * number
                feats = tf.reshape(layer, (-1, number))
                gram = tf.matmul(tf.transpose(feats), feats) / size
                style_gram = self.style_features[i][style_layer]
                style_losses.append(
                    self.style_layers_weights[style_layer] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
            self.style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)

    def denoise_image(self, tv_weight):
        """
        Odszumianie generowanego obrazu
        """
        tv_y_size = self.tensor_size(self.image[:, 1:, :, :])
        tv_x_size = self.tensor_size(self.image[:, :, 1:, :])
        self.tv_loss = tv_weight * 2 * (
                (tf.nn.l2_loss(self.image[:, 1:, :, :] - self.image[:, :self.shape[1] - 1, :, :]) /
                 tv_y_size) +
                (tf.nn.l2_loss(self.image[:, :, 1:, :] - self.image[:, :, :self.shape[2] - 1, :]) /
                 tv_x_size))

    def calculate_total_loss(self):
        """
        Obliczanie wszystkich strat
        """
        self.loss = self.content_loss + self.style_loss + self.tv_loss
        self.loss_store = OrderedDict([('content', self.content_loss),
                                  ('style', self.style_loss),
                                  ('tv', self.tv_loss),
                                  ('total', self.loss)])

    def tensor_size(self, tensor):
        """
        Oblicza rozmiar tensora (multi wymiarowy array)
        :param tensor: tensor do zmierzenia
        :return: rozmiar podanego tensora
        """
        return reduce(mul, (d.value for d in tensor.get_shape()), 1)

    def time_parser(self, seconds):
        """
        Przedstawia czas w jednostkach h - godzina, m - minuta, s - sekunda
        :param seconds: ilość sekund do konwersji
        :return:
        """
        seconds = int(seconds)
        hours = (seconds // (60 * 60))
        minutes = (seconds // 60) % 60
        seconds = seconds % 60
        if hours > 0:
            return '%d h %d m' % (hours, minutes)
        elif minutes > 0:
            return '%d m %d s' % (minutes, seconds)
        else:
            return '%d s' % seconds
