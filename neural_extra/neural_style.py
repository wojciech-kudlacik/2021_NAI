"""
Autorzy: Norbert Daniluk, Wojciech Kudłacik
Ten program pozwala na wygenerowanie nowego obrazaka na podstawie podanego stylu np. a'la Van Gogh
Głównym silnikiem programu jest tensorflow.
"""

import math
import imageio
from PIL import Image
import numpy as np
import scipy.misc
from stylize import StyleApplier


class ArtConverter:
    """
    Główna klasa uruchamiająca program
    """

    def __init__(self, image, style, output):
        """
        Inicjalizacja parametrów startowych
        image: wejściowe zdjęcie, które ma zostać zamienione
        style: styl na podstawie, którego zdjęcie ma zostać zamienione
        output: zdjęcie po przeróbce
        """
        self.image = image # obrazek
        self.style = style # styl
        self.output = output # wynik programu
        self.content_weight = 5e0 # waga obrazu
        self.content_weight_blend = 1  # współczynnik określający przenoszenie treści między warstwami. Im wyższy, tym więcej treści obrazu zostanie przekazane do outputu
        self.style_weight = 5e2 # waga stylu
        self.tv_weight = 1e2 # wariacja całkowita wag
        self.style_layer_weight_exp = 1  # określa jak obraz ma zostać przerobiony na bardziej abstrakcyjny. 1 jest wartością pośrednią
        self.learning_rate = 1e1
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-08
        self.style_scale = 1.0
        self.iterations = 10 # ilość iteracji, które ma przejść program
        self.vgg_path = 'imagenet-vgg-verydeep-19.mat'  # https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
        self.pooling = 'max'  # max lub avg. max ma tendencję do przenoszenia drobniejszych detali, ale może mieć problemy z niższymi rozdzielczościami
        self.img_width = 100 # szerokość obrazka w pixelach (wysokość dostosowywana proporcjonalnie)

    def run(self):
        """
        Uruchomienie głównego programu
        """

        # wybranie obrazka i stylu
        content_image = self.readImg(self.image)
        style_images = [self.readImg(self.style) for style in self.style]

        # ustawienie wielkości obrazka
        new_shape = (
            int(math.floor(float(content_image.shape[0]) / content_image.shape[1] * self.img_width)), self.img_width)
        content_image = scipy.misc.imresize(content_image, new_shape)
        target_shape = content_image.shape
        print(len(style_images))
        for i in range(len(style_images)):
            style_images[i] = scipy.misc.imresize(style_images[i],
                                                  self.style_scale * target_shape[1] / style_images[i].shape[1])

        # opcje rozmycia
        style_blend_weights = [1.0 / len(style_images) for _ in style_images]

        # iteracja nakładająca styl na obraz
        stylize_obj = StyleApplier()
        for iteration, image, loss_vals in stylize_obj.stylize(
                self.vgg_path,
                content_image,
                style_images,
                self.iterations,
                self.content_weight,
                self.content_weight_blend,
                self.style_weight,
                self.style_layer_weight_exp,
                style_blend_weights,
                self.tv_weight,
                self.learning_rate,
                self.beta1,
                self.beta1,
                self.epsilon,
                self.pooling,
        ):
            pass

        self.saveImg(self.output, image)

    def readImg(self, path):
        """
        Metoda wczytująca plik jako numpy float
        path: ścieżka do pliku
        return: img - obrazek jako tablica wartości
        """
        img = imageio.imread(path).astype(np.float)
        return img

    def saveImg(self, path, img):
        """
        Metoda zapisująca obrazek wyjściowy
        path: ścieżka do pliku
        img: obrazek, który chcemy zapisać
        """
        img = np.clip(img, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(path, quality=95)


artConverter = ArtConverter("content.jpg", "style.jpg", "output.jpg")
artConverter.run()
