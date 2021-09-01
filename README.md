# Deep-learning-applied-to-forest-fire-detection

El presente repositorio contiene una base de datos de imágenes térmicas, una base de datos de imágenes a color, varios scripts de algoritmos de deep learning utilizando la técnica de Transfer Learning, un script de clasificación por color, y scripts de una interfaz web. 

Toda esta información fue utilizada en el desarrollo de un trabajo de grado en Ingeniería electrónica titulado "Diseño e Implementación de un Sistema de Detección de Conatos de Incendios en Campo Abierto Basado en Visión por Computadora y un VANT".

### Database_color

Es una base de datos de imágenes a color, está compuesta por 1800 imágenes de incendios forestales y 1800 imágenes de no incendios forestales.

### Database_thermal

Es una base de datos de imágenes térmicas, está compuesta por 2650 imágenes de incendios forestales y 2650 imágenes de no incendios forestales.

### Interfaz web

Son los codigos y los archivos utilizados en la interfaz web desarrollada en esta investigación.

### Scripts train

Son códigos de entrenamiento de CNN utilizando PyTorch en python, se utilizaron 4 arquitecturas diferentes de CNN (ResNet, VggNet, SqueezeNet, DenseNet), de las cuales se utilizaron todos los modelos disponibles en PyTorch.

### Classifier_color

Es un código en python, el cuál fue utilizado para clasificar por color las imagenes térmicas.

### Mask_thermal

Es el código en python, el cuál fue utilizado para definir las mascaras utilizadas en el clasificador por color.



