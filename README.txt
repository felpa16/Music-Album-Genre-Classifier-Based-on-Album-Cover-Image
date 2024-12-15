Link al docs con los generos clasificados:
https://docs.google.com/document/d/1IcoRHM3r8_oV6LH0khRfKEihelxNIYt52BFO355FL2s/edit?tab=t.0#heading=h.69oh3ulkf4pp


El siguiente proyecto consta de distintos archivos en el que se encuentra el código necesario para 
la clasificación de géneros musicales a partir de imágenes de portadas de álbumes. Para ello, se han
utilizado distintos modelos. 

En las carpetas se tiene el nombre de un modelo y un distinctivo para reconocer el tipo de modelo.

- Data Augmentation
- Deleted Undersampling (Eliminacion de la clase 10 y 16, limitando las muestras a 500 por clase)
- Undersampling Oversampling (Balanceo de clases)
- HF Dataset (Dataset nuevo de Hugging Face con 20 mil muestras y 20 clases) 