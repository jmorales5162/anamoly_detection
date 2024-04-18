import numpy as np

params = {
    'figsize'           : (4, 3),                       #tamaño de la figura 
    'xticks'            : np.arange(-7.5, 12, 2.5),     #escala del eje x de la grafica
    'yticks'            : np.arange(-7.5, 12, 2.5),     #escala del eje y de la grafica
    'title'             : 'Agrupamiento obtenido',      #titulo de la grafica
    'legend_loc'        : "upper right",                 #localización de la leyenda [upper right],[lower left]

    'fsxtick'           : 10,
    'fsytick'           : 10,

}
params1 = {
    'figsize'           : (10, 10),                     #tamaño de la figura 
    'xticks'            : None,                         #escala del eje x de la grafica
    'yticks'            : None,                         #escala del eje y de la grafica
    'title'             : 'Agrupamiento obtenido por DBSCAN al aplicar TSNE',      #titulo de la grafica
    'legend_loc'        : "lower right",                #localización de la leyenda [upper right],[lower left]

    'fsxtick'           : 10,
    'fsytick'           : 10,

}
params_inertial = {
    'figsize'           : (7, 3),
    'title'             : None,
    'xlabel'            : 'k',
    'ylabel'            : 'Inercia',

    'xticks'            : None,
    'yticks'            : None,
    
    'fsxlbl'            : 12,
    'fsylbl'            : 12,

    'fsxtick'           : 10,
    'fsytick'           : 10,

    'maker'             : 'o',
}
params_silhouette = {
    'figsize'           : (7, 3),
    'title'             : None,
    'xlabel'            : 'k',
    'ylabel'            : 'Coeficiente de silueta',
    
    'xticks'            : None,
    'yticks'            : None,

    'fsxlbl'            : 12,
    'fsylbl'            : 12,

    'fsxtick'           : 10,
    'fsytick'           : 10,

    'maker'             : 'o',
}
params_nh = {
    'figsize'           : (4, 3),                       #tamaño de la figura 
    'title'             : None,
    'xlabel'            : 'Puntos ordenados por distancia al k-vecino mas cercano',
    'ylabel'            : 'Distancia al k-vecino más cercano',

    'xticks'            : np.arange(0, 225, 25),     #escala del eje x de la grafica
    'yticks'            : np.arange(0, 0.8, 0.1),     #escala del eje y de la grafica

    'fsxlbl'            : 8,
    'fsylbl'            : 8,

    'fsxtick'           : 8,
    'fsytick'           : 8,

    'maker'             : None,
}
params_nh_m32 = {
    'figsize'           : (4, 3),                       #tamaño de la figura 
    'title'             : None,
    'xlabel'            : None,
    'ylabel'            : None,

    'xticks'            : np.arange(0, 1200, 100),     #escala del eje x de la grafica
    'yticks'            : np.arange(0, 60, 10),     #escala del eje y de la grafica

    'fsxlbl'            : 8,
    'fsylbl'            : 8,

    'fsxtick'           : 8,
    'fsytick'           : 8,

    'maker'             : None,
}