from minisom import MiniSom
from sklearn.datasets import load_breast_cancer
import time
from pylab import *

# Set hyperparametres
som_grid_rows = 30
som_grid_columns = 20
iterations = 500
sigma = 1
learning_rate = 0.5

# Load dataset
data, target = load_breast_cancer(True)

#initialisation
som = MiniSom(x=som_grid_rows, y=som_grid_columns, input_len= data.shape[1],
              sigma = sigma, learning_rate=learning_rate)
som.random_weights_init(data)

#training
som.train_random(data, iterations)


bone()
pcolor(som.distance_map().T)
colorbar()

markers = ['o','s','D']
colors = ['r','g','b']
for cnt,xx in enumerate(data):
    w = som.winner(xx)
    plot(w[0]+.5,w[1]+.5,markers[target[cnt]],markerfacecolor='None',
        markeredgecolor=colors[target[cnt]],markersize=12, markeredgewidth=2)
axis([0,som._weights.shape[0],0,som._weights.shape[1]])
show()
