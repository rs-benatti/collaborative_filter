import numpy as np
import numpy.random as rd
from time import time

NUMBER_OF_CARAC = 10
NEGATIVE_THRESHOLD = 1e-6

R = np.load('./dataset/ratings_train.npy')

def relu(Y):
    return (Y+np.abs(Y))/2

R = np.nan_to_num(R, 0)

MAX_RATING = np.max(R)

def create_rating(W_user, W_item):
    nb_users, nb_items = R.shape
    p = np.zeros((nb_users, NUMBER_OF_CARAC))
    q = np.zeros((nb_items, NUMBER_OF_CARAC))

    for i in range(nb_users):
        a = W_user.dot(R[i,:])
        p[i,:] = a/np.sqrt(sum(a*a))
    for j in range(nb_items):
        b = W_item.dot(R[:,j])
        q[j,:] = b/np.sqrt(sum(b*b))

    Y_hat = p.dot(q.transpose())
    u = (Y_hat > NEGATIVE_THRESHOLD)
    Y_hat = Y_hat*u + NEGATIVE_THRESHOLD*(1 - u)

    return Y_hat

def W_matrix_aleatoire():

    W_user = rd.randn(NUMBER_OF_CARAC, R.shape[1])
    W_item = rd.randn(NUMBER_OF_CARAC, R.shape[0])

    return W_user, W_item

def loss(Y, Y_hat):
    log_Y_hat = np.log(Y_hat)
    log_inv_Y_hat = np.log(1 - Y_hat)
    Y_renormalized = Y/MAX_RATING
    Y_renormalized_inv = 1 - Y_renormalized

    loss_matrix = Y_renormalized*log_Y_hat + Y_renormalized_inv*log_inv_Y_hat
    loss = np.sum(loss_matrix)

    return -loss

def optimisation_directionnelle(W_user, W_item, dir_user, dir_item, Y):
    delta = np.linspace(-1, 1, 3)
    resultat = [(x, loss(Y, create_rating(W_user+x*dir_user, W_item+x*dir_item))) for x in delta]

    return resultat

def optimisation_aleatoire(R, W_user, W_item, iter = 1000): #iter a changer pour plus ou moins de précision

    score_so_far = loss(R, create_rating(W_user, W_item))

    begin = time()

    for i in range(iter):
        dir_user = rd.randn(W_user.shape[0], W_user.shape[1])/10
        dir_item = rd.randn(W_item.shape[0], W_item.shape[1])/10

        W_u2 = W_user + dir_user
        W_i2 = W_item + dir_item

        score = loss(R, create_rating(W_u2, W_i2))
        print(score)

        if score < score_so_far:
            W_user = W_u2
            W_item = W_i2
            score_so_far = score

    end = time()
    temps = end - begin
    print(temps)

    return W_user, W_item

def optimisation_de_tau_v1(R, W_user, W_item, dir_user, dir_item):
    tau = 10 # a changer pour plus de précision
    a = -tau
    b = tau

    for i in range(5): # a changer pour plus de précision
        taux = np.linspace(a, b, 7) # 7 is optimal (max de la fonction log((n-1)/2)/n)
        scores = [(t, loss(R, create_rating(W_user+t*dir_user, W_item+t*dir_item))) for t in taux]
        choix = min(scores, key = lambda x:x[1])[0]
        a,b = choix-(b-a)/6, choix+(b-a)/6

    W_u2 = W_user + ((a+b)/2)*dir_user
    W_i2 = W_item + ((a+b)/2)*dir_item

    return W_u2, W_i2

def optimisation_de_tau_v2(R, W_user, W_item, dir_user, dir_item):

    score_tau_m1 = loss(R, create_rating(W_user-1*dir_user, W_item-1*dir_item))
    score_tau_0 = loss(R, create_rating(W_user, W_item))
    score_tau_p1 = loss(R, create_rating(W_user+dir_user, W_item+dir_item))

    m = min(score_tau_0, score_tau_m1, score_tau_p1)
    if m == score_tau_m1:
        tau_opt = -1
    elif m == score_tau_p1:
        tau_opt = 1
    else:
        a = score_tau_p1 + score_tau_m1 - 2*score_tau_0
        b = (score_tau_p1 - score_tau_m1)/2
        tau_opt = -b/a
        score = loss(R, create_rating(W_user+tau_opt*dir_user, W_item+tau_opt*dir_item))
        if score >= score_tau_0:
            tau_opt = 0

    W_u2 = W_user + tau_opt*dir_user
    W_i2 = W_item + tau_opt*dir_item

    return W_u2, W_i2, m

def optimisation_aleatoire_tau_optimise(R, W_user, W_item, iter = 300): #iter a changer pour plus ou moins de précision

    begin = time()

    for i in range(iter):
        dir_user = rd.randn(W_user.shape[0], W_user.shape[1])/10
        dir_item = rd.randn(W_item.shape[0], W_item.shape[1])/10

        W_user, W_item, score = optimisation_de_tau_v2(R, W_user, W_item, dir_user, dir_item)

        print(score)

    end = time()
    temps = end - begin
    print(temps)

    return W_user, W_item

def gradient_ij(R, W_user, W_item, user, item):
    p_user = W_user.dot(R[user,:])
    q_item = W_item.dot(R[:,item])

    norm_p = np.sqrt(sum(p_user*p_user))
    norm_q = np.sqrt(sum(q_item*q_item))

    Y_hat_ij = sum(p_user*q_item/(norm_p*norm_q))
    if Y_hat_ij < NEGATIVE_THRESHOLD:
        return np.zeros(W_user.shape), np.zeros(W_item.shape)
    
    first_factor = (-R[user, item]/MAX_RATING)*(1/Y_hat_ij) \
                   + (1 - R[user, item]/MAX_RATING)*(1/(1 - Y_hat_ij))
    
    second_factor_users = np.zeros(NUMBER_OF_CARAC)
    for i in range(NUMBER_OF_CARAC):
        if p_user[i] <= 1e-8:
            second_factor_users[i] = 0
        else:
            second_factor_users[i] = q_item[i]/(norm_p*norm_q) - Y_hat_ij*p_user[i]/(norm_p**2)

    second_factor_items = np.zeros(NUMBER_OF_CARAC)
    for i in range(NUMBER_OF_CARAC):
        if p_user[i] <= 1e-8:
            second_factor_items[i] = 0
        else:
            second_factor_items[i] = p_user[i]/(norm_p*norm_q) - Y_hat_ij*q_item[i]/(norm_q**2)
        
    dir_user = first_factor*(second_factor_users.reshape(NUMBER_OF_CARAC, 1)).dot(R[user,:].reshape(1, R.shape[1]))
    dir_item = first_factor*(second_factor_items.reshape(NUMBER_OF_CARAC, 1)).dot(R[:,item].reshape(1, R.shape[0]))
    
    return dir_user, dir_item

def gradient(R, W_user, W_item, N = 1000):
    begin = time()
    dir_user = np.zeros(W_user.shape)
    dir_item = np.zeros(W_item.shape)

    for i in range(N):
        user = rd.randint(0, R.shape[0])
        item = rd.randint(0, R.shape[1])
        du, di = gradient_ij(R, W_user, W_item, user, item)
        dir_user += du
        dir_item += di

    end = time()
    temps = end - begin
    print(temps)

    return dir_user, dir_item

def gradient_descent(R, W_user, W_item, iter = 10):

    dir_user, dir_item = gradient(R, W_user, W_item)

    for i in range(iter):
        W_user, W_item, score = optimisation_de_tau_v2(R, W_user, W_item, dir_user, dir_item)
        print(i, score)

    return W_user, W_item

if '__name__' == '__main__':
    print(2)

i = 2