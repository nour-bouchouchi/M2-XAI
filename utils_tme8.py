import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

def SL(x,a0,a1,n):
    """
    Génère n points dans la couche sphérique de centre x et de rayon interne a0 et de rayon externe a1
    x : centre de la sphere 
    a0 : rayon interne
    a1 : rayon externe
    n : nombre de points générés 
    """
    d = x.shape[0] #dimension des données
    z = np.random.normal(0,1,(n,d)) # on génère des points suivant une loi normal

    norms = np.linalg.norm(z, axis=1) # on calcule la norme par ligne 
    z = z / norms[:, np.newaxis] # on divise chaque ligne par sa norme 
    
    u = (np.random.uniform(a0**d,a1**d,n))**(1/d)

    res = x + z*u[:,np.newaxis] #
    return res 


def GS(f,x,eta,n):
    """
    Algorithme des growking sphere : retourne l'enemy le plus proche de x  avec la distance l2 minimale (point de classe opposé le plus proche de x)
    f : classifieur binaire
    x : observation à interpréter
    eta : épaisseur de la sphere 
    n : nombre de points générés  
    """
    z = SL(x, 0, eta, n)
    predict_x = f.predict([x])

    while(not np.all(predict_x == f.predict(z))): 
        eta = eta/2
        z = SL(x,0,eta,n)

    a0 = eta
    a1 = 2*eta

    while (np.all(predict_x== f.predict(z))):
        z = SL(x,a0,a1,n)
        a0 = a1
        a1 += eta

    liste_e = z[np.where(predict_x != f.predict(z))[0]] #liste des enemy
    l2 = np.sqrt( np.sum((x - liste_e)**2, axis=1) ) #distance l2 calculée pour chaque enemy


    return liste_e[np.argmin(l2)] #selection de l'enemy avec la distance l2 minimale 



def feature_selection(f,x,e):
    """    
    Retourne l'enemy qui minimise le nombre d’attributs modifiés
    f : classifieur binaire
    x : observation à interpréter
    e : enemy le plus proche de x au sens l2
    """
    e_prim = e.copy()
    e_star = e_prim.copy()
    while f.predict([e_prim]) != f.predict([x]) :
        e_star = e_prim.copy()
        ind = np.where(e_prim != x )[0] 
        abs = np.absolute(e_prim[ind]-x[ind]) #valeurs absolues des différences entre e_prim_j et x_j (tels que : e_prim_j != x_j  )
        i = ind[np.argmin(abs)]
        e_prim[i] = x[i]
    
    return e_star


def affiche_frontiere2D(model, X_train):
    """    
    Affiche la frontière de décision d'un modèle
    model : modèle de détection d'anomalies
    X_train : les données d'apprentissage  
    """
    # Génération d'une grille
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Prédiction chaque point de la grille
    data_grid = np.c_[xx.ravel(), yy.ravel()]
    confidence = model.decision_function(data_grid)
    confidence = confidence.reshape(xx.shape)

    # Affichage du dégradé de couleurs
    plt.contourf(xx, yy, confidence, cmap='Blues', alpha=0.5)


def affichage(prediction, isolation, x, y, data, titre="Détection d'anomalies avec Isolation Forest"):
    # Séparer les anomalies et les points normaux
    anomalies_x = x[prediction == -1]
    anomalies_y = y[prediction == -1]

    normal_x = x[prediction == 1]
    normal_y = y[prediction == 1]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    axs[0].scatter(normal_x, normal_y, s=10, label='Points normaux')
    axs[0].scatter(anomalies_x, anomalies_y, s=50, color='r', marker='x', label='Anomalies')
    axs[0].set_title(titre)
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].legend()

    scatter = axs[1].scatter(normal_x, normal_y, s=10, label='Points normaux', cmap='cividis')
    scatter = axs[1].scatter(anomalies_x, anomalies_y, s=50, color='r', marker='x', label='Anomalies', cmap='cividis')
    axs[1].set_title(titre)
    affiche_frontiere2D(isolation, data)
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].legend()
    #fig.colorbar(scatter, ax=axs[1], label='Confiance (Degré de non-anomalie)', cmap='cividis')
    plt.tight_layout()
    plt.show()



def affichage_contrefactuel(prediction, isolation, x, y, instance, contrefactuel, data, titre="Détection d'anomalies avec Isolation Forest"):
    # Séparer les anomalies et les points normaux
    anomalies_x = x[prediction == -1]
    anomalies_y = y[prediction == -1]

    normal_x = x[prediction == 1]
    normal_y = y[prediction == 1]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    axs[0].scatter(normal_x, normal_y, s=10, label='Points normaux')
    axs[0].scatter(anomalies_x, anomalies_y, s=30, color='r', marker='x', label='Anomalies')
    axs[0].set_title(titre)
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].legend()

    scatter = axs[1].scatter(normal_x, normal_y, s=10, label='Points normaux', cmap='cividis')
    scatter = axs[1].scatter(anomalies_x, anomalies_y, s=30, color='r', marker='x', label='Anomalies', cmap='cividis')
    axs[1].set_title(titre)
    affiche_frontiere2D(isolation, data)
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")

    # Tracer l'instance et le contrefactuel au premier plan
    axs[0].scatter(instance[0], instance[1], s=70, color='green', marker='o', label='Instance')
    axs[1].scatter(instance[0], instance[1], s=40, color='green', marker='o', label='Instance')
    axs[1].scatter(contrefactuel[0], contrefactuel[1], s=40, color='orange', marker='o', label='Contrefactuel')


    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()
    plt.show()


def etude_contamination(x, y, liste_contamination, data):

    fig, axs = plt.subplots(nrows=1, ncols=len(liste_contamination), figsize=(5*len(liste_contamination), 5))

    liste_anomalies_x = []
    liste_anomalies_y = []
    liste_scores = []
    for i in range(len(liste_contamination)): 
        # Modèle IsolationForest
        isolation_forest = IsolationForest(contamination=liste_contamination[i])  
        isolation_forest.fit(data)

        # Prédiction des anomalies
        prediction = isolation_forest.predict(data)
        anomalies_x = x[prediction == -1]
        anomalies_y = y[prediction == -1]

        normal_x = x[prediction == 1]
        normal_y = y[prediction == 1]


        axs[i].scatter(normal_x, normal_y, s=10, label='Points normaux')
        axs[i].scatter(anomalies_x, anomalies_y, s=50, color='r', marker='x', label='Anomalies')
        axs[i].set_title(f"contamination = {liste_contamination[i]}")
        axs[i].set_xlabel("X")
        axs[i].set_ylabel("Y")
        axs[i].legend()

        liste_anomalies_x.append(anomalies_x)
        liste_anomalies_y.append(anomalies_y)
        scores = isolation_forest.score_samples(np.column_stack((x, y)))
        liste_scores.append(scores)

    plt.tight_layout()
    plt.show()
    return liste_anomalies_x, liste_anomalies_y, liste_scores







