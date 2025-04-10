# 📍 AI-Based Indoor Positioning System (IPS)

Ce projet vise à développer un système de localisation en intérieur utilisant des techniques d'apprentissage profond (Deep Learning) pour estimer les coordonnées **(X, Y)** à partir de données capteurs (IMU, magnétomètre) et de signaux Wi-Fi (RSSI).

## 🧠 Objectif

Prédire avec précision la position d’un utilisateur à l’intérieur d’un bâtiment en se basant sur les données issues de :
- Accéléromètre, gyroscope, magnétomètre (IMU)
- Force du signal Wi-Fi reçu (RSSI)

## 📁 Contenu du dépôt

| Fichier / Dossier                         | Description |
|------------------------------------------|-------------|
| `pos_estim.py`                           | Script principal de modélisation et d’entraînement pour la localisation. |
| `data_square_motion_samsung.csv`         | Données brutes de capteurs et RSSI correspondant à un trajet en carré. |
| `ETAI_IndoorProject.pdf`                 | Présentation technique détaillant le projet, ses étapes et objectifs. |
| `Quand l’intelligence artificielle...pdf`| Article de vulgarisation scientifique expliquant le contexte du projet. |
| `Poster.pdf`                             | Poster scientifique sur le projet (beam prediction). |
| `indoor-loc_.png`                        | Illustration du trajet intérieur. |
| `vulgarisation.png`                      | Schéma simplifié pour la vulgarisation du projet. |

## 🧪 Étapes du projet

1. **Exploration des données**
   - Visualisation des capteurs (Accel, Gyro, Magneto) et des signaux WiFi.
   - Analyse des valeurs manquantes, en particulier pour les positions X et Y.

2. **Prétraitement**
   - Estimation des positions manquantes (interpolation, modélisation supervisée).
   - Normalisation des données, sélection des features pertinentes.

3. **Modélisation**
   - Réseau de neurones (DNN ou LSTM) avec en entrée : données capteurs + RSSI.
   - Sortie : estimation des coordonnées (X, Y).

4. **Entraînement & Évaluation**
   - Séparation des jeux de données : entraînement / validation / test.
   - Utilisation du MAE comme métrique principale.
   - Visualisation des positions prédites vs réelles.

## 🛠️ Pré-requis

- Python ≥ 3.8
- Bibliothèques : `numpy`, `pandas`, `matplotlib`, `tensorflow` ou `pytorch`, `sklearn`

```bash
pip install -r requirements.txt
```

## 📊 Résultats attendus

Erreur de localisation moyenne : < 2 mètres

Visualisations des trajectoires réelles vs prédites

Analyse des performances du modèle sur différents trajets

## 🧾 Références
[Mokdadi et al., 2025] S. Mokdadi, S. E. Bouzid, and P. Chargé, "Millimeter-Wave Beam Prediction with Inverse Beamforming ML Model," in Proc. 7th Int. Conf. on Advances in Signal Processing and Artificial Intelligence (ASPAI), Innsbruck, Austria, Apr. 2025.

[Zafari et al., 2019] A Survey of Indoor Localization Systems and Technologies. IEEE Communications Surveys & Tutorials

[Bouzid et al., 2021] S. E. Bouzid, A. Simondet, and P. Chargé, "Artificial neural network-based indoor localization system using smartphone magnetometer," in Proc. IEEE Conf. on Antenna Measurements & Applications (CAMA), Antibes Juan-les-Pins, France, 2021, pp. 438–443. doi: 10.1109/CAMA49227.2021.9703502.

Poster ASPAI 2025 (beam prediction)

Vulgarisation LinkedIn par Salah Eddine Bouzid, 2025


## 👨‍🏫 Auteur

Salaheddine Bouzid

Enseignant-Chercheur à Polytech Nantes

📧 salaheddine.bouzid@univ-nantes.fr
