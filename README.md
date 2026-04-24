# Sentiment-Analysis-Project# 🎭 Sentiment Analysis Dashboard - Cyber-Lab

![Django](https://img.shields.io/badge/Framework-Django%205.0-092e20?style=for-the-badge&logo=django)
![Python](https://img.shields.io/badge/Python-3.12-3776ab?style=for-the-badge&logo=python)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffbd45?style=for-the-badge)

Une application web moderne d'analyse de sentiments permettant de comparer plusieurs approches d'Intelligence Artificielle : du lexique classique au Deep Learning de pointe.

## 🌟 Fonctionnalités

- **Analyse Instantanée** : Testez une phrase unique en temps réel.
- **Analyse par Lot (Batch)** : Importez un fichier **CSV** et analysez des centaines d'avis en un clic.
- **Système Multi-Modèles** :
    - **VADER** : Approche lexicale ultra-rapide optimisée pour les réseaux sociaux.
    - **DistilBERT** : Modèle de Transformer (Deep Learning) pour une compréhension contextuelle profonde.
    - **Stacking Classifier (PRO)** : Notre modèle personnalisé combinant plusieurs algorithmes pour une précision optimisée (76%).
- **Visualisation Dynamique** : Graphiques interactifs (Donut, Bar, Radar) via Chart.js.
- **Historique de Session** : Gardez une trace de vos analyses précédentes.

## 🧠 Les Modèles Embarqués

| Modèle | Type | Force |
| :--- | :--- | :--- |
| **VADER** | Lexical (NLTK) | Rapidité & Emojis |
| **DistilBERT** | Transformer (HF) | Contexte & Précision |
| **Stacking Pro** | Ensemble Learning | Robustesse & Stabilité |

## 🏗️ Architecture du Projet (Pipeline)
Le flux de données suit une architecture structurée allant de la collecte des données à la visualisation sur le dashboard.

![Architecture du Projet](static/images/project_architecture.png)
*Description : Pipeline de traitement, de l'injection du CSV via Django à l'inférence des modèles et au rendu Chart.js.*

## 🧠 Architecture du Modèle (Stacking Ensemble)
Notre modèle "PRO" repose sur une stratégie de Stacking, fusionnant les prédictions de plusieurs classifieurs de base pour une décision finale plus robuste.

![Architecture du Modèle](static/images/model_architecture.png)
*Description : Architecture à deux niveaux (Base Learners & Meta-Learner) avec extraction de caractéristiques TF-IDF.*

## 🚀 Installation et Lancement

### 1. Cloner le projet
```bash
git clone [https://github.com/votre-pseudo/sentiment-analysis-project.git](https://github.com/Mahdi88BH/Sentiment-Analysis-Project.git)
cd sentiment-analysis-project
