# API de Scoring Crédit

## Vue d'ensemble
Ce projet implémente un système de scoring crédit basé sur l'apprentissage automatique pour évaluer les demandes de prêt. Le système utilise un modèle LightGBM personnalisé, entraîné sur des données historiques de crédit, avec une attention particulière portée aux contraintes métier et à l'optimisation des coûts.

## Fonctionnalités principales
- Optimisation personnalisée du seuil prenant en compte les coûts métier (coût FN = 10 × coût FP)
- Implémentation MLOps avec suivi des expériences et registre des modèles via MLflow
- Pipeline de tests automatisés et déploiement continu
- API REST pour les prédictions de scoring en temps réel
- Interface de test locale avec un notebook

## Structure du projet
```
.
├── .github/
│   └── workflows/
│       └── main.yml          # Pipeline CI/CD pour les tests et le déploiement Heroku
├── models/
│   └── latest/
│       └── model.pkl         # Modèle de production sérialisé
├── app.py                    # Point d'entrée de l'application Flask
├── custom_threshold_model.py # Implémentation LGBM avec seuil optimisé
├── model.py                  # Fonctionnalités et méthodes du modèle
├── Procfile                  # Configuration pour le déploiement Heroku
├── requirements.txt         # Dépendances du projet
├── runtime.txt              # Spécification de la version Python
└── test_app.py             # Suite de tests de l'API
```

## Contexte métier
Le modèle répond à deux défis métier majeurs :
1. **Déséquilibre des classes** : Le jeu de données présente une distribution déséquilibrée entre bons et mauvais clients, gérée par des techniques d'échantillonnage appropriées.
2. **Coûts asymétriques** : Les Faux Négatifs (accorder un prêt à un mauvais client) sont considérés comme 10 fois plus coûteux que les Faux Positifs (refuser un prêt à un bon client).

## API Endpoint
L'API expose un endpoint principal :
- `/predict` : Renvoie à la fois la probabilité de défaut et la décision (accepté/refusé) basée sur le seuil optimisé pour un client donné.

Format de réponse :
```json
{
    "probability": 0.23,
    "score": 1,
    "threshold": 0.3 
}
```

## Développement local
Pour exécuter le projet en local :
1. Installer les dépendances : `pip install -r requirements.txt`
2. Démarrer le serveur Flask : `python app.py`
3. Lancer le notebook d'interface de test

## Tests
Le projet inclut des tests automatisés utilisant pytest. Pour exécuter les tests en local :
```bash
pytest test_app.py -v
```

## Déploiement
L'application est automatiquement déployée sur Heroku via GitHub Actions lors des modifications sur la branche principale. Le pipeline de déploiement comprend :
1. Exécution des tests automatisés
2. Construction de l'application
3. Déploiement sur Heroku si tous les tests passent

## Implémentation MLOps
Le projet suit les bonnes pratiques MLOps avec :
- Suivi des expériences via MLflow
- Versionnage et registre des modèles
- Tests et déploiement automatisés
- Intégration continue via GitHub Actions