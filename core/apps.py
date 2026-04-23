import os
import joblib
from django.apps import AppConfig
from django.conf import settings

class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'
    model_stacking = None

    def ready(self):
        # On charge le modèle uniquement si on n'est pas en mode "commande" (migration, etc.)
        model_path = os.path.join(settings.ML_MODELS_ROOT, 'sentiment_stacking_pro_v2.joblib')
        if os.path.exists(model_path):
            self.model_stacking = joblib.load(model_path)
            print("✅ Modèle Stacking chargé avec succès !")