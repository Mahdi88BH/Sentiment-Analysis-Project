from django.shortcuts import render
from django.apps import apps
from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import json
from datetime import datetime

# Initialisation des modèles
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()
hf_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def get_predictions(texts, model_choice):
  
    results = []
    try:
        if model_choice == 'custom':
            model = apps.get_app_config('core').model_stacking
            preds = model.predict(texts)
            labels = {0: "Négatif", 1: "Neutre", 2: "Positif"}
            results = [labels[p] for p in preds]
        
        elif model_choice == 'nltk':
            for t in texts:
                score = sia.polarity_scores(t)['compound']
                results.append("Positif" if score > 0.05 else ("Négatif" if score < -0.05 else "Neutre"))
                
        elif model_choice == 'hf':
            hf_results = hf_pipe(texts, truncation=True)
            label_map = {"POSITIVE": "Positif", "NEGATIVE": "Négatif"}
            results = [label_map.get(r['label'], "Neutre") for r in hf_results]
    except Exception:
        results = ["Neutre"] * len(texts)

    return {
        'Négatif': results.count('Négatif'),
        'Neutre': results.count('Neutre'),
        'Positif': results.count('Positif')
    }

def home(request):
    
    if 'analysis_history' not in request.session or not isinstance(request.session['analysis_history'], list):
        request.session['analysis_history'] = []
    
    # 2. Calcul des statistiques globales (AVEC PROTECTION)
    def calculate_global_stats(history_list):
        total = {'Positif': 0, 'Neutre': 0, 'Négatif': 0}
        for entry in history_list:
            # Vérification stricte de l'existence et du type de 'scores'
            entry_scores = entry.get('scores')
            if isinstance(entry_scores, dict):
                for k in total:
                    total[k] += entry_scores.get(k, 0)
        return total

    global_stats = calculate_global_stats(request.session['analysis_history'])

    context = {
        'history': request.session['analysis_history'],
        'global_stats_json': json.dumps(global_stats)
    }
    
    if request.method == 'POST':
        new_history_entry = {}

        # --- CAS 1 : ANALYSE UNITAIRE ---
        if 'text_input' in request.POST:
            text = request.POST.get('text_input', '')
            model_choice = request.POST.get('model_choice', 'custom')
            
            res_data = {'result': 'Neutre', 'confidence': 0, 'model_used': 'Erreur'}
            try:
                if model_choice == 'custom':
                    model = apps.get_app_config('core').model_stacking
                    if model:
                        pred = model.predict([text])[0]
                        prob = model.predict_proba([text])[0]
                        labels = {0: "Négatif", 1: "Neutre", 2: "Positif"}
                        res_data = {'result': labels[pred], 'confidence': round(max(prob) * 100, 2), 'model_used': "Stacking Pro"}
                
                elif model_choice == 'nltk':
                    score = sia.polarity_scores(text)['compound']
                    res = "Positif" if score > 0.05 else ("Négatif" if score < -0.05 else "Neutre")
                    res_data = {'result': res, 'confidence': round(abs(score)*100, 2), 'model_used': "NLTK VADER"}

                elif model_choice == 'hf':
                    res = hf_pipe(text)[0]
                    label_map = {"POSITIVE": "Positif", "NEGATIVE": "Négatif"}
                    res_data = {'result': label_map.get(res['label'], "Négatif"), 'confidence': round(res['score'] * 100, 2), 'model_used': "Hugging Face"}
            except Exception as e:
                print(f"Erreur unitaire: {e}")

            context.update(res_data)
            
            current_scores = {'Positif': 0, 'Neutre': 0, 'Négatif': 0}
            current_scores[res_data['result']] = 1

            new_history_entry = {
                'type': 'Unitaire',
                'preview': text[:30] + "..." if len(text) > 30 else text,
                'model': res_data.get('model_used'),
                'result': res_data.get('result'),
                'scores': current_scores,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            }

        # --- CAS 2 : ANALYSE PAR LOT (CSV) ---
        elif 'csv_file' in request.FILES:
            file = request.FILES['csv_file']
            model_choice = request.POST.get('model_choice_batch', 'compare')
            try:
                df = pd.read_csv(file)
                possible_cols = ['Text', 'text', 'review', 'comment']
                col_name = next((c for c in possible_cols if c in df.columns), df.columns[0])
                texts = df[col_name].astype(str).tolist()

                if model_choice == 'compare':
                    stats_custom = get_predictions(texts, 'custom')
                    stats_hf = get_predictions(texts, 'hf')
                    stats_nltk = get_predictions(texts, 'nltk')
                    current_scores = stats_custom 
                    batch_res = {
                        'compare_mode': True,
                        'stats_json': json.dumps({'Stacking': stats_custom, 'HuggingFace': stats_hf, 'NLTK': stats_nltk}),
                        'total_reviews': len(texts),
                        'table_preview': df.head(5).to_html(classes='table table-sm', index=False)
                    }
                    model_name = "Comparaison Triple"
                else:
                    current_scores = get_predictions(texts, model_choice)
                    batch_res = {
                        'batch_mode': True,
                        'stats_json': json.dumps(current_scores),
                        'total_reviews': len(texts),
                        'model_used': model_choice,
                        'table_preview': df.head(5).to_html(classes='table table-sm', index=False)
                    }
                    model_name = f"Batch ({model_choice})"

                context.update(batch_res)
                new_history_entry = {
                    'type': 'CSV',
                    'preview': file.name,
                    'model': model_name,
                    'result': f"{len(texts)} avis",
                    'scores': current_scores,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                }
            except Exception as e:
                context.update({'error': f"Erreur fichier: {e}"})

        # 3. Sauvegarde sécurisée de la session
        if new_history_entry:
            history = request.session.get('analysis_history', [])
            history.insert(0, new_history_entry)
            request.session['analysis_history'] = history[:10]
            request.session.modified = True
            
            # Recalcul des stats avec la nouvelle entrée
            updated_global = calculate_global_stats(request.session['analysis_history'])
            context['history'] = request.session['analysis_history']
            context['global_stats_json'] = json.dumps(updated_global)

    return render(request, 'core/index.html', context)