import pytest
import requests
import json

BASE_URL = "http://127.0.0.1:8001"

# Test de la route d'accueil
def test_welcome():
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    assert response.text == '"Welcome to the API"'

# Test de la recherche d'un client
def test_check_client_id():
    client_id = 192535  
    response = requests.get(f"{BASE_URL}/{client_id}")
    assert response.status_code == 200
    assert response.json() in [True, False]

# Test de la prédiction d'un client
def test_get_prediction():
    client_id = 192535  # Remplace avec un ID valide
    response = requests.get(f"{BASE_URL}/prediction/{client_id}")
    assert response.status_code == 200
    assert isinstance(response.json(), float)

# Test des clients similaires
import json  


def test_get_data_voisins():
    client_id = 192535  # Remplace avec un ID valide
    response = requests.get(f"{BASE_URL}/clients_similaires/{client_id}")
    assert response.status_code == 200

    json_response = json.loads(response.text)  # Désérialisation de la réponse
    assert isinstance(json_response, list)  # Vérifie que c'est bien une liste
    assert all(isinstance(item, dict) for item in json_response)  # Vérifie que chaque élément est un dictionnaire


# Test des SHAP values locales
def test_shap_values_local():
    client_id = 192535  # Remplace avec un ID valide
    response = requests.get(f"{BASE_URL}/shaplocal/{client_id}")
    assert response.status_code == 200
    json_response = response.json()
    assert 'shap_values' in json_response
    assert 'base_value' in json_response
    assert 'data' in json_response
    assert 'feature_names' in json_response

# Test des SHAP values globales
def test_shap_values():
    response = requests.get(f"{BASE_URL}/shap/")
    assert response.status_code == 200
    json_response = response.json()
    assert 'shap_values_0' in json_response
    assert 'shap_values_1' in json_response
