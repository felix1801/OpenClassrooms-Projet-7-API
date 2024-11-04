# faire des tests avec unitest ou pytest

# 2 tests, par exemple

# est-ce que ça retourne bien une erreur quand y'a une erreur
# est-ce que ça retourne bien 0 ou 1 en score ? entre 0 et 1 pour la proba ?

import pytest
import json
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_prediction_range(client):
    data = json.dumps([{
        'PAYMENT_RATE': 0.0361471518987341, 
        'EXT_SOURCE_3': 0.1595195404777181, 
        'EXT_SOURCE_2': 0.7896543511176771, 
        'DAYS_BIRTH': -19241.0, 
        'EXT_SOURCE_1': 0.7526144906031748, 
        'DAYS_EMPLOYED_PERC': 0.1210436048022452, 
        'ANNUITY_INCOME_PERC': 0.1523, 
        'INSTAL_DBD_MEAN': 8.857142857142858, 
        'DAYS_LAST_PHONE_CHANGE': -1740.0, 
        'REGION_POPULATION_RELATIVE': 0.01885, 
        'ACTIVE_DAYS_CREDIT_UPDATE_MEAN': -10.666666666666666
    }])
    response = client.post('/predict', data=data, content_type='application/json')
    data = json.loads(response.get_data(as_text=True))

    assert response.status_code == 200
    assert all(0 <= proba <= 1 for proba in data['probas'])
    assert all(score == 0 or score == 1 for score in data['scores'])

def test_invalid_input(client):
    data = json.dumps([{
        'PAYMENT_RATE': 'slow', 
        'EXT_SOURCE_3': 0.1595195404777181, 
        'EXT_SOURCE_2': 0.7896543511176771, 
        'DAYS_BIRTH': -19241.0, 
        'EXT_SOURCE_1': 0.7526144906031748, 
        'DAYS_EMPLOYED_PERC': 0.1210436048022452, 
        'ANNUITY_INCOME_PERC': 0.1523, 
        'INSTAL_DBD_MEAN': 8.857142857142858, 
        'DAYS_LAST_PHONE_CHANGE': -1740.0, 
        'REGION_POPULATION_RELATIVE': 0.01885, 
        'ACTIVE_DAYS_CREDIT_UPDATE_MEAN': -10.666666666666666
    }])
    response = client.post('/predict', data=data, content_type='application/json')

    assert response.status_code == 400
