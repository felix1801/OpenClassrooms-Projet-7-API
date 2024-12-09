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
        'PAYMENT_RATE': 0, 
        'EXT_SOURCE_3': 0, 
        'EXT_SOURCE_2': 0, 
        'DAYS_BIRTH': 0, 
        'EXT_SOURCE_1': 0, 
        'DAYS_EMPLOYED_PERC': 0, 
        'ANNUITY_INCOME_PERC': 0, 
        'INSTAL_DBD_MEAN': 0, 
        'DAYS_LAST_PHONE_CHANGE': 0, 
        'REGION_POPULATION_RELATIVE': 0, 
        'ACTIVE_DAYS_CREDIT_UPDATE_MEAN': 0
    }])
    response = client.post('/predict', data=data, content_type='application/json')
    data = json.loads(response.get_data(as_text=True))

    assert response.status_code == 200
    assert all(0 <= proba <= 1 for proba in data['probas'])
    assert all(score == 0 or score == 1 for score in data['scores'])

def test_invalid_input_format(client):
    data = json.dumps([{
        'PAYMENT_RATE': 'bad_format_value', 
        'EXT_SOURCE_3': 0, 
        'EXT_SOURCE_2': 0, 
        'DAYS_BIRTH': 0, 
        'EXT_SOURCE_1': 0, 
        'DAYS_EMPLOYED_PERC': 0, 
        'ANNUITY_INCOME_PERC': 0, 
        'INSTAL_DBD_MEAN': 0, 
        'DAYS_LAST_PHONE_CHANGE': 0, 
        'REGION_POPULATION_RELATIVE': 0, 
        'ACTIVE_DAYS_CREDIT_UPDATE_MEAN': 0
    }])
    response = client.post('/predict', data=data, content_type='application/json')

    assert response.status_code == 400
