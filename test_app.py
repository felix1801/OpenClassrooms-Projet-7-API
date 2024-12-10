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
        'AMT_ANNUITY': 0,
        'AMT_CREDIT': 0,
        'AMT_GOODS_PRICE': 0,
        'CODE_GENDER': 0,
        'DAYS_EMPLOYED': 0,
        'EXT_SOURCE_1': 0,
        'EXT_SOURCE_2': 0,
        'EXT_SOURCE_3': 0,
        'NAME_EDUCATION_TYPE_Highereducation': 0,
        'OWN_CAR_AGE': 0,
        'PAYMENT_RATE': 0,
    }])
    response = client.post('/predict', data=data, content_type='application/json')
    data = json.loads(response.get_data(as_text=True))

    assert response.status_code == 200
    assert all(0 <= proba <= 1 for proba in data['probas'])
    assert all(score == 0 or score == 1 for score in data['scores'])

def test_invalid_input_format(client):
    data = json.dumps([{
        'AMT_ANNUITY': 'bad_format_value',
        'AMT_CREDIT': 0,
        'AMT_GOODS_PRICE': 0,
        'CODE_GENDER': 0,
        'DAYS_EMPLOYED': 0,
        'EXT_SOURCE_1': 0,
        'EXT_SOURCE_2': 0,
        'EXT_SOURCE_3': 0,
        'NAME_EDUCATION_TYPE_Highereducation': 0,
        'OWN_CAR_AGE': 0,
        'PAYMENT_RATE': 0,
    }])
    response = client.post('/predict', data=data, content_type='application/json')

    assert response.status_code == 400
