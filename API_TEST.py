import requests
import json

API_KEY = '2c82b7c5-260a-42fb-828c-3961436b6272'
API_URL = 'https://www.vegvesen.no/ws/no/vegvesen/kjoretoy/felles/datautlevering/enkeltoppslag/kjoretoydata?kjennemerke='
LICENSE_PLATE = 'EF49897'

headers = {
    'SVV-Authorization': f'Apikey {API_KEY}'
}

response = requests.get(API_URL + LICENSE_PLATE, headers=headers)

if response.status_code == 200:
    data = response.json()
    with open('api_output.json', 'w') as f:
        json.dump(data, f)
else:
    print(f"Request failed with status code {response.status_code}")