import requests

sample = {
    'data': [[
        180.0,                 # Adult Mortality
        20,                   # infant deaths
        5.0,                  # Alcohol
        1000.0,               # percentage expenditure
        85.0,                 # Hepatitis B
        10,                   # Measles
        23.5,                 # BMI
        15,                   # under-five deaths
        80.0,                 # Polio
        4.5,                  # Total expenditure
        85.0,                 # Diphtheria
        0.1,                  # HIV/AIDS
        5000.0,               # GDP
        1500000,              # Population
        3.0,                  # thinness  1-19 years
        3.2,                  # thinness 5-9 years
        0.7,                  # Income composition of resources
        12.0,                 # Schooling
        1,                    # Status_Developed (one-hot)
        0                     # Status_Developing (one-hot)
    ]]
}

url = 'http://127.0.0.1:8000/predict'
response = requests.post(url, json=sample)

print("Response:", response.json())