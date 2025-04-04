url_get_prediction = "https://dashboard-2-6eo5.onrender.com/prediction/192535"
response = requests.get(url_get_prediction)

if response.status_code == 200:
    print("Prediction received:", response.json())
else:
    print(f"Error {response.status_code}: Unable to get prediction")
