from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and prepare the dataset
file_path = 'dataset/dataset.xlsx'

try:
    data = pd.read_excel(file_path)
    
    # Selected features for training
    X = data[['truck_slots', 'car_slots', 'arrival_rate', 'queue_length', 
              'service_rate', 'budget', 'max_truck_slots', 'queue_prob', 'gates_needed']]
    y = data['avg_time']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Calculate the average values for internal features not provided by the user
    avg_service_rate = data['service_rate'].mean()
    avg_budget = data['budget'].mean()
    avg_max_truck_slots = data['max_truck_slots'].mean()
    avg_queue_prob = data['queue_prob'].mean()
    avg_gates_needed = data['gates_needed'].mean()

except FileNotFoundError:
    data = pd.DataFrame()
    print("File not found.")
except Exception as e:
    data = pd.DataFrame()
    print(f"An error occurred: {e}")

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user inputs
        user_data = {
            'truck_slots': float(request.form.get('truck_slots')),
            'car_slots': float(request.form.get('car_slots')),
            'arrival_rate': float(request.form.get('arrival_rate')),
            'queue_length': float(request.form.get('queue_length'))
        }
        
        # Add average values for internal features
        user_data['service_rate'] = avg_service_rate
        user_data['budget'] = avg_budget
        user_data['max_truck_slots'] = avg_max_truck_slots
        user_data['queue_prob'] = avg_queue_prob
        user_data['gates_needed'] = avg_gates_needed

        # Convert to DataFrame
        input_df = pd.DataFrame([user_data])

        # Make prediction
        prediction = model.predict(input_df)[0]

        return render_template('results.html', prediction=prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
