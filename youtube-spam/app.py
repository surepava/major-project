from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('./model2.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_message = request.form['message']
        processed_message = process_message(input_message)
        result = predict_result(processed_message)
        return render_template('result.html', input_message=input_message, result=result)

def process_message(input_message):
    # Add your preprocessing steps here
    # For example, converting to lowercase
    processed_message = input_message.lower()
    return processed_message

def predict_result(processed_message):
    # Add your prediction logic here
    prediction = model.predict([processed_message])
    if prediction[0] == 1:
        return "Spam"
    else:
        return "Not Spam"

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
