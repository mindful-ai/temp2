from flask import Flask
from housingapp import housing

app = Flask(__name__)

@app.route('/')
def index():
    return "<h1>My first Flask Application</h1>"

@app.route('/getprice')
def getpredictions():
    h = housing()
    m = h.loadmodel()
    predicted_price = m.predict([[100000, 15, 5, 3, 50000]])
    return "<h1>{}</h1>".format(predicted_price)


if __name__ == "__main__":
    app.run(debug=True)