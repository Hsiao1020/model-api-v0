from distutils.log import debug
from flask import Flask, request, jsonify
from lstm_predict import predict

app = Flask(__name__)


@app.route("/")
def home():
    return "Hello World"


@app.route("/predict/lstm", methods=["POST"])
def lstm_predict():
    params = request.get_json()
    print(params)
    if not params:
        return jsonify({"Error": "Can't find any parameters"})
    else:
        BEGIN = params['begin']
        END = params['end']
        FEATURES = params['features']
        OHLC = params['OHLC']
        PREDICTED_TICKET = params['predicted_ticket'].upper()
        RATIO_OF_TRAIN = params['ratio_of_train']
        LOOK_BACK = params['look_back']
        FORECAST_DAYS = params['forecast_days']
        LAYERS = params['layers']
        LEARNING_RATE = params['learning_rate']
        EPOCHS = params['epochs']
        BATCH_SIZE = params['batch_size']

        predict_data = predict(
            begin=BEGIN,
            end=END,
            features=FEATURES,
            OHLC=OHLC,
            predict_ticket=PREDICTED_TICKET,
            ratio_of_train=RATIO_OF_TRAIN,
            look_back=LOOK_BACK,
            forecast_days=FORECAST_DAYS,
            layers=LAYERS,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )

        return jsonify(predict_data)


if __name__ == "__main__":
    app.run()


# example params
schema = {
    "begin": "2018-07-27",
    "end": "2022-07-28",
    "ratio_of_train": 0.7,
    "look_back": 5,
    "forecast_days": 1,
    "OHLC": "Adj Close",
    "features": ["BTC-USD", "^DJI", "^GSPC", "MWL=F"],
    "predicted_ticket": "BTC-USD",
    "layers": [
        {"units": 50, "dropout": 0.2}
    ],
    "learning_rate": 0.01,
    "epochs": 50,
    "batch_size": 32
}
