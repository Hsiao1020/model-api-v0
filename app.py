from distutils.log import debug
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from lstm_predict import predict
from index import calculate_all_index, calculate_index, calculate_moving_average
app = Flask(__name__)
CORS(app=app)


@app.route("/")
def home():
    return "請不要關掉 app.py 拜託 感謝~"

@app.before_request
def limit_remote_addr():
    if not request.remote_addr.startswith('140.119'):
        abort(404)  # Forbidden

@app.route("/predict", methods=["POST"])
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
        INDEX_FEATURES = params['index_features']
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
            index_features=INDEX_FEATURES,
            ratio_of_train=RATIO_OF_TRAIN,
            look_back=LOOK_BACK,
            forecast_days=FORECAST_DAYS,
            layers=LAYERS,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )

        return jsonify(predict_data)

@app.route("/index/all", methods=["POST"])
def get_index_all():
    params = request.get_json()
    print(params)
    if not params:
        return jsonify({"Error": "Can't find any parameters"})
    else:
        BEGIN = params['begin']
        END = params['end']
        FEATURE = params['feature']
        all_index = calculate_all_index(
            begin=BEGIN,
            end=END,
            feature=FEATURE
        )

        return jsonify(all_index)

@app.route("/index", methods=["POST"])
def get_index():
    params = request.get_json()
    print(params)
    if not params:
        return jsonify({"Error": "Can't find any parameters"})
    else:
        BEGIN = params['begin']
        END = params['end']
        FEATURE = params['feature']
        INDEX = params['index']
        index = calculate_index(
            begin=BEGIN,
            end=END,
            feature=FEATURE,
            index=INDEX
        )

        return jsonify(index)

@app.route("/index/ma", methods=["POST"])
def get_moving_average():
    params = request.get_json()
    print(params)
    if not params:
        return jsonify({"Error": "Can't find any parameters"})
    else:
        BEGIN = params['begin']
        END = params['end']
        FEATURE = params['feature']
        TIMEPERIOD = params['timeperiod']
        index = calculate_moving_average(
            begin=BEGIN,
            end=END,
            feature=FEATURE,
            timeperiod=TIMEPERIOD
        )

        return jsonify(index)


if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8888)
    # app.run(debug=True)


# example params
{
    "begin": "2020-07-27",
    "end": "2022-07-28",
    "ratio_of_train": 0.7,
    "look_back": 5,
    "forecast_days": 1,
    "OHLC": "Adj Close",
    "features": ["BTC-USD", "^DJI", "^GSPC"],
    "index_features": ["BBANDS", "MA"],
    "predicted_ticket": "BTC-USD",
    "layers": [
        {"units": 50, "dropout": 0.2}
    ],
    "learning_rate": 0.01,
    "epochs": 50,
    "batch_size": 32
}

