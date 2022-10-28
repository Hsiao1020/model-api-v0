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

# @app.before_request
# def limit_remote_addr():
#     if not request.remote_addr.startswith('140.119'):
#         abort(404)  # Forbidden

@app.route("/predict", methods=["POST"])
def lstm_predict():
    params = request.get_json()
    print(params)
    if not params:
        return jsonify({"Error": "Can't find any parameters"})
    else:
        BEGIN = params['Begin']
        END = params['End']
        FEATURES = params['Features']
        OHLC = params['OHLC']
        PREDICTED_TICKET = params['Predicted_ticket'].upper()
        INDEX_FEATURES = params['Index_features']
        RATIO_OF_TRAIN = params['Ratio_of_train']
        LOOK_BACK = params['Look_back']
        FORECAST_DAYS = params['Forecast_days']
        LAYERS = params['Layers']
        LEARNING_RATE = params['Learning_rate']
        EPOCHS = params['Epochs']
        BATCH_SIZE = params['Batch_size']

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
        BEGIN = params['Begin']
        END = params['End']
        FEATURE = params['Feature']
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
        BEGIN = params['Begin']
        END = params['End']
        FEATURE = params['Feature']
        INDEX = params['Index']
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
        BEGIN = params['Begin']
        END = params['End']
        FEATURE = params['Feature']
        TIMEPERIOD = params['Timeperiod']
        index = calculate_moving_average(
            begin=BEGIN,
            end=END,
            feature=FEATURE,
            timeperiod=TIMEPERIOD
        )

        return jsonify(index)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
    # app.run(debug=True)


# example params
{
    "Begin": "2020-07-27",
    "End": "2022-07-28",
    "Ratio_of_train": 0.7,
    "Look_back": 5,
    "Forecast_days": 1,
    "OHLC": "Adj Close",
    "Features": ["BTC-USD", "^DJI", "^GSPC"],
    "Index_features": ["BBANDS", "MA"],
    "Predicted_ticket": "BTC-USD",
    "Layers": [
        {"units": 50, "dropout": 0.2}
    ],
    "Learning_rate": 0.01,
    "Epochs": 50,
    "Batch_size": 32
}

