from flask import Flask, jsonify

import get_ticker_data as gtd
app = Flask(__name__)


@app.route('/')
def render_home_page():
    pass


@app.route('/update', methods=['POST'])
def update_data():
    gtd.update_sector_tickers('Information Technology')
    response_data = {'status': 'success', 'message': 'Data updated successfully'}
    gtd.update_sector_tickers('Information Technology')
    return jsonify(response_data)


if __name__ == '__main__':
    gtd.update_sector_tickers('Information Technology')
