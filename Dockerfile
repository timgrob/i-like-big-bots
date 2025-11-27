FROM freqtradeorg/freqtrade:develop_plot

WORKDIR /freqtrade

RUN pip install --no-cache-dir pykalman

COPY user_data/strategies /freqtrade/user_data/strategies
COPY user_data/config.json /freqtrade/user_data/config.json
COPY .env /freqtrade
