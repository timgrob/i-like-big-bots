FROM freqtradeorg/freqtrade:develop_plot

WORKDIR /freqtrade

RUN pip install --no-cache-dir pykalman

COPY --chown=ftuser:ftuser user_data/strategies/ /freqtrade/user_data/strategies/
COPY --chown=ftuser:ftuser user_data/config.json /freqtrade/user_data/config.json

CMD ["trade", "--config", "user_data/config.json", "--strategy", "MyCustomStrategy"]
