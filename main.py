from freqtrade import main as freqtrade_main


def main():
    freqtrade_main.main(
        [
            "trade",
            "--config",
            "user_data/config.json",
            "--strategy",
            "EmaSlopeStrategy",  # add your own strategy name here
        ]
    )


if __name__ == "__main__":
    main()
