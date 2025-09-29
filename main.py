from src.features import preprocess_input

if __name__ == "__main__":
    sample = {
        "SpO2": 88,
        "SBP": 120,
        "dolore_toracico": 1,
        "dispnea": 1,
        "alterazione_coscienza": 0
    }

    processed = preprocess_input(sample)
    print(processed)
