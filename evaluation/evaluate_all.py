import evaluate_light
import evaluate_gas
import evaluate_audio


# Calculate entropy from raw sensor data found at:
# https://zenodo.org/record/8263497
def calculate_entropy(file_path, light=True, gas=True, audio=False):
    # Entropy of light and gas data (released data)
    if light:
        evaluate_light.calc_entropy(file_path)
    if gas:
        evaluate_gas.calc_entropy(file_path)

    # We do not release audio data, but its entropy is calculated as follows
    if audio:
        evaluate_audio.calc_entropy(file_path)


# Calculate dtw distance for raw sensor data for light and gas recordings found at:
# https://zenodo.org/record/8263497
def calculate_dtw(file_path, light=True, gas=True):
    if light:
        # We compute DTW distance on chunks of light data, e.g., chunks of 0.25, 0.5, 1 minute, etc.
        evaluate_light.evaluate(file_path, minutes_split=0.25, normalize_data=True)
        evaluate_light.evaluate(file_path, minutes_split=0.5, normalize_data=True)
        evaluate_light.evaluate(file_path, minutes_split=1, normalize_data=True)
        evaluate_light.evaluate(file_path, minutes_split=1.5, normalize_data=True)
        evaluate_light.evaluate(file_path, minutes_split=2, normalize_data=True)
        evaluate_light.evaluate(file_path, minutes_split=5, normalize_data=True)

        # We compute DTW distance of full light data recorded in our experiment
        evaluate_light.evaluate(file_path, minutes_split=None, normalize_data=True)

    if gas:
        # We compute DTW distance on chunks of gas data, e.g., chunks of 0.5, 1, 2.5 minutes, etc.
        evaluate_gas.evaluate(file_path, minutes_split=0.5, normalize_data=True)
        evaluate_gas.evaluate(file_path, minutes_split=1, normalize_data=True)
        evaluate_gas.evaluate(file_path, minutes_split=2.5, normalize_data=True)
        evaluate_gas.evaluate(file_path, minutes_split=5, normalize_data=True)
        evaluate_gas.evaluate(file_path, minutes_split=10, normalize_data=True)
        evaluate_gas.evaluate(file_path, minutes_split=15, normalize_data=True)

        # We compute DTW distance of full gas data recorded in our experiment
        evaluate_gas.evaluate(file_path, minutes_split=None, normalize_data=True)


if __name__ == '__main__':
    # Example of computing entropy and DTW distance for light and gas data for the Home scenario:
    # data is inside the "hardzipa-data" folder from https://zenodo.org/record/8263497
    filepath = 'C:/Users/mfomichev/Desktop/hardzipa-data/Home'

    calculate_entropy(filepath)
    calculate_dtw(filepath)
