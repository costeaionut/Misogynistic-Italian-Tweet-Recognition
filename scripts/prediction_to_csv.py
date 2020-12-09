import pandas as pd

def write_to_csv(predict_labels):
    
    predict_id = []
    for idx in range(5001, 6001):
        predict_id.append(idx)

    dataSet = {
        'id': predict_id,
        'label': predict_labels
    }

    dataFrame = pd.DataFrame(dataSet, columns=['id', 'label'])
    dataFrame.to_csv(r'/home/costea/Documents/Facultate/Anul 3/IA/mysoginTweeterRecognition/data/prediction.csv', index=False, header=True)
