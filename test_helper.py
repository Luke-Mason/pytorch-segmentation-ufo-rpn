
restricted = False
def dataset_gateway(arr):
    return arr[:2] if restricted else arr[:5]

def epoch_gateway(epochs):
    return 2 if restricted else epochs