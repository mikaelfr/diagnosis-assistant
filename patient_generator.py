import random
import pandas as pd

def random_normal(mu: float, sigma: float, hard_min: float = None, hard_max: float = None) -> float:
    """ Returns a random float from a normal distribution, will reroll if outside [hard_min, hard_max] """
    result = random.gauss(mu, sigma)
    while (hard_min is not None and result < hard_min) or (hard_max is not None and result > hard_max):
        result = random.gauss(mu, sigma)
    return result

def random_bool(probability: float) -> int:
    """ Returns a 0 if uniform random float [0, 1) is below the given probability, otherwise returns 1 """
    return 0 if random.random() < probability else 1

diagnosesMap = {
    'normal': 0,
    'stroke': 1,
    'cao': 2,
    'pulm_emb': 3,
    'dehydration': 4,
    'sepsis': 5
}

def is_normal(value: float, referenceRange: tuple) -> tuple:
    """
    Returns tuple (bool, int)
    int == -1: lower
    int == 0: normal
    int == 1: higher
    """
    rangeInterval = referenceRange[1] - referenceRange[0]
    if value > referenceRange[1]:
        # inverse square type deal
        prob = 1 / 1 + ((value - referenceRange[1]) / rangeInterval)
        return (random_bool(prob) == 1, 1)
    elif value < referenceRange[0]:
        prob = 1 / 1 + ((referenceRange[0] - value) / rangeInterval)
        return (random_bool(prob) == 1, -1)

    return (True, 0)

# TODO: can have multiple?
def get_diagnosis(measurements: dict) -> int:
    """ 
    Returns the class of diagnosis and also has a chance of it being normal
    The closer to reference values, the higher chance of being normal
    See diagnoses map for mapping from str -> int
    """
    #pulse_normal = is_normal(measurements['pulse'], ())
    sap_normal = is_normal(measurements['sap'], (90, 120))
    spo2_normal = is_normal(measurements['spo2'], (94, 100))
    t_normal = is_normal(measurements['t'], (36, 37.5))
    # TODO: no heart rate, or non measurement indicators

    # TODO: within reference values but still diagnosis
    if not t_normal[0] and t_normal[1] > 0 and not sap_normal[0] and sap_normal[1] < 0 and not spo2_normal[0] and spo2_normal[1] < 0:
        return diagnosesMap['sepsis']
    elif not spo2_normal[0] and spo2_normal[1] < 0:
        return diagnosesMap['pulm_emb']
    elif not sap_normal[0] and sap_normal[1] < 0 :
        # TODO: now basically a coin toss
        return diagnosesMap['dehydration'] if random_bool(0.5) == 1 else diagnosesMap['cao']
    elif not sap_normal[0] and sap_normal[1] > 0:
        # TODO: now basically a coin toss
        return diagnosesMap['stroke'] if random_bool(0.5) == 1 else diagnosesMap['cao']

    return diagnosesMap['normal']

def generate_healthy_person():
    """ Generate a healthy person and return their measurements in a dict (requires some capping for values I guess) """
    pulse = random_normal(86, 13.5)
    spo2 = random_normal(97, 1.5, hard_max=100)
    sap = random_normal(141, 18)
    t = random_normal(36.8, 0.5)
    rr = random_normal(16, 2)
    gluc = random_normal(7.2, 1.55)
    age = random_normal(60.6, 21.4, hard_min=0, hard_max=100)
    # 0 = man, 1 = woman
    sex = random_bool(0.483)

    measurements = { 'pulse': pulse, 'spo2': spo2, 'sap': sap, 't': t, 'rr': rr, 'gluc': gluc, 'age': age, 'sex': sex }
    diagnosis = get_diagnosis(measurements)
    return { **measurements, 'diagnosis': diagnosis }

def generate_people(n: int = 10000) -> pd.DataFrame:
    """ Generate an n number of people and return their data in a dataframe """
    people_dict = [generate_healthy_person() for _ in range(n)]
    return pd.DataFrame(data=people_dict)

if __name__ == '__main__':
    generated_people = generate_people()
    print(generated_people)
