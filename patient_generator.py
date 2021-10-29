import math
import random
import pandas as pd
from classifier import classfier_col_order, categorical

def random_normal(mu: float, sigma: float, hard_min: float = None, hard_max: float = None) -> float:
    """ Returns a random float from a normal distribution, will reroll if outside [hard_min, hard_max] """
    result = random.gauss(mu, sigma)
    while (hard_min is not None and result < hard_min) or (hard_max is not None and result > hard_max):
        result = random.gauss(mu, sigma)
    return result

def random_bool(probability: float) -> int:
    """ Returns a 0 if uniform random float [0, 1) is below the given probability, otherwise returns 1 """
    return 0 if random.random() < probability else 1

diagnoses_map = {
    'normal': 0,
    'stroke': 1,
    'cao': 2,
    'pulm_emb': 3,
    'dehydration': 4,
    'sepsis': 5
}

diagnoses_params = {
    1: {'sap': 0.3, 'motor_impairment': 1},
    2: {'sap': [-0.3, 0.3], 'chest_pain': 1},
    3: {'spo2': -0.1, 'chest_pain': 1},
    4: {'pulse': 0.5, 'sap': -0.3},
    5: {'t': 0.1, 'pulse': 0.5, 'sap': -0.3, 'spo2': -0.1},
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
def get_diagnosis_and_add_symptoms(measurements: dict) -> int:
    """ 
    Returns the class of diagnosis and also has a chance of it being normal
    The closer to reference values, the higher chance of being normal
    See diagnoses map for mapping from str -> int
    """
    pulse_normal = is_normal(measurements['pulse'], (60, 100))
    sap_normal = is_normal(measurements['sap'], (90, 120))
    spo2_normal = is_normal(measurements['spo2'], (94, 100))
    t_normal = is_normal(measurements['t'], (36, 37.5))

    # default to no
    measurements['chest_pain'] = 0
    measurements['motor_impairment'] = 0

    # TODO: within reference values but still diagnosis
    if not t_normal[0] and t_normal[1] > 0 and not sap_normal[0] and sap_normal[1] < 0 and not spo2_normal[0] and spo2_normal[1] < 0:
        return diagnoses_map['sepsis']
    elif not spo2_normal[0] and spo2_normal[1] < 0:
        # 80% chance of having chest pain with pulmary embolism
        measurements['chest_pain'] = random_bool(0.2)
        return diagnoses_map['pulm_emb']
    elif not sap_normal[0] and sap_normal[1] < 0 and not pulse_normal[0] and pulse_normal[1] > 0:
        return diagnoses_map['dehydration']
    elif not sap_normal[0] and sap_normal[1] > 0:
        motor_impairment = random_bool(0.5)
        measurements['motor_impairment'] = motor_impairment
        if bool(motor_impairment):
            return diagnoses_map['stroke']
        else:
            # 80% chance of having chest pain with cao
            measurements['chest_pain'] = random_bool(0.2)
            return diagnoses_map['cao']

    return diagnoses_map['normal']

def generate_person():
    """ Generate a person and return their measurements in a dict (requires some capping for values I guess) """
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
    diagnosis = get_diagnosis_and_add_symptoms(measurements)
    values = { **measurements, 'diagnosis': diagnosis }

    # remove some values to simulate missing measurements
    final_values = {}
    for key in values.keys():
        # 10% chance per value to be missing
        remove = random_bool(0.9)
        if not bool(remove):
            final_values[key] = values[key]
    
    # diagnosis cannot be missing
    final_values['diagnosis'] = values['diagnosis']

    return final_values

def generate_people(n: int = 10000) -> pd.DataFrame:
    """ Generate an n number of people and return their data in a dataframe """
    people_dict = [generate_person() for _ in range(n)]
    df = pd.DataFrame(data=people_dict)
    df = df[classfier_col_order]
    return df


# TODO: if time, allow changing of all parameters in key places (patients, example ui, etc.)

# alt way

limits = {
    'pulse': {'min': 60, 'max': 100, 'slack_percent': 0.2},
    'sap': {'min': 90, 'max': 120, 'slack_percent': 0.2},
    'spo2': {'min': 94, 'max': 100, 'slack_percent': 0.2},
    't': {'min': 36, 'max': 37.5, 'slack_percent': 0.33},
}

# generate a healthy person (keep rerolling till within parameters above, allow some slack)
def generate_person_healthy():
    """ Generate a healthy person and return their measurements in a dict """
    pulse = random_normal(86, 13.5)
    spo2 = random_normal(97, 1.5, hard_max=100)
    sap = random_normal(141, 18)
    t = random_normal(36.8, 0.5)
    rr = random_normal(16, 2)
    gluc = random_normal(7.2, 1.55)
    age = random_normal(60.6, 21.4, hard_min=0, hard_max=100)
    # random chest pain even if healthy
    chest_pain = 1 - random_bool(0.02)
    # drunk? migraine?
    motor_impairment = 1 - random_bool(0.01)
    # 0 = man, 1 = woman
    sex = random_bool(0.483)

    value_params = {
        'pulse': {'args': (86, 13.5), 'kwargs': {}},
        'sap': {'args': (141, 18), 'kwargs': {}},
        'spo2': {'args': (97, 1.5), 'kwargs': {'hard_max': 100}},
        't': {'args': (36.8, 0.5), 'kwargs': {}}
    }

    measurements = { 
        'pulse': pulse, 
        'spo2': spo2, 
        'sap': sap, 
        't': t, 
        'rr': rr, 
        'gluc': gluc, 
        'age': age, 
        'sex': sex, 
        'chest_pain': chest_pain, 
        'motor_impairment': motor_impairment, 
        'diagnosis': diagnoses_map['normal'] 
    }

    for key in limits.keys():
        slack = (limits[key]['max'] - limits[key]['min']) * limits[key]['slack_percent']
        # if not within limits, reroll
        while measurements[key] - slack > limits[key]['max'] or measurements[key] + slack < limits[key]['min']:
            measurements[key] = random_normal(*value_params[key]['args'], **value_params[key]['kwargs'])

    # remove some values to simulate missing measurements
    final_values = {}
    for key in measurements.keys():
        # 10% chance per value to be missing
        remove = random_bool(0.9)
        if not bool(remove):
            final_values[key] = measurements[key]
    
    # diagnosis cannot be missing
    final_values['diagnosis'] = measurements['diagnosis']

    return final_values

# then take some and move the values accordingly to spec (with variance and skip some cols)
def generate_person_diagnosis():
    person = generate_person_healthy()
    diagnosis = random.randint(1, len(diagnoses_params))

    params = diagnoses_params[diagnosis]
    for key in params.keys():
        if key not in person:
            continue

        # 5% chance to skip this symptom altogether
        if not random_bool(0.05):
            continue

        val = params[key]
        if isinstance(val, list):
            index = random.randrange(0, len(val))
            val = val[index]

        if key in categorical:
            person[key] = val
        else:
            # move the value to desired direction for diagnosis
            # add a uniform 20% spread to each direction from the initial value
            baseline = limits[key]['max'] if val > 0 else limits[key]['min']
            person[key] = baseline + person[key] * val + person[key] * val * random.uniform(-0.2, 0.2)
    
    # 0.5% chance to have symptoms but no diagnosis
    person['diagnosis'] = diagnosis * random_bool(0.005)
    return person

def generate_people2(n: int = 10000, p: float = 0.5) -> pd.DataFrame:
    """ 
    Generate an n number of people and return their data in a dataframe
    p is the (approximate) percentage of people to have a diagnosis
    """
    diagnosis_n = math.floor(n * p)
    healthy_n = n - diagnosis_n
    people_dict = [generate_person_healthy() for _ in range(healthy_n)] + [generate_person_diagnosis() for _ in range(diagnosis_n)]
    df = pd.DataFrame(data=people_dict)
    df = df[classfier_col_order]
    return df

if __name__ == '__main__':
    generated_people = generate_people2()
    print(generated_people)
