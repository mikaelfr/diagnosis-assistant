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
    return { 'pulse': pulse, 'spo2': spo2, 'sap': sap, 't': t, 'rr': rr, 'gluc': gluc, 'age': age, 'sex': sex }

def generate_people(n: int = 10000) -> pd.DataFrame:
    """ Generate an n number of people and return their data in a dataframe """
    people_dict = [generate_healthy_person() for _ in range(n)]
    return pd.DataFrame(data=people_dict)

if __name__ == '__main__':
    generated_people = generate_people()
    print(generated_people)
