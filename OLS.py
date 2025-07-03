import queue as Q
import numpy as np
from Environment import Environment
from Learning import q_learning


def SolveSOMDP(env, weight_vector, algorithm):
    if algorithm == 'q_learning':
        _, v, _ = q_learning(env, weight_vector)
    else:
        print("WRONG ALGORITHM!!!")

    return v


def new_weight(v, S):

    x = -1
    for v_prima in S:
        if v[0]==v_prima[0] and v[1]==v_prima[1]:
            pass
        else:
            new_x = (v[0] - v_prima[0]) / (v[0] - v[1] - v_prima[0] + v_prima[1])
        if new_x >= x:
            x = new_x

    weight_vector = [1-x, x]
    print("The new weight is: ", weight_vector)
    return weight_vector


def OLS(env):
    """
    Computes Optimistic Linear Support to compute the convex hull of an MOMDP. Furthermore, it returns it as a list
    sorted by the second component (the ethical component). Thus, the first policy of the hull is the least ethical
    and so on.

    Adapted to the needs of our paper so that it does not compute the whole hull: it only tries to find the second-best
    value vector and the ethical-optimal value vector.

    :param env: the MOMDP
    :return: the convex hull
    """
    S = []
    W = []

    q = Q.PriorityQueue()

    q.put((-9999, [1.0, 0.0]))
    q.put((-9999, [0.01, 0.99]))

    while not q.empty():
        print("----------------------------")
        weight_vector = q.get()[1]
        print(weight_vector)

        v = SolveSOMDP(env, weight_vector, 'q_learning')

        print("The Learnt Policy has the following Value:")
        policy_value = v[10,11,8]
        print("Individual Value V_0 = " + str(round(policy_value[0],2)))
        print("Ethical Value (V_N + V_E) = " + str(round(policy_value[1],2)))

        W.append(weight_vector)
        w = new_weight(policy_value, S)

        S.append((policy_value[0], policy_value[1]))

        if w[1] != weight_vector[1] and w[1] > 0:
            q.put((-999, w))
            print(list(q.queue))

        print(S)

    ch = list(dict.fromkeys(S))
    sorted_ch = sorted(ch, key=lambda x:x[1])
    return sorted_ch

if __name__ == "__main__":
    env = Environment(is_deterministic=True)
    convex_hull = OLS(env)

    ethical_optimal = convex_hull[-1]
    second_best = convex_hull[-2]

    print("Ethical optimal : ", ethical_optimal)
    print("Second best : ", second_best)