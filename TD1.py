import numpy as np

int_to_char = {
    0 : 'u',
    1 : 'r',
    2 : 'd',
    3 : 'l'
}


policy_one_step_look_ahead = {
    0 : [-1,0],
    1 : [0,1],
    2 : [1,0],
    3 : [0,-1]
}

def policy_int_to_char(pi,n):

    pi_char = ['']

    for i in range(n):
        for j in range(n):

            if i == 0 and j == 0 or i == n-1 and j == n-1:

                continue

            pi_char.append(int_to_char[pi[i,j]])

    pi_char.append('')

    return np.asarray(pi_char).reshape(n,n)

def policy_evaluation(n,pi,v,Gamma,threshhold):

    while True:
        diff_max = 0
        new_v = v.copy()

        for i in range(n):
            for j in range(n):
        
              if (i == 0 and j == 0) or (i == n-1 and j == n-1):
                  new_v[i, j] = 0
                  continue
          
              a = pi[i, j]
              di, dj = policy_one_step_look_ahead[a]

              ni = i + di
              nj = j + dj
              if ni < 0 or ni >= n or nj < 0 or nj >= n:
                  ni, nj = i, j

              reward = -1

              new_v[i, j] = reward + Gamma * v[ni, nj]

              diff = abs(new_v[i, j] - v[i, j])
              if diff > diff_max:
                  diff_max = diff

        v = new_v

        if diff_max < threshhold:
            break
    return v


def policy_improvement(n,pi,v,Gamma):

    new_pi = pi.copy()
    stable = True

    for i in range(n):
        for j in range(n):

            if (i == 0 and j == 0) or (i == n-1 and j == n-1):
                continue

            best_a = None
            best_val = -999999

            for a in range(4):
                di, dj = policy_one_step_look_ahead[a]
                ni, nj = i + di, j + dj

                if ni < 0 or ni >= n or nj < 0 or nj >= n:
                    ni, nj = i, j

                r = -1
                val = r + Gamma * v[ni, nj]

                if val > best_val:
                    best_val = val
                    best_a = a

            if best_a != pi[i, j]:
                stable = False

            new_pi[i, j] = best_a

    return new_pi, stable


def policy_initialization(n):

    pi = np.random.randint(0, 4, (n, n))

    pi[0, 0] = 0
    pi[n-1, n-1] = 0

    return pi

def policy_iteration(n,Gamma,threshhold):

    pi = policy_initialization(n=n)

    v = np.zeros(shape=(n,n))

    while True:

        v = policy_evaluation(n=n,v=v,pi=pi,threshhold=threshhold,Gamma=Gamma)

        pi , pi_stable = policy_improvement(n=n,pi=pi,v=v,Gamma=Gamma)

        if pi_stable:

            break

    return pi , v


n = 4

Gamma = [0.8,0.9,0.95,0.99]

threshhold = 1e-4

for _gamma in Gamma:

    pi, v = policy_iteration(n=n,Gamma=_gamma,threshhold=threshhold)

    pi_char = policy_int_to_char(n=n,pi=pi)

    print()
    print("Gamma = ",_gamma)

    print()

    print(pi_char)

    print()
    print()

    print(v)
