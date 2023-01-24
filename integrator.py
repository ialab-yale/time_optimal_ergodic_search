def rk4(f, x, u, dt):
    k1 = f(x, u) * dt
    k2 = f(x + k1/2.0, u) * dt
    k3 = f(x + k2/2.0, u) * dt
    k4 = f(x + k3, u) * dt
    return x + (k1 + 2.0 * (k2 + k3) + k4)/6.0

def euler(f, x, u, dt):
    return x + f(x, u) * dt