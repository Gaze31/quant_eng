import numpy as np

def binomial_crr(S0, K, r, sigma, T, steps=100, option_type="call",
                 american=False, q=0.0):
    """
    Cox-Ross-Rubinstein binomial tree for European and American options.

    Parameters
    ----------
    S0 : float
        Spot price
    K : float
        Strike price
    r : float
        Risk-free rate (annual, continuous)
    sigma : float
        Volatility (annual)
    T : float
        Time to maturity (years)
    steps : int
        Number of binomial steps
    option_type : {"call","put"}
    american : bool
        If True, price an American option (allow early exercise)
    q : float
        Continuous dividend yield (annual)

    Returns
    -------
    price : float
        Option price at t=0
    delta : float
        Approximate Delta using first-step finite difference (None if steps<1)
    """

    # Validate
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    dt = T / steps
    # up and down factors (CRR)
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    # risk-neutral probability with continuous dividend yield
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)

    if not (0 <= p <= 1):
        # numerical issues for extreme parameters; fall back to bounded p
        p = min(max(p, 0.0), 1.0)

    # stock prices at maturity (vectorized)
    # S_T[j] = S0 * u**j * d**(steps-j) for j=0..steps
    j = np.arange(0, steps + 1)
    ST = S0 * (u ** j) * (d ** (steps - j))

    # payoff at maturity
    if option_type == "call":
        values = np.maximum(ST - K, 0.0)
    else:
        values = np.maximum(K - ST, 0.0)

    # step backwards
    for i in range(steps - 1, -1, -1):
        # discount expected value
        values = disc * (p * values[1:i + 2] + (1 - p) * values[0:i + 1])

        if american:
            # compute underlying prices at this node level
            j = np.arange(0, i + 1)
            S_t = S0 * (u ** j) * (d ** (i - j))
            if option_type == "call":
                exercise = np.maximum(S_t - K, 0.0)
            else:
                exercise = np.maximum(K - S_t, 0.0)
            # early exercise decision
            values = np.maximum(values, exercise)

    price = float(values[0])

    # approximate delta using first step (if steps >= 1)
    if steps >= 1:
        # stock up and down at first step
        S_up = S0 * u
        S_dn = S0 * d
        # option values at first step (we can compute one backward iteration)
        # Build values at level 1:
        # level 1 payoffs (for European/american consistency we compute continuation)
        # Use values currently representing level 0 after full backward induction; recompute level-1 quickly:
        # We'll perform one backward step from maturity to level 1
        j = np.arange(0, steps)
        ST1 = S0 * (u ** j) * (d ** (steps - 1 - j))
        # payoff at level1 nodes
        if option_type == "call":
            val_level1 = np.maximum(ST1 * u - K, 0.0) * 0  # placeholder not used
        # Simpler: do a small separate tree of 1 step to get children values at t=dt:
        # Price children by going from maturity but that's expensive; instead do:
        # Recompute values at level 1 by one backward iteration from terminal payoffs:
        # Build temp array for level = 1
        temp = np.maximum(ST - K, 0.0) if option_type == "call" else np.maximum(K - ST, 0.0)
        # Back up to level 1:
        for i in range(steps - 1, 0, -1):
            temp = disc * (p * temp[1:i + 2] + (1 - p) * temp[0:i + 1])
        val_up = float(temp[1])   # node with one up (S0*u)
        val_dn = float(temp[0])   # node with one down (S0*d)
        delta = (val_up - val_dn) / (S_up - S_dn)
    else:
        delta = None

    return price, delta


# ========================= Example Usage =========================
if __name__ == "__main__":
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0

    price_eur_call, delta = binomial_crr(S0, K, r, sigma, T, steps=200, option_type="call", american=False)
    price_am_put, _ = binomial_crr(S0, K, r, sigma, T, steps=200, option_type="put", american=True)

    print(f"European Call (CRR, 200 steps): {price_eur_call:.6f}")
    print(f"Delta estimate (1-step finite diff): {delta:.6f}")
    print(f"American Put (CRR, 200 steps): {price_am_put:.6f}")
