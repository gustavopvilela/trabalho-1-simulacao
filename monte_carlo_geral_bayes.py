import random

# monte_carlo_geral_bayes (N, P(A), P(E|A), P(E|A'))
def monte_carlo_geral_bayes(n, p_a, p_e_a, p_e_a_l):
    cont_evidencia  = 0
    cont_evidencia_e_a_priori = 0
    for i in range(n):
        evento_a_ocorreu = random.random() < p_a
        evidencia_ocorreu = False

        if evento_a_ocorreu:
            if random.random() < p_e_a:
                evidencia_ocorreu = True
        else:
            if random.random() < p_e_a_l:
                evidencia_ocorreu = True

        if evidencia_ocorreu:
            cont_evidencia += 1
            if evento_a_ocorreu:
                cont_evidencia_e_a_priori += 1

    return (cont_evidencia_e_a_priori / cont_evidencia) if cont_evidencia > 0 else 0.0



# Resoluções dos exercícios
# EX1 - Bayes: 0.481203 | Monte Carlo: 0.480452
ex1_mc = monte_carlo_geral_bayes(1000000, 0.08, 0.96, 0.09)
ex1_bayes = (0.96 * 0.08) / 0.1596

# EX2 - Bayes: 0.332215 | Monte Carlo: 0.329004
ex2_mc = monte_carlo_geral_bayes(1000000, 0.005, 0.99, 0.01)
ex2_bayes = (0.99 * 0.005) / 0.0149

# EX3B - Bayes: 0.473684 | Monte Carlo: 0.477773
ex3b_mc = monte_carlo_geral_bayes(1000000, 0.60, 0.03, 0.05)
ex3b_bayes = (0.03 * 0.60) / 0.038

# EX8 - Bayes: 0.981818 | Monte Carlo: 0.981861
ex8_mc = monte_carlo_geral_bayes(1000000, 0.75, 0.90, 0.05)
ex8_bayes = (0.90 * 0.75) / 0.6875

# EX9 - Bayes: 0.585366 | Monte Carlo: 0.587893
ex9_mc = monte_carlo_geral_bayes(1000000, 0.15, 0.40, 0.05)
ex9_bayes = (0.40 * 0.15) / 0.1025

# EX10 - Bayes: 0.286741 | Monte Carlo: 0.281854
ex10_mc = monte_carlo_geral_bayes(1000000, 0.008, 0.997, 0.02)
ex10_bayes = (0.997 * 0.008) / 0.027816

resultados = [
    ("1", ex1_mc, ex1_bayes),
    ("2", ex2_mc, ex2_bayes),
    ("3b", ex3b_mc, ex3b_bayes),
    ("8", ex8_mc, ex8_bayes),
    ("9", ex9_mc, ex9_bayes),
    ("10", ex10_mc, ex10_bayes),
]

print(f"{'Exercício':<12} | {'Monte Carlo':<15} | {'Bayes':<15} | {'Gap (%)':<15}")
print("-" * 65)

for ex, mc, bayes in resultados:
    gap = (abs(mc - bayes) / bayes) * 100
    print(f"{ex:<12} | {mc:<15.6f} | {bayes:<15.6f} | {gap:<15.4f}")