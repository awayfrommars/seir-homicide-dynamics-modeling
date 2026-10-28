"""
Calibração do modelo SEIR para criminalidade violenta em Sergipe

Dados: Atlas da Violência (2024, 2026)
       Taxa de homicídios de homens jovens (15-29 anos) por 100 mil hab.
       Sergipe, 2012-2024

Deriva analítica completa (equilíbrios, R0, estabilidade via Routh-Hurwitz)
em report/SEIR.pdf. Este script cobre a etapa de validação computacional:
calibração dos parâmetros livres (beta, sigma, gamma, E0) contra os dados
observados, para três cenários do parâmetro fixo delta.
"""

import csv
import os

import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# 1. DADOS OBSERVADOS
# ═══════════════════════════════════════════════════════════════════════

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "homicidios_sergipe.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def carregar_dados(path=DATA_PATH):
    anos, taxas = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            anos.append(int(row["ano"]))
            taxas.append(float(row["taxa_homicidios_por_100k"]))
    return np.array(anos), np.array(taxas)


ANOS, H_OBS = carregar_dados()

# ═══════════════════════════════════════════════════════════════════════
# 2. PARÂMETROS FIXADOS EXTERNAMENTE
# ═══════════════════════════════════════════════════════════════════════

N = 100_000.0      # população normalizada (padrão Atlas da Violência)
MU = 1.0 / 15.0     # saída natural: duração da coorte 15-29 anos (anos⁻¹)
LAMB = MU * N       # entrada: Lambda = mu*N -> dN/dt = 0
KAPPA = 1.0         # proxy: 1 homicídio de vítima por infrator ativo/ano
H = 0.1             # passo RK4 (anos)

# ═══════════════════════════════════════════════════════════════════════
# 3. MODELO E INTEGRAÇÃO RK4
# ═══════════════════════════════════════════════════════════════════════


def seir_rhs(S, E, I, R, beta, sigma, gamma, delta):
    """Campo vetorial do sistema SEIR adaptado (floats puros, sem overhead numpy)."""
    dS = LAMB - beta * S * I / N - MU * S
    dE = beta * S * I / N - (sigma + MU) * E
    dI = sigma * E - (gamma + delta + MU) * I
    dR = (gamma + delta) * I - MU * R
    return dS, dE, dI, dR


def rk4_simular(beta, sigma, gamma, delta, E0):
    """
    Integra o sistema SEIR por RK4 com passo H = 0.1 ano.
    Condição inicial: t0 = 2012.
    Retorna I(t) nos 13 pontos anuais de 2012 a 2024.
    """
    I0 = H_OBS[0] / KAPPA
    S0 = N - I0 - E0
    if S0 <= 0:
        return np.full(len(ANOS), 1e9)

    S, E, I, R = S0, E0, I0, 0.0
    steps_por_ano = int(round(1.0 / H))
    I_saida = [I]

    for _ in range(len(ANOS) - 1):
        for _ in range(steps_por_ano):
            k1 = seir_rhs(S, E, I, R, beta, sigma, gamma, delta)
            k2 = seir_rhs(S + H / 2 * k1[0], E + H / 2 * k1[1], I + H / 2 * k1[2], R + H / 2 * k1[3],
                          beta, sigma, gamma, delta)
            k3 = seir_rhs(S + H / 2 * k2[0], E + H / 2 * k2[1], I + H / 2 * k2[2], R + H / 2 * k2[3],
                          beta, sigma, gamma, delta)
            k4 = seir_rhs(S + H * k3[0], E + H * k3[1], I + H * k3[2], R + H * k3[3],
                          beta, sigma, gamma, delta)
            S += H / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            E += H / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            I += H / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            R += H / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
            if S < 0:
                return np.full(len(ANOS), 1e9)
        I_saida.append(I)

    return np.array(I_saida)


# ═══════════════════════════════════════════════════════════════════════
# 4. CALIBRAÇÃO
# ═══════════════════════════════════════════════════════════════════════


def funcao_objetivo(params, delta):
    """Soma dos erros quadráticos entre H_sim(t) e H_obs(t)."""
    beta, sigma, gamma, E0 = params
    I_sim = rk4_simular(beta, sigma, gamma, delta, E0)
    return np.sum((KAPPA * I_sim - H_OBS) ** 2)


def calibrar(delta, seed=42, verbose=True):
    """
    Estima theta = (beta, sigma, gamma, E0) por:
      1) Busca global via Differential Evolution
      2) Refinamento local via Nelder-Mead
    """
    bounds = [
        (0.01, 20.0),   # beta  - taxa de influência social
        (0.01, 5.0),    # sigma - progressão E -> I
        (0.01, 5.0),    # gamma - ressocialização
        (1.0, 3e4),     # E0    - expostos em t0 = 2012
    ]

    if verbose:
        print("  Busca global...", end=" ", flush=True)

    res_global = differential_evolution(
        funcao_objetivo, bounds, args=(delta,),
        seed=seed, maxiter=600, tol=1e-9,
        mutation=(0.5, 1.5), recombination=0.9,
        popsize=18, workers=1,
    )

    if verbose:
        print(f"SSE = {res_global.fun:,.2f}  |  Refinamento...", end=" ", flush=True)

    res_local = minimize(
        funcao_objetivo, res_global.x, args=(delta,),
        method="Nelder-Mead", bounds=bounds,
        options={"maxiter": 20_000, "xatol": 1e-10, "fatol": 1e-10},
    )

    beta, sigma, gamma, E0 = res_local.x

    # Verificação de sanidade: parâmetros devem permanecer fisicamente válidos
    for nome_p, val, (lo, hi) in zip(["beta", "sigma", "gamma", "E0"], res_local.x, bounds):
        if not (lo - 1e-6 <= val <= hi + 1e-6):
            print(f"\n  [AVISO] {nome_p} = {val:.4f} fora dos limites [{lo}, {hi}] após refinamento!")

    R0 = (beta * sigma) / ((sigma + MU) * (gamma + delta + MU))

    if verbose:
        print(f"SSE = {res_local.fun:,.2f}  |  R0 = {R0:.4f}")

    return {"beta": beta, "sigma": sigma, "gamma": gamma,
            "E0": E0, "R0": R0, "sse": res_local.fun}


def calcular_metricas(p, delta):
    """Computa R² e RMSE do ajuste."""
    I_sim = rk4_simular(p["beta"], p["sigma"], p["gamma"], delta, p["E0"])
    H_sim = KAPPA * I_sim
    ss_res = np.sum((H_OBS - H_sim) ** 2)
    ss_tot = np.sum((H_OBS - H_OBS.mean()) ** 2)
    R2 = 1 - ss_res / ss_tot
    RMSE = np.sqrt(ss_res / len(H_OBS))
    return R2, RMSE, H_sim


# ═══════════════════════════════════════════════════════════════════════
# 5. EXECUÇÃO - TRÊS CENÁRIOS DE DELTA
# ═══════════════════════════════════════════════════════════════════════

CENARIOS = {
    "Baixo   (delta=0.10)": 0.10,
    "Central (delta=0.15)": 0.15,
    "Alto    (delta=0.20)": 0.20,
}

CORES = {
    "Baixo   (delta=0.10)": "#1976D2",
    "Central (delta=0.15)": "#E53935",
    "Alto    (delta=0.20)": "#388E3C",
}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Calibração SEIR - Criminalidade violenta em Sergipe")
    print("  Homens jovens (15-29 anos) | Atlas da Violência")
    print("=" * 60)

    resultados = {}
    H_sims = {}
    metricas_d = {}

    for nome, delta_val in CENARIOS.items():
        print(f"\nCenário {nome} :")
        resultados[nome] = calibrar(delta_val)
        R2, RMSE, H_sim = calcular_metricas(resultados[nome], delta_val)
        metricas_d[nome] = (R2, RMSE)
        H_sims[nome] = H_sim

    # ─── Relatório cenário central ──────────────────────────────────────
    NOME_C = "Central (delta=0.15)"
    P_C = resultados[NOME_C]
    R2_C, RMSE_C = metricas_d[NOME_C]

    print(f"\n{'=' * 60}")
    print("  RESULTADOS - CENÁRIO CENTRAL  (delta = 0.15 ano⁻¹)")
    print(f"{'=' * 60}")
    print(f"  beta  = {P_C['beta']:.6f}  ano⁻¹")
    print(f"  sigma = {P_C['sigma']:.6f}  ano⁻¹")
    print(f"  gamma = {P_C['gamma']:.6f}  ano⁻¹")
    print(f"  E0    = {P_C['E0']:.2f}  (expostos em 2012)")
    print(f"  R0    = {P_C['R0']:.6f}")
    print(f"  R2    = {R2_C:.6f}")
    print(f"  RMSE  = {RMSE_C:.4f}  por 100 mil hab.")
    print(f"{'=' * 60}")

    print(f"\n{'-' * 60}")
    print("  ANÁLISE DE SENSIBILIDADE - R0 por cenário de delta")
    print(f"{'-' * 60}")
    for nome, p in resultados.items():
        R2, RMSE = metricas_d[nome]
        print(f"  {nome}:  R0 = {p['R0']:.4f}   R2 = {R2:.4f}   RMSE = {RMSE:.2f}")
    print(f"{'-' * 60}")

    print(f"\n{'-' * 60}")
    print("  PARÂMETROS COMPLETOS - TODOS OS CENÁRIOS (diagnóstico)")
    print(f"{'-' * 60}")
    bounds_check = {"beta": 0.01, "sigma": 0.01, "gamma": 0.01}
    for nome, p in resultados.items():
        print(f"  {nome}:")
        print(f"     beta={p['beta']:.5f}  sigma={p['sigma']:.5f}  gamma={p['gamma']:.5f}  "
              f"E0={p['E0']:.1f}  SSE={p['sse']:.1f}")
        colados = [k for k, lo in bounds_check.items() if abs(p[k] - lo) < 1e-3]
        if colados:
            print(f"     [AVISO] colados no limite inferior: {colados}")
    print(f"{'-' * 60}")

    # ═══════════════════════════════════════════════════════════════════
    # 6. VISUALIZAÇÃO
    # ═══════════════════════════════════════════════════════════════════

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(ANOS, H_OBS, color="black", zorder=5, s=60,
                label="Observado (Atlas da Violência)", marker="o")
    ax1.plot(ANOS, H_sims[NOME_C], color=CORES[NOME_C], linewidth=2.5,
             label=f"Simulado - cenário central  (R0 = {P_C['R0']:.3f})")
    ax1.fill_between(ANOS, H_OBS, H_sims[NOME_C], alpha=0.10, color=CORES[NOME_C])
    ax1.axvline(x=2016, color="gray", linestyle=":", linewidth=1.2,
                label="Pico observado (2016 | H = 297.9)")
    ax1.set_title("Ajuste do modelo SEIR - Sergipe, homens jovens 15-29 anos",
                   fontsize=13, fontweight="bold")
    ax1.set_xlabel("Ano", fontsize=11)
    ax1.set_ylabel("Homicídios por 100 mil hab.", fontsize=11)
    ax1.set_xticks(ANOS)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, alpha=0.25)
    ax1.text(0.015, 0.95, f"R2 = {R2_C:.4f}     RMSE = {RMSE_C:.2f} por 100 mil hab.",
              transform=ax1.transAxes, fontsize=10, verticalalignment="top",
              bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray", alpha=0.8))

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(ANOS, H_OBS, color="black", zorder=5, s=35, label="Observado")
    for nome, H_sim in H_sims.items():
        ax2.plot(ANOS, H_sim, color=CORES[nome], linewidth=2,
                  label=f"{nome.strip()}  R0={resultados[nome]['R0']:.3f}")
    ax2.set_title("Sensibilidade a delta - curvas simuladas", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Ano", fontsize=10)
    ax2.set_ylabel("Homicídios por 100 mil hab.", fontsize=10)
    ax2.set_xticks(ANOS)
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(gs[1, 1])
    nomes_c = list(CENARIOS.keys())
    R0_vals = [resultados[n]["R0"] for n in nomes_c]
    bar_c = [CORES[n] for n in nomes_c]
    bars = ax3.bar(range(len(nomes_c)), R0_vals, color=bar_c, width=0.5, alpha=0.85, edgecolor="gray")
    ax3.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, label="Limiar R0 = 1")
    for bar, val in zip(bars, R0_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                  f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax3.set_title("R0 empírico por cenário de delta", fontsize=11, fontweight="bold")
    ax3.set_ylabel("R0", fontsize=10)
    ax3.set_xticks(range(len(nomes_c)))
    ax3.set_xticklabels([n.strip() for n in nomes_c], fontsize=8)
    ax3.set_ylim(0, max(R0_vals) * 1.25)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.25, axis="y")

    plt.suptitle("Validação - Modelo SEIR para Criminalidade Violenta em Sergipe\n"
                  "Fonte: Atlas da Violência (edições 2024 e 2026)",
                  fontsize=12, fontweight="bold")

    out_path = os.path.join(RESULTS_DIR, "seir_validacao_sergipe.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigura salva: {out_path}")


if __name__ == "__main__":
    main()
