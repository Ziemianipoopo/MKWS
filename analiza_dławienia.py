"""
==============================================================================
ANALIZA DŁAWIENIA SILNIKA RAKIETOWEGO H2/H2O2 98%
Zakres ciągu: F = 100 – 800 N  (zmienny strumień masy)
==============================================================================
GEOMETRIA STAŁA (wyznaczona w punkcie projektowym):
  F_des = 701.8 N,  ṁ_des = 0.2 kg/s,  MR_des = 7.94,  Pe = Pa

  At = A* – przekrój gardła (niezmienny)
  Ae        – przekrój wylotu (niezmienny)
  ε_e = Ae/At – stopień ekspansji (niezmienny)

METODOLOGIA:
  Dla każdego strumienia masy ṁ, przy stałej geometrii (At, Ae, ε_e):
  1. Wyznacz optymalny stosunek mieszanki MR* (min. komp. ekwiwalencji φ*)
     maksymalizujący siłę ciągu F = ṁ·ve + Ae·(Pe − Pa)
  2. Ciśnienie komory Pc wynika z warunku krytycznego gardła:
       ṁ = ρt · vt · At = Pc · At / C*(Pc, MR)
     → iteracyjne wyznaczenie Pc dla danego (ṁ, MR)
  3. Liczba Macha na wylocie Me z równania obszarów A/A*(Me) = ε_e
     (gałąź naddźwiękowa – geometria dyszy niezmienna)
  4. Ciśnienie Pe i prędkość ve na wylocie (isentropowe)
  5. Weryfikacja: przy ṁ ≠ ṁ_des → Pe ≠ Pa (praca poza-projektowa)
     Silnik może pracować w trybie:
       · niedorozprężonym (Pe > Pa) – Pe/Pa > 1
       · nadrozprężonym  (Pe < Pa) – Pe/Pa < 1  → możliwe oderwanie strumienia

WYKRESY:
  1. F(ṁ)    – ciąg vs strumień masy
  2. Pc(ṁ)   – ciśnienie komory vs strumień masy
  3. φ*(ṁ)   – optymalny współczynnik ekwiwalencji vs strumień masy
  4. Pe/Pa(ṁ) – stan ekspansji dyszy
==============================================================================
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from rocketcea.cea_obj import CEA_Obj
from rocketcea.blends import newOxBlend


# ══════════════════════════════════════════════════════════════════════════════
# 0. STAŁE I DANE WEJŚCIOWE
# ══════════════════════════════════════════════════════════════════════════════
R_UNIV     = 8314.46
G0         = 9.80665
PSI_PER_PA = 1 / 6894.76
PA         = 101325.0       # [Pa]
MR_STOICH  = 17.0           # stechiometria H2 / H2O2

# Punkt projektowy (wyznaczony w silnik_rakietowy_H2_H2O2.py)
F_DES   = 701.8             # [N]
MDOT_DES = 0.200            # [kg/s]
MR_DES  = 7.94              # O/F w punkcie projektowym
EPS_C   = 8.0               # Ac/At


# ══════════════════════════════════════════════════════════════════════════════
# 1. INICJALIZACJA ROCKETCEA
# ══════════════════════════════════════════════════════════════════════════════
ox_blend = newOxBlend(oxL=['H2O2', 'H2O'], oxPcentL=[98, 2])
cea = CEA_Obj(fuelName='H2', oxName=ox_blend)


# ══════════════════════════════════════════════════════════════════════════════
# 2. FUNKCJE POMOCNICZE
# ══════════════════════════════════════════════════════════════════════════════

def get_cea_params(Pc_bar, MR):
    Pc_psia = Pc_bar * 1e5 * PSI_PER_PA
    Tc      = cea.get_Tcomb(Pc=Pc_psia, MR=MR)
    Mw, gamma = cea.get_Chamber_MolWt_gamma(Pc=Pc_psia, MR=MR)
    return Tc, Mw, gamma, R_UNIV / Mw


def Cstar_from_params(gamma, R_sp, Tc):
    return (np.sqrt(R_sp * Tc / gamma)
            * ((gamma + 1) / 2) ** ((gamma + 1) / (2 * (gamma - 1))))


def area_ratio(M, gamma):
    f = (2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * M**2)
    return (1 / M) * f ** ((gamma + 1) / (2 * (gamma - 1)))


def find_Pc_from_mdot(m_dot, MR, At,
                      Pc_lo=0.3, Pc_hi=60.0):
    """
    Wyznacza Pc [bar] z warunku krytycznego gardła:
      ṁ = Pc · At / C*(Pc, MR)
    """
    def resid(Pc_bar):
        Tc, _, gamma, R_sp = get_cea_params(Pc_bar, MR)
        Cst = Cstar_from_params(gamma, R_sp, Tc)
        return Pc_bar * 1e5 * At / Cst - m_dot

    # sprawdź granice
    if resid(Pc_lo) * resid(Pc_hi) >= 0:
        # rozszerz przedział
        while resid(Pc_lo) > 0 and Pc_lo > 1e-3:
            Pc_lo /= 2
        while resid(Pc_hi) < 0 and Pc_hi < 200:
            Pc_hi *= 2
    return brentq(resid, Pc_lo, Pc_hi, xtol=1e-5)


def compute_thrust_fixed_geom(m_dot, MR, At, Ae, eps_e):
    """
    Oblicza ciąg i parametry silnika dla:
      - stałej geometrii (At, Ae, ε_e)
      - zadanego strumienia masy m_dot
      - zadanego stosunku mieszanki MR

    Zwraca: (F, Pc_bar, phi, Pe_Pa, ve, Me, Tc, Mw, gamma)
    """
    try:
        # 1. Pc z warunku krytycznego gardła
        Pc_bar = find_Pc_from_mdot(m_dot, MR, At)
        Pc_Pa  = Pc_bar * 1e5

        # 2. Parametry termodynamiczne
        Tc, Mw, gamma, R_sp = get_cea_params(Pc_bar, MR)
        rho_c = Pc_Pa / (R_sp * Tc)

        # 3. Gardło (M = 1)
        Tt    = Tc * 2 / (gamma + 1)
        rho_t = rho_c * (2 / (gamma + 1)) ** (1 / (gamma - 1))
        vt    = np.sqrt(gamma * R_sp * Tt)

        # 4. Wylot – Me z równania obszarów (gałąź naddźwiękowa, M > 1)
        # eps_e = Ae/At = area_ratio(Me, gamma)
        try:
            Me = brentq(lambda M: area_ratio(M, gamma) - eps_e,
                        1.0 + 1e-6, 15.0, xtol=1e-6)
        except ValueError:
            return None

        # 5. Pe, Te, ve na wylocie (isentropowe)
        Pe = Pc_Pa * (1 + (gamma - 1) / 2 * Me**2) ** (-(gamma / (gamma - 1)))
        Te = Tc / (1 + (gamma - 1) / 2 * Me**2)
        ve = Me * np.sqrt(gamma * R_sp * Te)

        # 6. Ciąg
        F = m_dot * ve + Ae * (Pe - PA)

        phi = MR_STOICH / MR
        return F, Pc_bar, phi, Pe / PA, ve, Me, Tc, Mw, gamma

    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 3. WYZNACZENIE GEOMETRII Z PUNKTU PROJEKTOWEGO
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 65)
print("WYZNACZANIE GEOMETRII PUNKTU PROJEKTOWEGO")
print("─" * 65)

# Punkt projektowy: Pe = Pa (zał.), ve = F/ṁ, iteracja na Pc
def ve_isen_design(Pc_bar, MR):
    if Pc_bar * 1e5 <= PA:
        return 0.0
    Tc, _, gamma, R_sp = get_cea_params(Pc_bar, MR)
    Pe_Pc = PA / (Pc_bar * 1e5)
    return np.sqrt(2 * gamma / (gamma - 1) * R_sp * Tc
                   * (1 - Pe_Pc ** ((gamma - 1) / gamma)))

ve_des    = F_DES / MDOT_DES
Pc_bar_des = brentq(lambda p: ve_isen_design(p, MR_DES) - ve_des, 5.0, 20.0, xtol=1e-5)
Pc_des    = Pc_bar_des * 1e5
Tc_des, Mw_des, gamma_des, R_sp_des = get_cea_params(Pc_bar_des, MR_DES)

rho_c_des = Pc_des / (R_sp_des * Tc_des)
Tt_des    = Tc_des * 2 / (gamma_des + 1)
rho_t_des = rho_c_des * (2 / (gamma_des + 1)) ** (1 / (gamma_des - 1))
vt_des    = np.sqrt(gamma_des * R_sp_des * Tt_des)

At  = MDOT_DES / (rho_t_des * vt_des)       # [m²] – stałe!
Dt  = np.sqrt(4 * At / np.pi)               # [m]

Te_des  = Tc_des * (PA / Pc_des) ** ((gamma_des - 1) / gamma_des)
Me_des  = np.sqrt(2 / (gamma_des - 1) * (Tc_des / Te_des - 1))
eps_e   = area_ratio(Me_des, gamma_des)      # Ae/At – stałe!
Ae      = eps_e * At                         # [m²] – stałe!
De      = np.sqrt(4 * Ae / np.pi)           # [m]

print(f"  Pc_des  = {Pc_bar_des:.4f} bar")
print(f"  Tc_des  = {Tc_des:.2f} K")
print(f"  γ_des   = {gamma_des:.4f}")
print(f"  At      = {At*1e6:.4f} mm²   Dt = {Dt*1e3:.4f} mm")
print(f"  Ae      = {Ae*1e6:.4f} mm²   De = {De*1e3:.4f} mm")
print(f"  ε_e = Ae/At = {eps_e:.5f}")
print(f"  ve_des  = {ve_des:.2f} m/s   Isp_des = {ve_des/G0:.2f} s")


# ══════════════════════════════════════════════════════════════════════════════
# 4. SWEEP – ANALIZA DŁAWIENIA
# ══════════════════════════════════════════════════════════════════════════════
# Zakres strumieni: ~100–800 N ≈ ṁ × ve, szacunek: ṁ ∈ [0.025, 0.235] kg/s
m_dot_arr = np.linspace(0.025, 0.235, 70)

MR_LO, MR_HI = 2.5, 18.0   # zakres przeszukiwania MR (φ: 17/18–17/2.5 = 0.94–6.8)

results = []

print("\n" + "─" * 65)
print("SWEEP STRUMIENIA MASY  (stała geometria At, Ae)")
print("─" * 65)
print(f"  {'ṁ [kg/s]':>10}  {'F [N]':>8}  {'Pc [bar]':>10}  {'φ*':>7}  "
      f"{'MR*':>7}  {'Pe/Pa':>8}  {'Isp [s]':>9}")
print(f"  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*9}")

for m_dot in m_dot_arr:
    # Optymalizacja MR → max F przy stałej geometrii i danym m_dot
    def neg_thrust(MR):
        res = compute_thrust_fixed_geom(m_dot, MR, At, Ae, eps_e)
        return -res[0] if res is not None else 0.0

    opt = minimize_scalar(neg_thrust, bounds=(MR_LO, MR_HI),
                          method='bounded', options={'xatol': 1e-3})
    MR_opt = opt.x
    res = compute_thrust_fixed_geom(m_dot, MR_opt, At, Ae, eps_e)
    if res is None:
        continue
    F_opt, Pc_opt, phi_opt, Pe_Pa, ve_opt, Me_opt, Tc_opt, Mw_opt, gamma_opt = res
    Isp_opt = ve_opt / G0  # (uwzględnia tylko impuls prędkościowy, nie ciśnieniowy)
    Isp_eff = F_opt / (m_dot * G0)  # efektywny Isp uwzględniający człon ciśnieniowy

    results.append({
        'm_dot' : m_dot,
        'F'     : F_opt,
        'Pc'    : Pc_opt,
        'phi'   : phi_opt,
        'MR'    : MR_opt,
        'Pe_Pa' : Pe_Pa,
        've'    : ve_opt,
        'Me'    : Me_opt,
        'Tc'    : Tc_opt,
        'Mw'    : Mw_opt,
        'gamma' : gamma_opt,
        'Isp_eff': Isp_eff,
    })

    if len(results) % 7 == 1:  # drukuj co kilka punktów
        print(f"  {m_dot:>10.4f}  {F_opt:>8.2f}  {Pc_opt:>10.4f}  "
              f"{phi_opt:>7.4f}  {MR_opt:>7.4f}  {Pe_Pa:>8.4f}  {Isp_eff:>9.2f}")

# tablice wynikowe
r = results
mdot_r = np.array([x['m_dot'] for x in r])
F_r    = np.array([x['F']     for x in r])
Pc_r   = np.array([x['Pc']    for x in r])
phi_r  = np.array([x['phi']   for x in r])
PePa_r = np.array([x['Pe_Pa'] for x in r])
Tc_r   = np.array([x['Tc']    for x in r])
ve_r   = np.array([x['ve']    for x in r])
Isp_r  = np.array([x['Isp_eff'] for x in r])

# Punkt projektowy z dokładną interpolacją
idx_des = np.argmin(np.abs(mdot_r - MDOT_DES))

print(f"\n  Punkt projektowy (ṁ ≈ {MDOT_DES} kg/s):")
print(f"    F     = {F_r[idx_des]:.2f} N   (zadane: {F_DES} N)")
print(f"    Pc    = {Pc_r[idx_des]:.4f} bar")
print(f"    φ*    = {phi_r[idx_des]:.4f}")
print(f"    Pe/Pa = {PePa_r[idx_des]:.5f}")
print(f"    Isp_eff = {Isp_r[idx_des]:.2f} s")


# ══════════════════════════════════════════════════════════════════════════════
# 5. WYKRESY
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 14))
fig.suptitle('Analiza dławienia silnika rakietowego H₂/H₂O₂ 98%\n'
             f'Stała geometria: At = {At*1e6:.2f} mm²,  '
             f'Ae = {Ae*1e6:.1f} mm²,  ε = {eps_e:.3f}',
             fontsize=13, y=1.002)

gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32)
ax1 = fig.add_subplot(gs[0, 0])   # F(ṁ)
ax2 = fig.add_subplot(gs[0, 1])   # Pc(ṁ)
ax3 = fig.add_subplot(gs[1, 0])   # φ*(ṁ)
ax4 = fig.add_subplot(gs[1, 1])   # Pe/Pa(ṁ) + Isp(ṁ)

BLUE   = 'royalblue'
ORANGE = 'darkorange'
GREEN  = 'seagreen'
RED    = 'crimson'
PURPLE = 'mediumpurple'

# Pionowa linia projektowa (wspólna)
for ax in [ax1, ax2, ax3, ax4]:
    ax.axvline(MDOT_DES, color='gray', ls=':', lw=1.3, zorder=1)

# ── Wykres 1: F(ṁ) ─────────────────────────────────────────────────────────
# Zakres ciągów 100 i 800 N
try:
    m_100 = np.interp(100, F_r, mdot_r)
    m_800 = np.interp(800, F_r, mdot_r)
    ax1.axhline(100, color='lightcoral', ls='--', lw=1.2)
    ax1.axhline(800, color='lightcoral', ls='--', lw=1.2)
    ax1.annotate('100 N', xy=(mdot_r[0], 100), xytext=(mdot_r[0]+0.004, 108),
                 fontsize=8.5, color='tomato')
    ax1.annotate('800 N', xy=(mdot_r[0], 800), xytext=(mdot_r[0]+0.004, 808),
                 fontsize=8.5, color='tomato')
except Exception:
    pass

ax1.plot(mdot_r, F_r, color=BLUE, lw=2.5, label='F(ṁ)  –  optymalny φ*')
ax1.scatter([MDOT_DES], [F_r[idx_des]], color=RED, s=100, zorder=6,
            edgecolors='darkred', lw=1.0, label=f'Punkt proj.  F={F_r[idx_des]:.1f} N')
ax1.set_xlabel('Strumień masy  ṁ  [kg/s]', fontsize=11)
ax1.set_ylabel('Ciąg  F  [N]', fontsize=11)
ax1.set_title('Ciąg vs strumień masy', fontsize=12)
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, ls='--', alpha=0.45)

# Drugorzędna oś X: Isp_eff
ax1b = ax1.twinx()
ax1b.plot(mdot_r, Isp_r, color=PURPLE, lw=1.6, ls='-.', alpha=0.85,
          label='$I_{sp,eff}$ [s]')
ax1b.set_ylabel('Efektywny  $I_{sp}$  [s]', color=PURPLE, fontsize=10)
ax1b.tick_params(axis='y', labelcolor=PURPLE)
ax1b.legend(fontsize=9, loc='lower right')

# ── Wykres 2: Pc(ṁ) ────────────────────────────────────────────────────────
ax2.plot(mdot_r, Pc_r, color=ORANGE, lw=2.5)
ax2.scatter([MDOT_DES], [Pc_r[idx_des]], color=RED, s=100, zorder=6,
            edgecolors='darkred', lw=1.0,
            label=f'Punkt proj.  Pc={Pc_r[idx_des]:.2f} bar')
ax2.set_xlabel('Strumień masy  ṁ  [kg/s]', fontsize=11)
ax2.set_ylabel('Ciśnienie komory  Pc  [bar]', fontsize=11)
ax2.set_title('Ciśnienie komory vs strumień masy', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, ls='--', alpha=0.45)

# Dodaj Tc na drugiej osi
ax2b = ax2.twinx()
ax2b.plot(mdot_r, Tc_r, color='firebrick', lw=1.6, ls='-.', alpha=0.8)
ax2b.set_ylabel('Temperatura komory  Tc  [K]', color='firebrick', fontsize=10)
ax2b.tick_params(axis='y', labelcolor='firebrick')

# ── Wykres 3: φ*(ṁ) ────────────────────────────────────────────────────────
ax3.plot(mdot_r, phi_r, color=GREEN, lw=2.5, label='φ*(ṁ)  – optymalny')
ax3.scatter([MDOT_DES], [phi_r[idx_des]], color=RED, s=100, zorder=6,
            edgecolors='darkred', lw=1.0,
            label=f'Punkt proj.  φ={phi_r[idx_des]:.3f}  (MR={MR_STOICH/phi_r[idx_des]:.2f})')
ax3.axhline(MR_STOICH / MR_DES, color='navy', ls=':', lw=1.4,
            label=f'φ_proj = {MR_STOICH/MR_DES:.2f}  (MR=7.94, wyjściowy)')

# Oś MR po prawej
ax3b = ax3.twinx()
_ylim = ax3.get_ylim()
ax3b.set_ylim(MR_STOICH / _ylim[1], MR_STOICH / _ylim[0])   # odwrotna skala MR
ax3b.plot([], [])  # pusta, tylko dla etykiety
ax3b.set_ylabel('O/F  (MR)  [-]', fontsize=10, color='dimgray')
ax3b.tick_params(axis='y', labelcolor='dimgray')

ax3.set_xlabel('Strumień masy  ṁ  [kg/s]', fontsize=11)
ax3.set_ylabel('Optymalny współ. ekwiwalencji  φ*  [-]', fontsize=11)
ax3.set_title('Optymalny skład mieszanki vs strumień masy', fontsize=12)
ax3.legend(fontsize=9, loc='best')
ax3.grid(True, ls='--', alpha=0.45)

# ── Wykres 4: Pe/Pa(ṁ) ─────────────────────────────────────────────────────
# Kolorowanie: over/design/underexpanded
ax4.fill_between(mdot_r, PePa_r, 1.0,
                 where=(PePa_r > 1.0), alpha=0.15, color='steelblue',
                 label='Niedorozprężenie (Pe > Pa)')
ax4.fill_between(mdot_r, PePa_r, 1.0,
                 where=(PePa_r < 1.0), alpha=0.15, color='salmon',
                 label='Nadrozprężenie (Pe < Pa)')
ax4.plot(mdot_r, PePa_r, color='black', lw=2.5, label='Pe/Pa(ṁ)')
ax4.axhline(1.0, color='seagreen', ls='--', lw=1.8,
            label='Punkt projektowy (Pe = Pa)')
ax4.axhline(0.35, color='red', ls=':', lw=1.4, alpha=0.7,
            label='Granica oderwania ≈ 0.35·Pa')
ax4.scatter([MDOT_DES], [PePa_r[idx_des]], color=RED, s=100, zorder=6,
            edgecolors='darkred', lw=1.0)
ax4.set_xlabel('Strumień masy  ṁ  [kg/s]', fontsize=11)
ax4.set_ylabel('Pe / Pa  [-]', fontsize=11)
ax4.set_title('Stan ekspansji dyszy vs strumień masy', fontsize=12)
ax4.legend(fontsize=8.5, loc='best')
ax4.grid(True, ls='--', alpha=0.45)

# Wspólna legenda dla linii projektowej
leg_des = Line2D([0], [0], color='gray', ls=':', lw=1.3,
                 label=f'ṁ_des = {MDOT_DES} kg/s')
for ax in [ax1, ax2, ax3, ax4]:
    handles, labels = ax.get_legend_handles_labels()
    # (linia projektowa dodawana tylko jeśli jest w legendzie)

plt.tight_layout()
out_path = '/share/Do_clauda_taty/analiza_dławienia.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nWykres zapisany: {out_path}")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 6. TABELA WYNIKOWA – wybranych punktów w zakresie 100–800 N
# ══════════════════════════════════════════════════════════════════════════════
# Interpolacja do równomiernych wartości F
F_targets = [100, 150, 200, 250, 300, 350, 400, 450,
             500, 550, 600, 650, 700, F_DES, 750, 800]
F_targets = sorted(set(F_targets))

print("\n" + "═" * 95)
print("  TABELA WYNIKOWA  –  wybrane wartości ciągu")
print("═" * 95)
print(f"  {'F [N]':>8}  {'ṁ [kg/s]':>10}  {'Pc [bar]':>10}  {'φ*':>7}  {'MR*':>7}  "
      f"{'Pe/Pa':>8}  {'ve [m/s]':>10}  {'Isp_eff [s]':>12}  {'Tc [K]':>8}")
print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*7}  {'─'*7}  "
      f"{'─'*8}  {'─'*10}  {'─'*12}  {'─'*8}")

for F_t in F_targets:
    if F_t < F_r.min() or F_t > F_r.max():
        continue
    # interpolacja
    m_t    = np.interp(F_t, F_r, mdot_r)
    Pc_t   = np.interp(F_t, F_r, Pc_r)
    phi_t  = np.interp(F_t, F_r, phi_r)
    PePa_t = np.interp(F_t, F_r, PePa_r)
    ve_t   = np.interp(F_t, F_r, ve_r)
    Isp_t  = np.interp(F_t, F_r, Isp_r)
    Tc_t   = np.interp(F_t, F_r, Tc_r)
    MR_t   = MR_STOICH / phi_t
    flag   = '  ← punkt proj.' if abs(F_t - F_DES) < 2 else ''
    print(f"  {F_t:>8.1f}  {m_t:>10.5f}  {Pc_t:>10.4f}  "
          f"{phi_t:>7.4f}  {MR_t:>7.4f}  {PePa_t:>8.4f}  "
          f"{ve_t:>10.2f}  {Isp_t:>12.2f}  {Tc_t:>8.1f}{flag}")

print("\n  UWAGA:")
print(f"  - Geometria stała: At = {At*1e6:.4f} mm², Ae = {Ae*1e6:.2f} mm², ε = {eps_e:.4f}")
print(f"  - Pe/Pa = 1 tylko w punkcie projektowym (F = {F_DES} N)")
print(f"  - Pe/Pa < 0.35: możliwe oderwanie strumienia w dyszy (nadrozprężenie)")
print(f"  - φ* maleje (mieszanka mniej paliwobogata) wraz ze wzrostem Pc / ṁ")
print("═" * 95)
