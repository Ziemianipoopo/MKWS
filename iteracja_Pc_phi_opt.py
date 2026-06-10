"""
==============================================================================
ITERACYJNE WYZNACZENIE OPTYMALNEGO Pc I φ – SILNIK H2/H2O2 98%
==============================================================================
Paliwo:    H2 (ciekły wodór)
Utleniacz: H2O2 98%  (blend 98% H2O2 + 2% H2O wagowo)
Zadane:    F = 701.8 N,  ṁ = 0.2 kg/s,  Pa = 1 atm

ALGORYTM:
─────────────────────────────────────────────────────────────────────────────
Przy zadanym Pe = Pa równanie ciągu: F = ṁ·ve → ve_cel = F/ṁ = 3509 m/s

Szukamy pary (Pc*, MR*) takiej, że:
  (1) MR* = argmax_MR  ve_isen(Pc*, MR)     ← optymalny φ dla danego Pc
  (2) ve_isen(Pc*, MR*) = ve_cel             ← wymagany ciąg

Iteracja zewnętrzna (bisekcja po Pc):
  - ve_max(Pc) = max_MR ve_isen(Pc, MR) jest monotonicznie rosnące z Pc
  - szukamy Pc* bisekcją: ve_max(Pc*) = ve_cel

Iteracja wewnętrzna (minimalizacja po MR dla danego Pc):
  - scipy.optimize.minimize_scalar (metoda bounded, Brent)

WYNIKI WYKRESU:
  - krzywe Isp(φ) dla Pc = 2, 4, 6, 8, 10, 12, 15, 20, 25, 30 bar
  - linia łącząca maksima (locus φ_opt(Pc)) – wyznaczona na gęstszej siatce
  - prawy panel: φ_opt(Pc) i Isp_max(Pc) jako funkcja Pc
==============================================================================
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from rocketcea.cea_obj import CEA_Obj
from rocketcea.blends import newOxBlend


# ══════════════════════════════════════════════════════════════════════════════
# 0. STAŁE I DANE WEJŚCIOWE
# ══════════════════════════════════════════════════════════════════════════════
R_UNIV     = 8314.46        # [J/(kmol·K)]
G0         = 9.80665        # [m/s²]
PSI_PER_PA = 1 / 6894.76    # Pa → psi
PA         = 101325.0       # [Pa]  ciśnienie otoczenia
MR_STOICH  = 17.0           # stechiometria H2/H2O2: 34/2

F          = 701.8          # [N]   zadany ciąg
M_DOT      = 0.2            # [kg/s] zadany strumień masy
ve_cel     = F / M_DOT      # [m/s] = 3509.0 – wymagana efektywna prędkość wylotowa
Isp_cel    = ve_cel / G0    # [s]   ≈ 357.82

EPS_C      = 8.0            # Ac/At – stosunek przekrojów komory do gardła
L_STAR     = 0.80           # [m]   charakterystyczna długość komory (nominalna)

print(f"Wymagana ve_cel = {ve_cel:.4f} m/s  →  Isp_cel = {Isp_cel:.4f} s")
print(f"Cel iteracji:  M_DOT × ve_max(Pc*, MR*(Pc*)) = F = {F} N\n")


# ══════════════════════════════════════════════════════════════════════════════
# 1. INICJALIZACJA ROCKETCEA
# ══════════════════════════════════════════════════════════════════════════════
ox_blend = newOxBlend(oxL=['H2O2', 'H2O'], oxPcentL=[98, 2])
cea      = CEA_Obj(fuelName='H2', oxName=ox_blend)


# ══════════════════════════════════════════════════════════════════════════════
# 2. FUNKCJE POMOCNICZE
# ══════════════════════════════════════════════════════════════════════════════

def get_cea_params(Pc_bar: float, MR: float):
    """Tc [K], Mw [g/mol], gamma [-], R_sp [J/(kg·K)]"""
    Pc_psia = Pc_bar * 1e5 * PSI_PER_PA
    Tc      = cea.get_Tcomb(Pc=Pc_psia, MR=MR)
    Mw, gamma = cea.get_Chamber_MolWt_gamma(Pc=Pc_psia, MR=MR)
    return Tc, Mw, gamma, R_UNIV / Mw


def ve_isen(Pc_bar: float, MR: float) -> float:
    """
    Isentropowa prędkość wylotowa dyszy dopasowanej do Pa.
      ve = sqrt(2γ/(γ-1) · R_sp · Tc · [1 − (Pa/Pc)^((γ-1)/γ)])
    """
    if Pc_bar * 1e5 <= PA:
        return 0.0
    try:
        Tc, _, gamma, R_sp = get_cea_params(Pc_bar, MR)
        Pe_Pc = PA / (Pc_bar * 1e5)
        return np.sqrt(2 * gamma / (gamma - 1) * R_sp * Tc
                       * (1 - Pe_Pc ** ((gamma - 1) / gamma)))
    except Exception:
        return 0.0


def area_ratio(M: float, gamma: float) -> float:
    """A/A* = 1/M · [2/(γ+1) · (1 + (γ-1)/2 · M²)]^((γ+1)/(2(γ-1)))"""
    f = (2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * M**2)
    return (1 / M) * f ** ((gamma + 1) / (2 * (gamma - 1)))


def find_opt_at_pc(Pc_bar: float, MR_lo: float = 2.5, MR_hi: float = 12.0):
    """
    Wyznacza MR minimalizujące −ve_isen (= maksymalizujące Isp) przy danym Pc.
    Używa metody Brenta (bounded minimize_scalar).
    Zwraca: (MR_opt, phi_opt, ve_max, Isp_max)
    """
    res = minimize_scalar(
        lambda MR: -ve_isen(Pc_bar, MR),
        bounds=(MR_lo, MR_hi),
        method='bounded',
        options={'xatol': 5e-4}
    )
    MR_opt = res.x
    ve_max = -res.fun
    return MR_opt, MR_STOICH / MR_opt, ve_max, ve_max / G0


# ══════════════════════════════════════════════════════════════════════════════
# 3. KROK A – Krzywe Isp(φ) dla wybranych Pc (do wykresu)
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 60)
print("KROK A: Obliczam krzywe Isp(φ) dla wybranych Pc...")
print("─" * 60)

Pc_plot = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]   # [bar]
MR_arr  = np.linspace(2.5, 22.0, 150)
phi_arr = MR_STOICH / MR_arr

curves = {}
for Pc_bar in Pc_plot:
    isp = []
    for MR in MR_arr:
        ve = ve_isen(Pc_bar, MR)
        isp.append(ve / G0 if ve > 0 else np.nan)
    curves[Pc_bar] = np.array(isp)
    print(f"  Pc = {Pc_bar:2d} bar  →  Isp_max ≈ {np.nanmax(curves[Pc_bar]):.1f} s"
          f"  przy φ ≈ {phi_arr[np.nanargmax(curves[Pc_bar])]:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. KROK B – Linia maksimów φ_opt(Pc) na gęstej siatce Pc
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("KROK B: Wyznaczam φ_opt(Pc) na gęstej siatce Pc...")
print("─" * 60)

Pc_fine = np.unique(np.concatenate([
    np.linspace(1.5,  6.0, 20),
    np.linspace(6.0, 15.0, 25),
    np.linspace(15.0, 30.0, 15),
]))

# opt_rows: [Pc_bar, phi_opt, MR_opt, ve_max, Isp_max]
opt_rows = []
for Pc_bar in Pc_fine:
    try:
        MR_o, phi_o, ve_o, Isp_o = find_opt_at_pc(Pc_bar)
        opt_rows.append([Pc_bar, phi_o, MR_o, ve_o, Isp_o])
    except Exception:
        pass
opt_data = np.array(opt_rows)
# kolumny: 0=Pc, 1=phi_opt, 2=MR_opt, 3=ve_max, 4=Isp_max

print(f"  Gotowe – {len(opt_data)} punktów")
print(f"  Zakres φ_opt:   {opt_data[:,1].min():.3f} – {opt_data[:,1].max():.3f}")
print(f"  Zakres Isp_max: {opt_data[:,4].min():.1f} – {opt_data[:,4].max():.1f} s")


# ══════════════════════════════════════════════════════════════════════════════
# 5. KROK C – Iteracja bisekcją: szukanie Pc* gdzie ve_max(Pc*) = ve_cel
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print(f"KROK C: Iteracja bisekcją  (cel: ve_max = {ve_cel:.2f} m/s = F/ṁ)")
print("─" * 60)
print(f"  Sprawdzenie granic przedziału:")

_, _, ve_lo, _ = find_opt_at_pc(1.5)
_, _, ve_hi, _ = find_opt_at_pc(30.0)
print(f"    ve_max(Pc= 1.5 bar) = {ve_lo:.2f} m/s  {'< cel ✓' if ve_lo < ve_cel else '≥ cel ✗'}")
print(f"    ve_max(Pc=30.0 bar) = {ve_hi:.2f} m/s  {'> cel ✓' if ve_hi > ve_cel else '≤ cel ✗'}")
print()

hdr = (f"{'Iter':>4}  {'Pc [bar]':>10}  {'φ_opt':>8}  {'MR_opt':>8}  "
       f"{'ve_max [m/s]':>14}  {'F_eq [N]':>10}  {'ΔF/F [%]':>10}")
print(hdr)
print("─" * len(hdr))

Pc_lo_bis, Pc_hi_bis = 1.5, 30.0
Pc_star = MR_star = phi_star = ve_star = Isp_star = None

for i in range(35):
    Pc_mid = (Pc_lo_bis + Pc_hi_bis) / 2.0
    MR_o, phi_o, ve_o, Isp_o = find_opt_at_pc(Pc_mid)
    F_eq   = M_DOT * ve_o
    delta  = (F_eq - F) / F * 100.0

    print(f"{i+1:>4}  {Pc_mid:>10.5f}  {phi_o:>8.4f}  {MR_o:>8.4f}  "
          f"{ve_o:>14.5f}  {F_eq:>10.5f}  {delta:>+10.5f}")

    Pc_star, MR_star, phi_star, ve_star, Isp_star = Pc_mid, MR_o, phi_o, ve_o, Isp_o

    if abs(delta) < 1e-4:
        print(f"\n  ✓ Zbieżność po {i+1} iteracjach (|ΔF/F| < 0.0001 %)")
        break
    if F_eq > F:
        Pc_hi_bis = Pc_mid
    else:
        Pc_lo_bis = Pc_mid


# ══════════════════════════════════════════════════════════════════════════════
# 6. KROK D – Pełne parametry silnika w (Pc*, MR*)
# ══════════════════════════════════════════════════════════════════════════════
Pc_Pa = Pc_star * 1e5
Tc, Mw, gamma, R_sp = get_cea_params(Pc_star, MR_star)
rho_c = Pc_Pa / (R_sp * Tc)
Cstar = (np.sqrt(R_sp * Tc / gamma)
         * ((gamma + 1) / 2) ** ((gamma + 1) / (2 * (gamma - 1))))

# Gardło (M = 1)
Tt    = Tc * 2 / (gamma + 1)
Pt    = Pc_Pa * (2 / (gamma + 1)) ** (gamma / (gamma - 1))
rho_t = rho_c * (2 / (gamma + 1)) ** (1 / (gamma - 1))
vt    = np.sqrt(gamma * R_sp * Tt)
At    = M_DOT / (rho_t * vt)
Dt    = np.sqrt(4 * At / np.pi)
Cstar_chk = Pc_Pa * At / M_DOT

# Wylot dyszy (Pe = Pa)
Pe      = PA
Te      = Tc * (Pe / Pc_Pa) ** ((gamma - 1) / gamma)
Me      = np.sqrt(2 / (gamma - 1) * (Tc / Te - 1))
rho_e   = rho_c * (Pe / Pc_Pa) ** (1 / gamma)
ve_e    = np.sqrt(2 * gamma / (gamma - 1) * R_sp * Tc
                  * (1 - (Pe / Pc_Pa) ** ((gamma - 1) / gamma)))
eps_e   = area_ratio(Me, gamma)
Ae      = eps_e * At
De      = np.sqrt(4 * Ae / np.pi)
F_chk   = M_DOT * ve_e + Ae * (Pe - PA)

# Komora – dokładna (Ac/At = EPS_C, bez założenia Mc ≈ 0)
Mc      = brentq(lambda Mc_: area_ratio(Mc_, gamma) - EPS_C, 1e-8, 1.0 - 1e-10)
Tc_loc  = Tc / (1 + (gamma - 1) / 2 * Mc**2)
Pc_loc  = Pc_Pa * (1 + (gamma - 1) / 2 * Mc**2) ** (-(gamma / (gamma - 1)))
rho_loc = Pc_loc / (R_sp * Tc_loc)
vc      = Mc * np.sqrt(gamma * R_sp * Tc_loc)
Ac      = EPS_C * At
Dc      = np.sqrt(4 * Ac / np.pi)

# Objętość komory (L* = 0.80 m)
theta_rad = np.radians(30.0)
Rc     = np.sqrt(Ac / np.pi)
Rt     = np.sqrt(At / np.pi)
L_conv = (Rc - Rt) / np.tan(theta_rad)
V_conv = np.pi / 3 * L_conv * (Rc**2 + Rc * Rt + Rt**2)
Vc     = L_STAR * At
V_cyl  = Vc - V_conv
L_cyl  = V_cyl / Ac if V_cyl > 0 else 0.0
L_tot  = L_cyl + L_conv
tau_ms = rho_c * Vc / M_DOT * 1e3

# ── Drukowanie wyników ────────────────────────────────────────────────────────
SEP = "═" * 70
print(f"\n{SEP}")
print(f"  WYNIKI KOŃCOWE – SILNIK H2/H2O2 98%")
print(f"  Optymalny (Pc*, φ*) przy zadanym F = {F} N, ṁ = {M_DOT} kg/s")
print(SEP)

print(f"\n  PUNKT PRACY (iteracja zbieżna):")
print(f"    φ*  (ekwiwalencja)   = {phi_star:.4f}  (φ_stoich=1 → nadmiar H₂)")
print(f"    MR* (O/F)            = {MR_star:.4f}")
print(f"    Pc*                  = {Pc_Pa:>14.2f} Pa  =  {Pc_star:.5f} bar")
print(f"    Isp*                 = {Isp_star:.4f} s      ve* = {ve_star:.4f} m/s")

print(f"\n  PARAMETRY TERMODYNAMICZNE (NASA CEA):")
print(f"    γ  (wykładnik adi.)  = {gamma:.4f}")
print(f"    Mw (masa molarna)    = {Mw:.4f} g/mol")
print(f"    R_sp (właściwa R)    = {R_sp:.4f} J/(kg·K)")
print(f"    C*  (char. prędkość) = {Cstar:.4f} m/s")
print(f"    C*  (Pc·At/ṁ, wery.)= {Cstar_chk:.4f} m/s")

W = 26  # szerokość kolumny opisu
print(f"\n  {'─'*68}")
print(f"  {'Parametr':<{W}}  {'Komora':>16}  {'Gardło':>10}  {'Wylot':>10}  Jedn.")
print(f"  {'─'*W}  {'─'*16}  {'─'*10}  {'─'*10}  {'─'*6}")
print(f"  {'Ciśnienie  P':<{W}}  {Pc_Pa:>16.1f}  {Pt:>10.1f}  {Pe:>10.1f}  Pa")
print(f"  {'Ciśnienie  P':<{W}}  {Pc_star:>16.5f}  {Pt/1e5:>10.5f}  {Pe/1e5:>10.6f}  bar")
print(f"  {'Temperatura  T':<{W}}  {Tc:>16.2f}  {Tt:>10.2f}  {Te:>10.2f}  K")
print(f"  {'Gęstość  ρ':<{W}}  {rho_c:>16.6f}  {rho_t:>10.6f}  {rho_e:>10.6f}  kg/m³")
print(f"  {'Liczba Macha  M':<{W}}  {'≈0 ('+f'{Mc:.4f}'+')':>16}  {'1.0000':>10}  {Me:>10.4f}  –")
print(f"  {'Prędkość  v':<{W}}  {'≈0 ('+f'{vc:.1f}'+')':>16}  {vt:>10.2f}  {ve_e:>10.2f}  m/s")
print(f"  {'Przekrój  A':<{W}}  {Ac:>16.4e}  {At:>10.4e}  {Ae:>10.4e}  m²")
print(f"  {'Średnica  D':<{W}}  {Dc*1e3:>16.4f}  {Dt*1e3:>10.4f}  {De*1e3:>10.4f}  mm")
print(f"  {'A/A*':<{W}}  {Ac/At:>16.4f}  {'1.0000':>10}  {eps_e:>10.4f}  –")
print(f"  {'─'*68}")

print(f"\n  WERYFIKACJA CIĄGU:")
print(f"    F = ṁ·ve + Ae·(Pe−Pa) = {F_chk:.4f} N")
print(f"    Zadany F              = {F:.4f} N")
print(f"    Błąd                  = {abs(F_chk - F)/F*100:.5f} %")

print(f"\n  OBJĘTOŚĆ KOMORY  (L* = {L_STAR} m, θ_zbieżna = 30°):")
print(f"    Vc                   = {Vc*1e6:.2f} cm³  = {Vc*1e3:.4f} L")
print(f"    Lcyl + Lconv         = {L_cyl*1e3:.1f} + {L_conv*1e3:.1f} = {L_tot*1e3:.1f} mm")
print(f"    τ  (czas przebywania) = {tau_ms:.3f} ms")

print(f"\n  PORÓWNANIE Z WARIANTEM PROJEKTOWYM (MR=7.94, Pc=12.944 bar):")
print(f"    Δφ:     {phi_star:.4f} vs 2.1400  →  mieszanka mniej paliwobogata przy Pc*")
print(f"    ΔPc:    {Pc_star:.3f} vs 12.944 bar  →  {'niższe' if Pc_star < 12.944 else 'wyższe'} Pc* przy optymalnym MR")
print(f"    ΔMw:    {Mw:.3f} vs 12.163 g/mol")
print(f"    ΔTc:    {Tc:.1f} vs 4335.3 K")
print(f"    Isp:    oba = {Isp_star:.2f} s  (wynika z zadanego F i ṁ)")

print(f"\n{SEP}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. KROK E – Wykres
# ══════════════════════════════════════════════════════════════════════════════

# Schemat kolorów: plasma od ciemnego (niskie Pc) do jasnego (wysokie Pc)
norm_c   = mcolors.LogNorm(vmin=min(Pc_plot), vmax=max(Pc_plot))
cmap_c   = cm.plasma
col_list = [cmap_c(norm_c(p)) for p in Pc_plot]

fig, (ax_L, ax_R) = plt.subplots(1, 2, figsize=(17, 7))
fig.suptitle('Optymalizacja składu mieszanki i ciśnienia komory\n'
             'H₂ / H₂O₂ 98%,  F = 701.8 N,  ṁ = 0.2 kg/s,  Pe = Pa = 1 atm',
             fontsize=13, y=1.01)

# ── Panel lewy: Isp(φ) ────────────────────────────────────────────────────────
for Pc_bar, col in zip(Pc_plot, col_list):
    isp_c = curves[Pc_bar]
    valid = ~np.isnan(isp_c)
    ax_L.plot(phi_arr[valid], isp_c[valid], color=col, lw=1.9,
              label=f'{Pc_bar} bar')
    if valid.any():
        idx = np.nanargmax(isp_c)
        ax_L.scatter(phi_arr[idx], isp_c[idx],
                     color=col, s=55, zorder=5,
                     edgecolors='k', linewidths=0.6)

# Linia przez maksima (locus φ_opt(Pc))
ax_L.plot(opt_data[:, 1], opt_data[:, 4], 'k--', lw=2.4, zorder=8,
          label='Linia maksimów  $\\phi_{opt}(P_c)$')

# Pozioma linia: docelowy Isp
ax_L.axhline(Isp_cel, color='dimgray', ls=':', lw=1.6,
             label=f'$I_{{sp}}$ = {Isp_cel:.1f} s  (zadany)')

# Punkt optymalny Pc*
ax_L.scatter([phi_star], [Isp_star],
             color='red', s=160, zorder=10,
             edgecolors='darkred', linewidths=1.2,
             label=(f'Optimum: Pc* = {Pc_star:.2f} bar\n'
                    f'               φ* = {phi_star:.2f}  (MR* = {MR_star:.2f})'))

# Colorbar
sm = cm.ScalarMappable(cmap=cmap_c, norm=norm_c)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax_L, pad=0.02, aspect=25)
cbar.set_label('Pc  [bar]', fontsize=10)
cbar.set_ticks(Pc_plot)
cbar.set_ticklabels([str(p) for p in Pc_plot], fontsize=8)

ax_L.set_xlabel('Współczynnik ekwiwalencji  φ  [-]', fontsize=12)
ax_L.set_ylabel('Impuls właściwy  $I_{sp}$  [s]', fontsize=12)
ax_L.set_title('$I_{sp}(\\phi)$ przy różnych  Pc', fontsize=12)
ax_L.legend(fontsize=8.5, loc='lower right', ncol=2, framealpha=0.9)
ax_L.grid(True, ls='--', alpha=0.4)
ax_L.set_xlim(phi_arr.min() - 0.1, phi_arr.max() + 0.1)

# ── Panel prawy: φ_opt(Pc) i Isp_max(Pc) ─────────────────────────────────────
c1, c2 = 'royalblue', 'darkorange'
ax_R2 = ax_R.twinx()

ln1, = ax_R.plot(opt_data[:, 0], opt_data[:, 1], color=c1, lw=2.3,
                 label='$\\phi_{opt}(P_c)$')
ln2, = ax_R2.plot(opt_data[:, 0], opt_data[:, 4], color=c2, lw=2.3,
                  label='$I_{sp,\\max}(P_c)$')

# Punkt optymalny
for ax_tmp in [ax_R, ax_R2]:
    ax_tmp.axvline(Pc_star, color='red', ls='--', lw=1.5)
ax_R.scatter([Pc_star], [phi_star], color='red', s=110, zorder=10,
             edgecolors='darkred', linewidths=1.0)
ax_R2.scatter([Pc_star], [Isp_star], color='red', s=110, zorder=10,
              edgecolors='darkred', linewidths=1.0)

# Linia docelowego Isp
ax_R2.axhline(Isp_cel, color='dimgray', ls=':', lw=1.5,
              label=f'Isp cel = {Isp_cel:.1f} s')

# Adnotacja
ax_R.annotate(f'Pc* = {Pc_star:.2f} bar\nφ* = {phi_star:.2f}',
              xy=(Pc_star, phi_star),
              xytext=(Pc_star + 2.5, phi_star + 0.15),
              fontsize=9.5, color='darkred',
              arrowprops=dict(arrowstyle='->', color='darkred', lw=1.1))

ax_R.set_xlabel('Ciśnienie komory  Pc  [bar]', fontsize=12)
ax_R.set_ylabel('$\\phi_{opt}$  [-]', color=c1, fontsize=12)
ax_R.tick_params(axis='y', labelcolor=c1)
ax_R2.set_ylabel('$I_{sp,\\max}$  [s]', color=c2, fontsize=12)
ax_R2.tick_params(axis='y', labelcolor=c2)

ax_R.set_title(f'$\\phi_{{opt}}$ i $I_{{sp,\\max}}$ jako funkcja  Pc\n'
               f'Optimum: Pc* = {Pc_star:.2f} bar,  φ* = {phi_star:.2f}', fontsize=12)

all_lines = [ln1, ln2]
ax_R.legend(all_lines, [l.get_label() for l in all_lines],
            fontsize=10.5, loc='lower right')
ax_R.grid(True, ls='--', alpha=0.4)

plt.tight_layout()
out_path = '/share/Do_clauda_taty/iteracja_Pc_phi_opt.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nWykres zapisany: {out_path}")
plt.show()
