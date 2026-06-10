"""
Impuls właściwy Isp jako funkcja współczynnika ekwiwalencji φ
=====================================================================
Paliwo:    H2 (ciekły)
Utleniacz: H2O2 98% (blend 98% H2O2 + 2% H2O wagowo)

Założenia:
  - Stałe ciśnienie komory Pc = Pc_design (z obliczeń projektowych)
  - Przepływ isentropowy 1D, gaz doskonały, γ = const (z CEA)
  - Pe = Pa = 101325 Pa (dysza dopasowana do atmosfery)
  - Isp = ve / g0,  ve wyznaczone ze wzoru isentropowego

Stosunek stechiometryczny (czysty H2O2):
  H2 + H2O2 → 2 H2O
  MR_stoich = M(H2O2)/M(H2) = 34/2 = 17.0

Współczynnik ekwiwalencji:
  φ = MR_stoich / MR_actual
  φ > 1 → nadmiar paliwa (fuel-rich)
  φ < 1 → nadmiar utleniacza (fuel-lean)
  φ = 1 → stoichiometria
"""

import numpy as np
import matplotlib.pyplot as plt
from rocketcea.cea_obj import CEA_Obj
from rocketcea.blends import newOxBlend

# ─── STAŁE ────────────────────────────────────────────────────────────────────
R_UNIV     = 8314.46       # J/(kmol·K)
G0         = 9.80665       # m/s²
PSI_PER_PA = 1 / 6894.76   # Pa → psi
PA         = 101325.0      # Pa

# Ciśnienie komory stałe = wartość projektowa (MR = 7.94, F = 701.8 N, ṁ = 0.2 kg/s)
PC_BAR_DESIGN = 12.944     # bar

# Stechiometria: H2 + H2O2 → 2H2O, MR_stoich = 34/2 = 17.0 (dla czystego H2O2)
MR_STOICH = 17.0

# Punkt projektowy
MR_DESIGN = 7.94
PHI_DESIGN = MR_STOICH / MR_DESIGN   # ≈ 2.14

# ─── INICJALIZACJA ROCKETCEA ──────────────────────────────────────────────────
ox_blend = newOxBlend(oxL=['H2O2', 'H2O'], oxPcentL=[98, 2])
cea = CEA_Obj(fuelName='H2', oxName=ox_blend)

# ─── SWEEP MR ────────────────────────────────────────────────────────────────
# MR od 1 do 30; φ = MR_STOICH/MR → od 17 (rich) do 0.57 (lean)
MR_values = np.linspace(1.0, 28.0, 400)

Isp_list = []
phi_list = []
ve_list  = []

Pc_psia = PC_BAR_DESIGN * 1e5 * PSI_PER_PA
Pc_Pa   = PC_BAR_DESIGN * 1e5

for MR in MR_values:
    try:
        Tc         = cea.get_Tcomb(Pc=Pc_psia, MR=MR)
        Mw, gamma  = cea.get_Chamber_MolWt_gamma(Pc=Pc_psia, MR=MR)
        R_sp       = R_UNIV / Mw
        Pe_Pc      = PA / Pc_Pa

        if Pe_Pc >= 1.0:
            raise ValueError("Pe >= Pc – niemożliwa ekspansja")

        ve = np.sqrt(
            2 * gamma / (gamma - 1)
            * R_sp * Tc
            * (1 - Pe_Pc ** ((gamma - 1) / gamma))
        )
        Isp = ve / G0

        Isp_list.append(Isp)
        ve_list.append(ve)
        phi_list.append(MR_STOICH / MR)

    except Exception:
        Isp_list.append(np.nan)
        ve_list.append(np.nan)
        phi_list.append(MR_STOICH / MR)

phi_arr = np.array(phi_list)
Isp_arr = np.array(Isp_list)

# Punkt maksimum
idx_max  = np.nanargmax(Isp_arr)
phi_max  = phi_arr[idx_max]
Isp_max  = Isp_arr[idx_max]
MR_max   = MR_STOICH / phi_max

print(f"  Maksimum Isp:")
print(f"    φ_opt   = {phi_max:.3f}")
print(f"    MR_opt  = {MR_max:.3f}")
print(f"    Isp_max = {Isp_max:.2f} s")
print(f"\n  Punkt projektowy (MR = {MR_DESIGN}):")
print(f"    φ       = {PHI_DESIGN:.3f}")
idx_design = np.argmin(np.abs(phi_arr - PHI_DESIGN))
print(f"    Isp     = {Isp_arr[idx_design]:.2f} s")
print(f"\n  Stechiometria (φ = 1, MR = {MR_STOICH:.1f}):")
idx_stoich = np.argmin(np.abs(phi_arr - 1.0))
print(f"    Isp     = {Isp_arr[idx_stoich]:.2f} s")

# ─── WYKRES ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(phi_arr, Isp_arr, color='royalblue', linewidth=2.5, label='Isp(φ)')

# Pionowe linie pomocnicze
ax.axvline(phi_max, color='crimson', linestyle='--', linewidth=1.8,
           label=f'Maximum: φ = {phi_max:.2f}  (MR = {MR_max:.2f})  →  Isp = {Isp_max:.0f} s')

ax.axvline(PHI_DESIGN, color='darkorange', linestyle='-.', linewidth=1.8,
           label=f'Punkt projektowy: φ = {PHI_DESIGN:.2f}  (MR = {MR_DESIGN})  →  Isp = {Isp_arr[idx_design]:.0f} s')

ax.axvline(1.0, color='dimgray', linestyle=':', linewidth=1.5,
           label=f'Stechiometria: φ = 1  (MR = {MR_STOICH:.0f})  →  Isp = {Isp_arr[idx_stoich]:.0f} s')

# Oznaczenie maksimum
ax.scatter([phi_max], [Isp_max], color='crimson', zorder=5, s=80)
ax.annotate(f'  φ_opt = {phi_max:.2f}\n  Isp = {Isp_max:.0f} s',
            xy=(phi_max, Isp_max), xytext=(phi_max + 0.4, Isp_max - 25),
            fontsize=10, color='crimson',
            arrowprops=dict(arrowstyle='->', color='crimson', lw=1.2))

# Oznaczenie punktu projektowego
ax.scatter([PHI_DESIGN], [Isp_arr[idx_design]], color='darkorange', zorder=5, s=80)

# Regiony paliwobogaty / stechiometria / utleniaczbogaty
ax.axvspan(phi_arr.min(), 1.0,  alpha=0.04, color='steelblue',  label='Nadmiar utleniacza (φ < 1)')
ax.axvspan(1.0, phi_arr.max(), alpha=0.04, color='tomato', label='Nadmiar paliwa (φ > 1)')

# Formatowanie
ax.set_xlabel('Współczynnik ekwiwalencji  φ = MR$_{stoich}$ / MR  [-]', fontsize=12)
ax.set_ylabel('Impuls właściwy  $I_{sp}$  [s]', fontsize=12)
ax.set_title('Impuls właściwy vs. współczynnik ekwiwalencji\n'
             'H₂ / H₂O₂ 98%,  $P_c$ = 12.944 bar,  $P_e = P_a$ = 1 atm',
             fontsize=13)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlim(phi_arr.min() - 0.05, phi_arr.max() + 0.05)

# Dodatkowa oś górna: MR
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
# Etykiety MR w wybranych punktach φ
phi_ticks = [0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
phi_ticks = [p for p in phi_ticks if phi_arr.min() <= p <= phi_arr.max()]
ax2.set_xticks(phi_ticks)
ax2.set_xticklabels([f'{MR_STOICH/p:.1f}' for p in phi_ticks], fontsize=8)
ax2.set_xlabel('Stosunek O/F  (MR)  [-]', fontsize=10)

plt.tight_layout()
plt.savefig('/share/Do_clauda_taty/Isp_vs_phi.png', dpi=150, bbox_inches='tight')
print("\n  Wykres zapisany: Isp_vs_phi.png")
plt.show()
