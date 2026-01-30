from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List
import numpy as np
from scipy.integrate import solve_ivp

from src.utils import celsius_to_kelvin, clamp


@dataclass
class CapacityParams:
    """
    Capacity model. For this project we use a constant nominal capacity (from static.csv),
    but keep the interface extensible.

    C_eff_Ah = C_nom_Ah * eta(T)
    """
    C_nom_Ah: float

    def ceff_ah(self, T_C: float) -> float:
        return float(self.C_nom_Ah)


@dataclass
class ResistanceParams:
    """
    Simple Arrhenius + SOC dependence:
      R(SOC,T) = R_ref * exp(Ea_over_R * (1/T - 1/Tref)) * (1 + b*(1-SOC))

    Notes:
      - Ea_over_R_K has unit Kelvin.
      - This is a deliberately simple 'reasonable' form.
    """
    R_ref_ohm: float = 0.06
    Ea_over_R_K: float = 3500.0
    Tref_C: float = 25.0
    b_soc: float = 0.25

    def r_ohm(self, soc: float, T_C: float) -> float:
        soc = clamp(float(soc), 0.0, 1.0)
        T = float(celsius_to_kelvin(T_C))
        Tref = float(celsius_to_kelvin(self.Tref_C))
        arr = np.exp(self.Ea_over_R_K * (1.0 / T - 1.0 / Tref))
        soc_factor = 1.0 + self.b_soc * (1.0 - soc)
        return float(max(1e-5, self.R_ref_ohm * arr * soc_factor))


@dataclass
class OCVParams:
    """
    Smooth OCV curve:
      E_ocv(SOC) = Vmin + (Vmax-Vmin)*sigmoid(k*(SOC-s0)) + a_lin*(SOC-0.5)

    This is a generic Li-ion shape. You can later calibrate it to the dataset.
    """
    Vmin: float = 3.0
    Vmax: float = 4.2
    k: float = 8.0
    s0: float = 0.5
    a_lin: float = 0.05

    def e_ocv(self, soc: float) -> float:
        s = clamp(float(soc), 0.0, 1.0)
        sig = 1.0 / (1.0 + np.exp(-self.k * (s - self.s0)))
        return float(self.Vmin + (self.Vmax - self.Vmin) * sig + self.a_lin * (s - 0.5))


@dataclass
class BatterySOCModel:
    """
    Continuous-time SOC model (from your PDF):

      dSOC/dt = - I(t) / (C_eff * 3600)                         (0)
      I(t) = P(t) / V_term(t)                                  (1)
      V_term(t) = E_ocv(SOC) - I(t) * R_int(SOC, T)             (2)

    Solving (1)(2) gives:
      R I^2 - E I + P = 0  (choose the physical discharge root)

    We integrate SOC(t) with solve_ivp, and we can stop when:
      - SOC reaches cutoff_soc (default 0.0)
      - terminal voltage reaches cutoff_voltage_V (e.g., 3.0V)
    """
    capacity: CapacityParams
    ocv: OCVParams = field(default_factory=OCVParams)
    rint: ResistanceParams = field(default_factory=ResistanceParams)

    cutoff_soc: float = 0.0
    cutoff_voltage_V: Optional[float] = 3.0

    def current_from_power(self, P_W: float, soc: float, T_C: float) -> float:
        """
        Solve R I^2 - E I + P = 0 for I (A). We assume discharge current I>=0.

        If P is too high such that discriminant < 0, we clamp it to 0
        (meaning the load is at the feasibility boundary given E and R).
        """
        P = max(0.0, float(P_W))
        E = self.ocv.e_ocv(soc)
        R = self.rint.r_ohm(soc, T_C)

        disc = E * E - 4.0 * R * P
        if disc < 0.0:
            disc = 0.0
        sqrt_disc = float(np.sqrt(disc))

        # Discharge physical root (smaller one)
        I = (E - sqrt_disc) / (2.0 * R)
        if R < 1e-5:
            I = P / max(E, 1e-6)
        return float(max(0.0, I))

    def terminal_voltage(self, I_A: float, soc: float, T_C: float) -> float:
        E = self.ocv.e_ocv(soc)
        R = self.rint.r_ohm(soc, T_C)
        return float(E - float(I_A) * R)

    def simulate_power_driven(
        self,
        t_span: Tuple[float, float],
        soc0: float,
        T_C: float,
        power_W: Callable[[float], float],
        t_eval: Optional[np.ndarray] = None,
        rtol: float = 1e-6,
        atol: float = 1e-9,
    ):
        """
        Integrate SOC under power profile P(t). Returns scipy solve_ivp solution.
        """
        Ceff_Ah = self.capacity.ceff_ah(T_C)
        if Ceff_Ah <= 0:
            raise ValueError("Capacity must be > 0")
        Ceff_c = Ceff_Ah * 3600.0

        def rhs(t, y):
            soc = clamp(float(y[0]), 0.0, 1.0)
            P = float(power_W(t))
            I = self.current_from_power(P, soc, T_C)
            return [-I / Ceff_c]

        events: List[Callable] = []

        def event_soc(t, y):
            return float(y[0] - self.cutoff_soc)
        event_soc.terminal = True
        event_soc.direction = -1
        events.append(event_soc)

        if self.cutoff_voltage_V is not None:
            Vcut = float(self.cutoff_voltage_V)

            def event_vterm(t, y):
                soc = clamp(float(y[0]), 0.0, 1.0)
                P = float(power_W(t))
                I = self.current_from_power(P, soc, T_C)
                V = self.terminal_voltage(I, soc, T_C)
                return float(V - Vcut)

            event_vterm.terminal = True
            event_vterm.direction = -1
            events.append(event_vterm)

        sol = solve_ivp(
            rhs, t_span, [float(soc0)],
            t_eval=t_eval, rtol=rtol, atol=atol,
            events=events if events else None
        )
        return sol
