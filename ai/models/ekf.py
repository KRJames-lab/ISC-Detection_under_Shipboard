"""
Battery EKF — Extended Kalman Filter for 1RC ECM (Module 8S24P)

State:  x = [SOC, V_RC1]
Model:  V_pred = OCV(SOC) - I*R0 - V_RC1
        SOC(k+1) = SOC(k) + I*dt / (Q_ah * 3600)
        V_RC(k+1) = V_RC(k)*exp(-dt/tau1) + I*R1*(1 - exp(-dt/tau1))

Parameters from Simscape NCR18650PF model, scaled to 8S24P module.
"""
import numpy as np
from ai.config import (
    EKF_SOC_VEC, EKF_OCV_MODULE,
    EKF_R0_SOC_VEC, EKF_R0_MODULE, EKF_R1_MODULE, EKF_TAU1,
    EKF_AH_MODULE, EKF_SOC_INIT,
    EKF_Q_SOC, EKF_Q_VRC, I_LOAD,
)


def _interp_deriv(soc, soc_vec, table):
    """Numerical derivative of lookup table w.r.t. SOC (for Jacobian)."""
    ds = 1e-5
    f_plus = np.interp(soc + ds, soc_vec, table)
    f_minus = np.interp(soc - ds, soc_vec, table)
    return (f_plus - f_minus) / (2 * ds)


class BatteryEKF:
    """Extended Kalman Filter for 1RC battery module."""

    def __init__(self, dt, Q_diag=None, R_scalar=None):
        """
        Args:
            dt: Sampling interval (seconds), e.g. 0.01 for 100Hz.
            Q_diag: Process noise [q_soc, q_vrc] std. None uses config defaults.
            R_scalar: Measurement noise std (V). None = must be set via calibrate().
        """
        self.dt = dt
        self.Q_ah = EKF_AH_MODULE

        q_soc = Q_diag[0] if Q_diag is not None else EKF_Q_SOC
        q_vrc = Q_diag[1] if Q_diag is not None else EKF_Q_VRC
        self.Q = np.diag([q_soc ** 2, q_vrc ** 2])

        self.R_scalar = R_scalar  # scalar variance (sigma^2)

        # OCV table (18-point, SOC 0.73-0.90, from simulation)
        self.OCV_table = np.array(EKF_OCV_MODULE)
        self.OCV_soc = np.array(EKF_SOC_VEC)
        # R tables (7-point, SOC 0-1.0, from Simscape params, mOhm → Ohm)
        self.R0_table = np.array(EKF_R0_MODULE) * 1e-3  # Ohm
        self.R1_table = np.array(EKF_R1_MODULE) * 1e-3
        self.R_soc = np.array(EKF_R0_SOC_VEC)
        self.tau1_table = np.array(EKF_TAU1, dtype=float)

        self.reset()

    def reset(self):
        """Reset state to initial conditions."""
        self.x = np.array([EKF_SOC_INIT, 0.0])  # [SOC, V_RC1]
        self.P = np.diag([1e-4, 1e-6])           # Initial covariance

    def _get_params(self, soc):
        """Interpolate ECM parameters at current SOC."""
        ocv = np.interp(soc, self.OCV_soc, self.OCV_table)
        r0 = np.interp(soc, self.R_soc, self.R0_table)
        r1 = np.interp(soc, self.R_soc, self.R1_table)
        tau1 = np.interp(soc, self.R_soc, self.tau1_table)
        return ocv, r0, r1, tau1

    def predict(self, I):
        """
        State prediction step.
        Args:
            I: Current (A). Negative = discharge (convention from Simscape).
        """
        soc, v_rc = self.x
        _, _, r1, tau1 = self._get_params(soc)
        dt = self.dt

        # State transition
        exp_term = np.exp(-dt / tau1)
        soc_new = soc + I * dt / (self.Q_ah * 3600)
        v_rc_new = v_rc * exp_term + I * r1 * (1 - exp_term)

        # Jacobian F = df/dx
        F = np.array([
            [1.0, 0.0],
            [0.0, exp_term],
        ])

        self.x = np.array([soc_new, v_rc_new])
        self.P = F @ self.P @ F.T + self.Q

    def update(self, V_meas, I):
        """
        Measurement update step.
        Args:
            V_meas: Measured terminal voltage (V).
            I: Current (A).
        Returns:
            residual: V_meas - V_pred (scalar).
        """
        soc, v_rc = self.x
        ocv, r0, _, _ = self._get_params(soc)

        # Predicted measurement
        V_pred = ocv - I * r0 - v_rc

        # Residual (innovation)
        residual = V_meas - V_pred

        # Jacobian H = dh/dx
        dOCV_dSOC = _interp_deriv(soc, self.OCV_soc, self.OCV_table)
        dR0_dSOC = _interp_deriv(soc, self.R_soc, self.R0_table)
        H = np.array([[dOCV_dSOC - I * dR0_dSOC, -1.0]])

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_scalar
        S_inv = 1.0 / S[0, 0]

        # Kalman gain
        K = self.P @ H.T * S_inv  # (2, 1)

        # State update
        self.x = self.x + (K @ np.array([[residual]])).flatten()
        self.P = (np.eye(2) - K @ H) @ self.P

        # Clamp SOC to [0, 1]
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)

        return residual

    def run(self, V_array, I=None):
        """
        Run EKF over entire voltage time series.

        Args:
            V_array: Terminal voltage array (N,) in physical units (V).
            I: Current (A). Scalar (constant) or array. Default: I_LOAD from config.

        Returns:
            dict with:
                residuals: (N,) measurement residuals.
                soc_ekf: (N,) EKF SOC estimates (voltage-based).
                soc_cc: (N,) Coulomb counting SOC (current-based only).
                delta_soc: (N,) SOC inconsistency = soc_cc - soc_ekf.
        """
        if I is None:
            I = I_LOAD
        I_scalar = isinstance(I, (int, float))

        self.reset()
        N = len(V_array)
        residuals = np.zeros(N)
        soc_ekf = np.zeros(N)
        soc_cc = np.zeros(N)

        # Coulomb counting initial state (same as EKF)
        soc_cc_val = EKF_SOC_INIT

        for k in range(N):
            i_k = I if I_scalar else I[k]
            self.predict(i_k)
            residuals[k] = self.update(V_array[k], i_k)

            soc_ekf[k] = self.x[0]

            # Coulomb counting (independent of voltage, uses current only)
            soc_cc_val += i_k * self.dt / (self.Q_ah * 3600)
            soc_cc[k] = soc_cc_val

        return {
            "residuals": residuals,
            "soc_ekf": soc_ekf,
            "soc_cc": soc_cc,
            "delta_soc": soc_cc - soc_ekf,
        }

