"""
=============================================================================
  Simulador Interactivo — Método de Euler · Modelo Poblacional
=============================================================================
  dP/dt = P·(r − a·P)        Ecuación logística
  Controles: sliders + recuadros de texto para valores exactos.
=============================================================================
  Requiere:  pip install matplotlib numpy
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as ticker

# ══════════════════════════════ ESTILO OSCURO ════════════════════════════════
plt.rcParams.update({
    "figure.facecolor":  "#0f0f0f",
    "axes.facecolor":    "#1a1a2e",
    "axes.edgecolor":    "#333",
    "axes.labelcolor":   "#aaa",
    "axes.grid":         True,
    "grid.color":        "#2a2a3a",
    "grid.alpha":        0.6,
    "xtick.color":       "#888",
    "ytick.color":       "#888",
    "text.color":        "#ddd",
    "font.family":       "monospace",
    "font.size":         10,
    "legend.facecolor":  "#1a1a2e",
    "legend.edgecolor":  "#444",
    "legend.fontsize":   9,
})

# ══════════════════════════ PARÁMETROS INICIALES ═════════════════════════════
a       = 1e-7
r0      = 0.10
P0_init = 5000
h0      = 0.5
T_FINAL = 150
FPS0    = 30


# ══════════════════════════ FUNCIONES DEL MODELO ═════════════════════════════
def euler(r, P0, h, tf):
    n = int(np.ceil(tf / h))
    t = np.zeros(n + 1)
    P = np.zeros(n + 1)
    t[0], P[0] = 0.0, P0
    for i in range(n):
        P[i + 1] = P[i] + h * P[i] * (r - a * P[i])
        t[i + 1] = t[i] + h
    return t, P


def analitica(t, r, P0):
    K = r / a
    C = (K - P0) / P0
    return K / (1 + C * np.exp(-r * t))


# ══════════════════════════ CREAR FIGURA ═════════════════════════════════════
fig = plt.figure(figsize=(15, 8.5))
fig.canvas.manager.set_window_title("Método de Euler — Modelo Poblacional")

ax = fig.add_axes([0.07, 0.35, 0.62, 0.56])

t_euler, P_euler = euler(r0, P0_init, h0, T_FINAL)
t_fine = np.linspace(0, T_FINAL, 2000)
P_anal = analitica(t_fine, r0, P0_init)
K0 = r0 / a

line_anal, = ax.plot(t_fine, P_anal, color="#f59e0b", linewidth=2.5,
                     label="Analítica", zorder=4)
line_euler, = ax.plot([], [], color="#34d399", linewidth=1.5,
                      label="Euler", zorder=5)
dots_euler, = ax.plot([], [], "o", color="#34d399", markersize=4,
                      markeredgewidth=0, zorder=6)
hline_K = ax.axhline(K0, color="#555", linestyle=":", linewidth=1.2)
txt_K = ax.text(T_FINAL - 1, K0 * 1.02, f"K = {K0:,.0f}", ha="right",
                fontsize=9, color="#777", fontstyle="italic")

P5_val = analitica(np.array([5.0]), r0, P0_init)[0]
dot_5, = ax.plot(5, P5_val, "o", color="#e67e22", markersize=9,
                 markeredgecolor="white", markeredgewidth=1.2, zorder=7)
txt_5 = ax.annotate(f"P(5) ≈ {P5_val:,.0f}", xy=(5, P5_val),
                     xytext=(20, P5_val + K0 * 0.08),
                     fontsize=10, fontweight="bold", color="#e67e22",
                     arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.3))

ax.set_xlim(0, T_FINAL)
ax.set_ylim(0, K0 * 1.15)
ax.set_xlabel("Tiempo  t  (meses)", fontsize=12)
ax.set_ylabel("Población  P(t)", fontsize=12)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(loc="center right")
title_text = ax.set_title(
    f"Población vs Tiempo  ·  r = {r0:.2f}  ·  h = {h0}  ·  P₀ = {P0_init:,}",
    fontsize=12, fontweight="bold", color="#34d399", pad=10
)


# ══════════════════════════ PANEL DERECHO — CONTROLES ════════════════════════
PX     = 0.73
PW_SL  = 0.14
PW_TB  = 0.07
PH     = 0.022
GAP_TB = PX + PW_SL + 0.015

slider_color = "#34d399"
track_color  = "#1a1a2e"


def add_label(y, text):
    fig.text(PX, y + PH + 0.005, text, fontsize=8, color="#34d399",
             fontweight="bold", family="monospace",
             transform=fig.transFigure)


def make_textbox(rect, initial_text):
    """Crea un TextBox compatible con matplotlib 3.5+."""
    ax_tb = fig.add_axes(rect, facecolor="#1a1a2e")
    for spine in ax_tb.spines.values():
        spine.set_color("#34d39966")
    tb = TextBox(ax_tb, "", initial=initial_text)
    tb.text_disp.set_color("#34d399")
    tb.text_disp.set_fontsize(10)
    return tb


# ---------- Tasa r ----------
add_label(0.82, "TASA  r")
ax_sl_r = fig.add_axes([PX, 0.82, PW_SL, PH], facecolor=track_color)
sl_rate = Slider(ax_sl_r, "", 0.01, 1.0, valinit=r0,
                 valstep=0.01, color=slider_color)
sl_rate.valtext.set_visible(False)
tb_rate = make_textbox([GAP_TB, 0.815, PW_TB, 0.030], f"{r0:.2f}")

# ---------- P₀ ----------
add_label(0.74, "POBLACIÓN INICIAL  P₀")
ax_sl_p = fig.add_axes([PX, 0.74, PW_SL, PH], facecolor=track_color)
sl_pop = Slider(ax_sl_p, "", 100, 500_000, valinit=P0_init,
                valstep=100, color=slider_color)
sl_pop.valtext.set_visible(False)
tb_pop = make_textbox([GAP_TB, 0.735, PW_TB, 0.030], f"{P0_init}")

# ---------- Paso h ----------
add_label(0.66, "PASO  h  (meses)")
ax_sl_h = fig.add_axes([PX, 0.66, PW_SL, PH], facecolor=track_color)
sl_h = Slider(ax_sl_h, "", 0.1, 5.0, valinit=h0,
              valstep=0.1, color=slider_color)
sl_h.valtext.set_visible(False)
tb_h = make_textbox([GAP_TB, 0.655, PW_TB, 0.030], f"{h0}")

# ---------- Tiempo final ----------
add_label(0.58, "TIEMPO FINAL  (meses)")
ax_sl_t = fig.add_axes([PX, 0.58, PW_SL, PH], facecolor=track_color)
sl_tf = Slider(ax_sl_t, "", 20, 500, valinit=T_FINAL,
               valstep=5, color=slider_color)
sl_tf.valtext.set_visible(False)
tb_tf = make_textbox([GAP_TB, 0.575, PW_TB, 0.030], f"{T_FINAL}")

# ---------- FPS ----------
add_label(0.50, "VELOCIDAD  FPS")
ax_sl_fps = fig.add_axes([PX, 0.50, PW_SL, PH], facecolor=track_color)
sl_fps = Slider(ax_sl_fps, "", 5, 120, valinit=FPS0,
                valstep=5, color=slider_color)
sl_fps.valtext.set_visible(False)
tb_fps = make_textbox([GAP_TB, 0.495, PW_TB, 0.030], f"{FPS0}")


# ══════════════════════════ BOTONES ══════════════════════════════════════════
ax_play  = fig.add_axes([PX,        0.42, 0.07, 0.045])
ax_reset = fig.add_axes([PX + 0.08, 0.42, 0.07, 0.045])
ax_clear = fig.add_axes([PX + 0.16, 0.42, 0.07, 0.045])

btn_play  = Button(ax_play,  "▶ Play",  color="#1a3a2a", hovercolor="#34d399")
btn_reset = Button(ax_reset, "Reset",   color="#2a2a3a", hovercolor="#555")
btn_clear = Button(ax_clear, "Clear",   color="#3a1a1a", hovercolor="#d34343")

for b in [btn_play, btn_reset, btn_clear]:
    b.label.set_fontweight("bold")
    b.label.set_fontsize(10)

# ── Info box ──
ax_info = fig.add_axes([PX, 0.35, 0.235, 0.055], facecolor="#0d1f15")
ax_info.set_xticks([]); ax_info.set_yticks([])
for spine in ax_info.spines.values():
    spine.set_color("#34d39944")
info_txt = ax_info.text(0.05, 0.5, "", transform=ax_info.transAxes,
                        fontsize=9, color="#34d399", verticalalignment="center",
                        family="monospace")


# ══════════════════════════ LÓGICA PRINCIPAL ═════════════════════════════════
_updating = {"flag": False}

state = {
    "anim": None,
    "t_data": t_euler,
    "P_data": P_euler,
    "running": False,
}


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def stop_anim():
    """Detiene la animación de forma segura (corrige AttributeError)."""
    if state["anim"] is not None:
        try:
            if state["anim"].event_source is not None:
                state["anim"].event_source.stop()
        except Exception:
            pass
        state["anim"] = None
    state["running"] = False


def recompute():
    r  = sl_rate.val
    P0 = int(sl_pop.val)
    h  = sl_h.val
    tf = int(sl_tf.val)
    K  = r / a

    t_e, P_e = euler(r, P0, h, tf)
    state["t_data"] = t_e
    state["P_data"] = P_e

    t_a = np.linspace(0, tf, 2000)
    P_a = analitica(t_a, r, P0)

    line_anal.set_data(t_a, P_a)
    hline_K.set_ydata([K, K])
    txt_K.set_position((tf - 1, K * 1.02))
    txt_K.set_text(f"K = {K:,.0f}")

    P5 = analitica(np.array([5.0]), r, P0)[0]
    dot_5.set_data([5], [P5])
    txt_5.xy = (5, P5)
    txt_5.set_position((20, P5 + K * 0.08))
    txt_5.set_text(f"P(5) ≈ {P5:,.0f}")

    ax.set_xlim(0, tf)
    ax.set_ylim(0, max(K * 1.15, np.max(P_e) * 1.1))
    title_text.set_text(
        f"Población vs Tiempo  ·  r = {r:.2f}  ·  h = {h}  ·  P₀ = {P0:,}"
    )
    info_txt.set_text(
        f" K = {K:>12,.0f}   P(5) = {P5:>10,.0f}   Pts = {len(t_e):>6,}"
    )
    return t_e, P_e


def show_full():
    t_e, P_e = recompute()
    stride = max(1, len(t_e) // 300)
    line_euler.set_data(t_e, P_e)
    dots_euler.set_data(t_e[::stride], P_e[::stride])
    fig.canvas.draw_idle()


# ─── Slider → TextBox ───────────────────────────────────────────────────────
def sync_textboxes():
    if _updating["flag"]:
        return
    _updating["flag"] = True
    tb_rate.set_val(f"{sl_rate.val:.2f}")
    tb_pop.set_val(f"{int(sl_pop.val)}")
    tb_h.set_val(f"{sl_h.val:.1f}")
    tb_tf.set_val(f"{int(sl_tf.val)}")
    tb_fps.set_val(f"{int(sl_fps.val)}")
    _updating["flag"] = False


def on_slider_change(val):
    if _updating["flag"]:
        return
    stop_anim()
    sync_textboxes()
    show_full()


sl_rate.on_changed(on_slider_change)
sl_pop.on_changed(on_slider_change)
sl_h.on_changed(on_slider_change)
sl_tf.on_changed(on_slider_change)


# ─── TextBox → Slider ───────────────────────────────────────────────────────
def on_tb_rate(text):
    if _updating["flag"]:
        return
    try:
        v = clamp(float(text), 0.01, 1.0)
        _updating["flag"] = True
        sl_rate.set_val(v)
        _updating["flag"] = False
        stop_anim(); show_full()
    except ValueError:
        pass


def on_tb_pop(text):
    if _updating["flag"]:
        return
    try:
        v = clamp(int(float(text)), 100, 500_000)
        _updating["flag"] = True
        sl_pop.set_val(v)
        _updating["flag"] = False
        stop_anim(); show_full()
    except ValueError:
        pass


def on_tb_h(text):
    if _updating["flag"]:
        return
    try:
        v = clamp(float(text), 0.1, 5.0)
        _updating["flag"] = True
        sl_h.set_val(v)
        _updating["flag"] = False
        stop_anim(); show_full()
    except ValueError:
        pass


def on_tb_tf(text):
    if _updating["flag"]:
        return
    try:
        v = clamp(int(float(text)), 20, 500)
        _updating["flag"] = True
        sl_tf.set_val(v)
        _updating["flag"] = False
        stop_anim(); show_full()
    except ValueError:
        pass


def on_tb_fps(text):
    if _updating["flag"]:
        return
    try:
        v = clamp(int(float(text)), 5, 120)
        _updating["flag"] = True
        sl_fps.set_val(v)
        _updating["flag"] = False
    except ValueError:
        pass


tb_rate.on_submit(on_tb_rate)
tb_pop.on_submit(on_tb_pop)
tb_h.on_submit(on_tb_h)
tb_tf.on_submit(on_tb_tf)
tb_fps.on_submit(on_tb_fps)


# ══════════════════════════ ANIMACIÓN ════════════════════════════════════════
def on_play(event):
    stop_anim()
    t_e, P_e = recompute()
    line_euler.set_data([], [])
    dots_euler.set_data([], [])

    total = len(t_e)
    step = max(1, total // 250)

    def animate(frame):
        idx = min(frame * step, total - 1)
        line_euler.set_data(t_e[:idx + 1], P_e[:idx + 1])
        s = max(1, (idx + 1) // 60)
        dots_euler.set_data(t_e[:idx + 1:s], P_e[:idx + 1:s])
        if idx >= total - 1:
            state["running"] = False
        return line_euler, dots_euler

    fps = int(sl_fps.val)
    n_frames = (total // step) + 1
    state["anim"] = FuncAnimation(fig, animate, frames=n_frames,
                                  interval=1000 // fps, blit=True, repeat=False)
    state["running"] = True
    fig.canvas.draw_idle()


def on_reset(event):
    stop_anim()
    show_full()


def on_clear(event):
    stop_anim()
    recompute()
    line_euler.set_data([], [])
    dots_euler.set_data([], [])
    fig.canvas.draw_idle()


btn_play.on_clicked(on_play)
btn_reset.on_clicked(on_reset)
btn_clear.on_clicked(on_clear)

# ── Estado inicial ──
show_full()
sync_textboxes()

plt.show()