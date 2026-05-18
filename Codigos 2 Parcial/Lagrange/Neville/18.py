"""
Coeficiente de arrastre cD  —  Polinomio de Neville (4 vecinos más cercanos)
Problema 18: Mismo dataset que Prob. 17, pero con interpolación polinomial
usando los 4 puntos más cercanos (en escala log-log)

Datos: Re = [0.2, 2, 20, 200, 2000, 20000]
       cD = [103, 13.9, 2.72, 0.800, 0.401, 0.433]

Transformación: u = log₁₀(Re),  v = log₁₀(cD)

Resultados Neville 4pts:
  cD(Re=5)    ≈ 6.933   vs Spline: 6.902
  cD(Re=50)   ≈ 1.581   vs Spline: 1.591
  cD(Re=500)  ≈ 0.563   vs Spline: 0.557
  cD(Re=5000) ≈ 0.372   vs Spline: 0.387
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, RadioButtons
from matplotlib.animation import FuncAnimation

# ── Paleta ─────────────────────────────────────────────────────────────────
BG_FIG    = "#1A1A2E"
BG_AX     = "#16213E"
BG_TABLE  = "#0F3460"
GRID_CLR  = "#2A2A4A"
SPINE_CLR = "#3A3A5C"
TXT_PRI   = "#E2E2F0"
TXT_SEC   = "#8888AA"
ACCENT    = "#4FC3F7"

TC = ["#FF5252", "#CE93D8", "#66BB6A", "#FFB74D"]

# ── Datos ──────────────────────────────────────────────────────────────────
Re_data = np.array([0.2, 2., 20., 200., 2000., 20000.])
cD_data = np.array([103., 13.9, 2.72, 0.800, 0.401, 0.433])
lRe     = np.log10(Re_data)
lcD     = np.log10(cD_data)
n       = len(Re_data)

TARGETS   = [5., 50., 500., 5000.]
SPLINE    = {5.: 6.90160, 50.: 1.59084, 500.: 0.55743, 5000.: 0.38682}

# ── Neville ────────────────────────────────────────────────────────────────
def neville_table(xs, ys, x):
    n = len(xs)
    Q = np.zeros((n, n))
    Q[:, 0] = ys.copy()
    for j in range(1, n):
        for i in range(j, n):
            Q[i, j] = ((x - xs[i-j])*Q[i, j-1] -
                       (x - xs[i])*Q[i-1, j-1]) / (xs[i] - xs[i-j])
    return Q

def get_4nn(lRe_t):
    """4 vecinos más cercanos en escala log."""
    dists = np.abs(lRe - lRe_t)
    idx4  = np.sort(np.argsort(dists)[:4])
    return idx4, lRe[idx4], lcD[idx4]

# Precompute
INFO = {}
for Re_t in TARGETS:
    lRe_t = np.log10(Re_t)
    idx4, xs4, ys4 = get_4nn(lRe_t)
    Q     = neville_table(xs4, ys4, lRe_t)
    lcD_t = Q[-1, -1]
    cD_t  = 10**lcD_t
    INFO[Re_t] = {"idx4": idx4, "xs4": xs4, "ys4": ys4,
                  "Q": Q, "lcD_t": lcD_t, "cD_t": cD_t}

# Curva Neville completa (todos los puntos) para fondo
def neville_all(lRe_t):
    Q = neville_table(lRe, lcD, lRe_t)
    return Q[-1, -1]

lRe_r_all = np.linspace(lRe[0], lRe[-1], 600)
lcD_r_all = np.array([neville_all(v) for v in lRe_r_all])

# Curvas Neville 4pts por target (una por tramo activo)
def neville_4pt_curve(Re_t, lRe_arr):
    inf = INFO[Re_t]
    xs4, ys4 = inf["xs4"], inf["ys4"]
    return np.array([10**neville_table(xs4, ys4, v)[-1,-1] for v in np.atleast_1d(lRe_arr)])

# ── Figura ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.color": TXT_PRI, "axes.labelcolor": TXT_SEC,
    "xtick.color": TXT_SEC, "ytick.color": TXT_SEC, "font.size": 9,
})

fig = plt.figure(figsize=(14, 8), facecolor=BG_FIG)
fig.text(0.5, 0.975,
         "cD(Re)  —  Neville 4 vecinos (log-log)  vs  Spline Natural",
         ha="center", va="top", fontsize=12, color=ACCENT)

gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1],
                      left=0.06, right=0.97,
                      bottom=0.21, top=0.94, wspace=0.06)
ax  = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])

# Eje log-log
ax.set_facecolor(BG_AX)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.15, 25000); ax.set_ylim(0.3, 200)
ax.set_xlabel("Re  (número de Reynolds)", fontsize=11)
ax.set_ylabel("cD  (coeficiente de arrastre)", fontsize=11)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)
ax.tick_params(colors=TXT_SEC, labelsize=8.5, which="both")
ax.grid(True, color=GRID_CLR, lw=0.5, which="both", zorder=0)

# Líneas de targets
for Re_t, tc in zip(TARGETS, TC):
    ax.axvline(Re_t, color=tc, lw=0.9, ls="--", alpha=0.7, zorder=2)
    ax.text(Re_t*1.06, 0.38, f"Re={Re_t:.0f}", fontsize=7.5, color=tc)

# Puntos de datos
ax.scatter(Re_data, cD_data, zorder=7, s=90, color=ACCENT,
           edgecolors="white", linewidths=1.8)
for ri, ci in zip(Re_data, cD_data):
    ax.annotate(f"({ri:.0f},{ci})", (ri, ci),
                textcoords="offset points", xytext=(6, 6),
                fontsize=7.5, color=ACCENT)

# Curva Neville global (fondo, tenue)
all_line, = ax.plot(10**lRe_r_all, 10**lcD_r_all,
                    color=TXT_SEC, lw=1.2, ls=":", alpha=0, zorder=3,
                    label="Neville global (6 pts)")

# Curvas Neville 4pt (una por target, animadas)
nev4_lines = [ax.plot([], [], color=tc, lw=2.6, zorder=5,
                      ls="-", label=f"Neville 4pt Re={Re_t:.0f}")[0]
              for Re_t, tc in zip(TARGETS, TC)]

# Puntos usados resaltados (scatter por target)
used_scats = [ax.scatter([], [], s=110, color=tc, zorder=8,
                          edgecolors="white", linewidths=1.5, marker="D")
              for tc in TC]

# Puntos de evaluación
eval_dots = [ax.scatter([], [], s=160, color=tc, zorder=9,
                         edgecolors="white", linewidths=2, marker="*")
             for tc in TC]
eval_txts = [ax.text(0, 0, "", fontsize=8.5, color=tc) for tc in TC]

status_txt = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                     fontsize=10, va="top", color=ACCENT)

ax.legend(loc="upper right", fontsize=7.5, framealpha=0.35,
          facecolor=BG_AX, edgecolor=SPINE_CLR, labelcolor=TXT_PRI, ncol=1)

# Panel tabla
axt.set_facecolor(BG_TABLE)
for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
axt.set_axis_off()

# ── Tabla ──────────────────────────────────────────────────────────────────
def draw_table(target_idx=None):
    axt.cla(); axt.set_facecolor(BG_TABLE)
    for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
    axt.set_xlim(0,1); axt.set_ylim(0,1); axt.set_axis_off()

    if target_idx is None:
        axt.text(0.5, 0.52,
                 "Presiona  ▶  Animar\npara ver la interpolación",
                 ha="center", va="center", fontsize=11,
                 color=TXT_SEC, linespacing=2.2, transform=axt.transAxes)
        fig.canvas.draw_idle(); return

    Re_t = TARGETS[target_idx]
    tc   = TC[target_idx]
    inf  = INFO[Re_t]
    Q    = inf["Q"]; xs4 = inf["xs4"]; ys4 = inf["ys4"]
    cD_t = inf["cD_t"]; lcD_t = inf["lcD_t"]
    lRe_t= np.log10(Re_t)
    idx4 = inf["idx4"]

    axt.text(0.5, 0.975, f"Neville 4pts  —  Re = {Re_t:.0f}",
             ha="center", va="top", fontsize=9.5, color=tc,
             transform=axt.transAxes)

    # Tabla Q 4×4
    col_labels = ["log(Re)", "log(cD)", "ord 1", "ord 2", "ord 3"]
    col_w      = [0.18, 0.18, 0.20, 0.20, 0.24]

    n4      = 4
    total_h = 0.32
    row_h   = total_h / (n4 + 1.2)
    y_start = 0.89

    def draw_row(cells, y_c, is_hdr=False, hi_col=-1):
        if is_hdr:
            rect = mpatches.FancyBboxPatch(
                (0.01,y_c-row_h*0.47),0.98,row_h*0.94,
                boxstyle="round,pad=0.003",facecolor="#1A3A5C",
                edgecolor="none",transform=axt.transAxes,zorder=1,clip_on=False)
            axt.add_patch(rect)
        x = 0.015
        for ci,(cell,cw) in enumerate(zip(cells,col_w)):
            val=str(cell)
            if val=="—": clr=SPINE_CLR
            elif is_hdr: clr=TXT_SEC
            elif ci==hi_col:
                box=mpatches.FancyBboxPatch(
                    (x+0.004,y_c-row_h*0.42),cw-0.008,row_h*0.84,
                    boxstyle="round,pad=0.002",
                    facecolor="#0D47A1",edgecolor=tc,linewidth=0.8,
                    transform=axt.transAxes,zorder=2,clip_on=False)
                axt.add_patch(box); clr="#FFFFFF"
            elif not is_hdr and ci>=2:
                cols=["#FF7043","#66BB6A","#CE93D8"]
                clr=cols[min(ci-2,2)]
            else: clr=TXT_PRI
            fs=6.8 if is_hdr else 8.0
            fw="bold" if ci==hi_col and not is_hdr else "normal"
            axt.text(x+cw/2,y_c,val,ha="center",va="center",
                     fontsize=fs,color=clr,fontweight=fw,
                     transform=axt.transAxes,zorder=3,clip_on=False)
            x+=cw

    y_h=y_start-row_h*0.5
    draw_row(col_labels,y_h,is_hdr=True)
    axt.plot([0.01,0.99],[y_h-row_h*0.52]*2,
             color=SPINE_CLR,lw=0.8,transform=axt.transAxes,zorder=4)

    for i in range(n4):
        y_r=y_start-row_h*(i+1.5)
        hi=n4 if i==n4-1 else -1
        cells=[f"{xs4[i]:.4f}",f"{ys4[i]:.5f}"]
        for j in range(1,n4):
            cells.append(f"{Q[i,j]:.5f}" if i>=j else "—")
        draw_row(cells,y_r,hi_col=hi)

    axt.plot([0.01,0.99],[y_start-row_h*(n4+1.0)]*2,
             color=SPINE_CLR,lw=0.5,ls="--",transform=axt.transAxes,zorder=4)

    # Puntos usados
    y_pts=y_start-row_h*(n4+1.6)
    r_h2=0.055
    axt.text(0.5,y_pts,"Puntos usados (log₁₀):",
             ha="center",va="center",fontsize=7.5,color=TXT_SEC,
             transform=axt.transAxes,zorder=3,clip_on=False)
    y_pts-=r_h2*0.9
    for k,(xs_k,ys_k,idx_k) in enumerate(zip(xs4,ys4,idx4)):
        Re_k=Re_data[idx_k]; cD_k=cD_data[idx_k]
        axt.text(0.5,y_pts,
                 f"Re={Re_k:.0f}  log(Re)={xs_k:.4f}  log(cD)={ys_k:.5f}",
                 ha="center",va="center",fontsize=7.5,color=TXT_PRI,
                 transform=axt.transAxes,zorder=3,clip_on=False)
        y_pts-=r_h2*0.8

    axt.plot([0.01,0.99],[y_pts-r_h2*0.1]*2,
             color=SPINE_CLR,lw=0.5,ls="--",transform=axt.transAxes,zorder=4)

    # Bloque comparativo
    y_cmp=y_pts-r_h2*0.6
    axt.text(0.5,y_cmp,"Comparación con Spline Natural:",
             ha="center",va="center",fontsize=7.5,color=TXT_SEC,
             transform=axt.transAxes,zorder=3,clip_on=False)
    y_cmp-=r_h2*0.85
    r_h3=0.065
    col_hdr=["Re","Neville 4pts","Spline","Δ"]
    cw3=[0.18,0.27,0.27,0.28]; x3=0.015
    for lbl,cw in zip(col_hdr,cw3):
        axt.text(x3+cw/2,y_cmp,lbl,ha="center",va="center",
                 fontsize=7,color=TXT_SEC,transform=axt.transAxes,
                 zorder=3,clip_on=False); x3+=cw
    y_cmp-=r_h3
    axt.plot([0.02,0.98],[y_cmp+r_h3*0.2]*2,color=SPINE_CLR,lw=0.6,
             transform=axt.transAxes,zorder=4)
    for Re_ti,tc2 in zip(TARGETS,TC):
        nv=INFO[Re_ti]["cD_t"]; sp=SPLINE[Re_ti]
        is_cur=(Re_ti==Re_t)
        if is_cur:
            rect=mpatches.FancyBboxPatch(
                (0.02,y_cmp-r_h3*0.46),0.96,r_h3*0.92,
                boxstyle="round,pad=0.003",facecolor="#0A2A50",
                edgecolor="none",transform=axt.transAxes,zorder=1,clip_on=False)
            axt.add_patch(rect)
        row_vals=[f"{Re_ti:.0f}",f"{nv:.4f}",f"{sp:.4f}",f"{abs(nv-sp):.4f}"]
        x3=0.015
        for rv,cw in zip(row_vals,cw3):
            axt.text(x3+cw/2,y_cmp,rv,ha="center",va="center",
                     fontsize=8.0 if is_cur else 7.5,
                     color=tc2 if rv==f"{nv:.4f}" else TXT_PRI,
                     fontweight="bold" if is_cur and rv==f"{nv:.4f}" else "normal",
                     transform=axt.transAxes,zorder=3,clip_on=False); x3+=cw
        y_cmp-=r_h3

    # Caja final
    y_box=y_cmp-r_h3*0.4
    box=mpatches.FancyBboxPatch(
        (0.03,y_box-0.04),0.94,0.072,
        boxstyle="round,pad=0.01",
        facecolor="#0D47A1",edgecolor=tc,linewidth=1.2,
        transform=axt.transAxes,zorder=4,clip_on=False)
    axt.add_patch(box)
    axt.text(0.5,y_box-0.004,
             f"cD(Re={Re_t:.0f})  =  {cD_t:.5f}",
             ha="center",va="center",fontsize=11,
             color="#FFFFFF",fontweight="bold",
             transform=axt.transAxes,zorder=5,clip_on=False)
    fig.canvas.draw_idle()

# ── Animación ──────────────────────────────────────────────────────────────
N_FRAMES   = 80
MSG_FRAMES = 25
state = {"running": False, "anim": None}

BUILD = [
    "Transformando a escala log₁₀: u=log(Re), v=log(cD)...",
    "Seleccionando 4 vecinos más cercanos para cada objetivo...",
    "Curva Neville global (referencia, 6 pts)...",
    "Neville 4pts — Re=5  (vecinos: 0.2, 2, 20, 200)...",
    "Neville 4pts — Re=50  (vecinos: 2, 20, 200, 2000)...",
    "Neville 4pts — Re=500  (vecinos: 20, 200, 2000, 20000)...",
    "Neville 4pts — Re=5000  (vecinos: 20, 200, 2000, 20000)...",
    "Marcando puntos de evaluación...",
]

def clear_all():
    for ln in nev4_lines: ln.set_data([],[])
    all_line.set_alpha(0)
    for d in used_scats+eval_dots: d.set_offsets(np.empty((0,2)))
    for t in eval_txts: t.set_text("")
    status_txt.set_text("")
    draw_table(None)

def run_animation(event=None):
    if state["running"]: return
    clear_all(); state["running"]=True

    n_msg=2; n_cur=4; n_fin=1
    total=n_msg*MSG_FRAMES + MSG_FRAMES + n_cur*(MSG_FRAMES+N_FRAMES) + n_fin*MSG_FRAMES

    # Rango x para cada curva Neville 4pts
    lRe_ranges = []
    for Re_t in TARGETS:
        xs4 = INFO[Re_t]["xs4"]
        lRe_ranges.append(np.linspace(xs4[0], xs4[-1], 200))

    def update(frame):
        # Bloque 0,1: mensajes
        # Bloque 2: curva global
        # Bloques 3..6: curvas 4pt
        # Bloque 7: marcar evaluaciones
        bk0 = n_msg*MSG_FRAMES
        bk1 = bk0 + MSG_FRAMES  # fin curva global
        bk2 = bk1 + n_cur*(MSG_FRAMES+N_FRAMES)

        if frame < bk0:
            ph=frame//MSG_FRAMES; status_txt.set_text(BUILD[ph])
        elif frame < bk1:
            status_txt.set_text(BUILD[2])
            cf=frame-bk0; n_pts=max(2,int((cf+1)/MSG_FRAMES*len(lRe_r_all)))
            all_line.set_data(10**lRe_r_all[:n_pts],10**lcD_r_all[:n_pts])
            all_line.set_alpha(0.4)
        elif frame < bk2:
            f2=frame-bk1; ph_len=MSG_FRAMES+N_FRAMES
            ph2=f2//ph_len; loc=f2%ph_len
            if ph2<n_cur:
                Re_t=TARGETS[ph2]
                status_txt.set_text(BUILD[3+ph2])
                xs4=INFO[Re_t]["xs4"]
                used_scats[ph2].set_offsets(
                    np.column_stack([10**INFO[Re_t]["xs4"],
                                     10**INFO[Re_t]["ys4"]]))
                if loc>=MSG_FRAMES:
                    cf=loc-MSG_FRAMES
                    n_pts=max(2,int((cf+1)/N_FRAMES*len(lRe_ranges[ph2])))
                    lxp=lRe_ranges[ph2][:n_pts]
                    yp=neville_4pt_curve(Re_t,lxp)
                    nev4_lines[ph2].set_data(10**lxp, yp)
        else:
            status_txt.set_text(BUILD[-1])
            for ti,Re_t in enumerate(TARGETS):
                cD_t=INFO[Re_t]["cD_t"]
                eval_dots[ti].set_offsets([[Re_t,cD_t]])
                eval_txts[ti].set_text(f"  {cD_t:.3f}")
                eval_txts[ti].set_x(Re_t*1.1); eval_txts[ti].set_y(cD_t*1.15)

        if frame==total-1:
            r=INFO; sp=SPLINE
            status_txt.set_text(
                f"✓  cD: {r[5.]['cD_t']:.3f} | {r[50.]['cD_t']:.3f} | "
                f"{r[500.]['cD_t']:.3f} | {r[5000.]['cD_t']:.3f}")
            draw_table(0); state["running"]=False
        return tuple(nev4_lines)+(all_line,)

    state["anim"]=FuncAnimation(fig,update,frames=total,
                                 interval=18,blit=False,repeat=False)
    fig.canvas.draw_idle()

def reset_animation(event=None):
    if state["anim"]:
        state["anim"].event_source.stop(); state["anim"]=None
    state["running"]=False; clear_all(); fig.canvas.draw_idle()

def show_tab(label):
    idx={"Re=5":0,"Re=50":1,"Re=500":2,"Re=5000":3}.get(label,0)
    draw_table(idx)

# ── Widgets ────────────────────────────────────────────────────────────────
ax_play =fig.add_axes([0.06,0.08,0.14,0.07])
ax_reset=fig.add_axes([0.22,0.08,0.12,0.07])
ax_radio=fig.add_axes([0.38,0.03,0.28,0.14])

for wax in [ax_play,ax_reset,ax_radio]:
    wax.set_facecolor(BG_FIG)
    for sp in wax.spines.values(): sp.set_color(SPINE_CLR)

btn_play =Button(ax_play,"▶  Animar",color="#0D2B55",hovercolor="#1A3A6E")
btn_reset=Button(ax_reset,"↺ Reset",color="#1A1A3E",hovercolor="#2A2A5E")
radio    =RadioButtons(ax_radio,("Re=5","Re=50","Re=500","Re=5000"),
                       activecolor=ACCENT)

for b in [btn_play,btn_reset]:
    b.label.set_color(TXT_PRI); b.label.set_fontsize(10)
for lbl in radio.labels:
    lbl.set_color(TXT_PRI); lbl.set_fontsize(9.5)

btn_play.on_clicked(run_animation)
btn_reset.on_clicked(reset_animation)
radio.on_clicked(show_tab)

draw_table(None)
plt.show()