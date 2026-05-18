"""
Spline Cúbico con Extremos Parabólicos (Condición k₀=k₁, k_{n-1}=kₙ)
Datos (Ejemplo 3.6): x=[1,2,3,4,5], y=[13,15,12,9,13]

Condición: S''=constante en tramos extremos → M₀=M₁, M₃=M₄
Sistema 3×3:  [5 1 0][M₁]   [-30]
              [1 4 1][M₂] =  [ 0]
              [0 1 5][M₃]   [ 42]

Momentos: M = [-5.8667, -5.8667, -0.6667, 8.5333, 8.5333]
vs. Natural:   M = [0, -7.2857, -0.8571, 10.7143, 0]

S(3.4) parabólico  ≈ 10.3648
S(3.4) natural     ≈ 10.2549
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

SEG_PAR  = ["#FF7043","#66BB6A","#CE93D8","#FFB74D"]  # parabólico
SEG_NAT  = ["#FF7043","#66BB6A","#CE93D8","#FFB74D"]  # natural (mismo esquema, más tenue)

# ── Datos ──────────────────────────────────────────────────────────────────
xs = np.array([1.,2.,3.,4.,5.]); ys = np.array([13.,15.,12.,9.,13.])
n  = len(xs); h = np.diff(xs)
X_EVAL = 3.4

# Momentos parabólico
M_par = np.array([-5.866667,-5.866667,-0.666667,8.533333,8.533333])
# Momentos natural
M_nat = np.array([0.,-7.285714,-0.857143,10.714286,0.])

def spline_eval(xv, M):
    i = min(max(np.searchsorted(xs,xv,'right')-1,0),n-2)
    hi=h[i]; xi,xi1=xs[i],xs[i+1]; Mi,Mi1=M[i],M[i+1]; yi,yi1=ys[i],ys[i+1]
    return (Mi/(6*hi)*(xi1-xv)**3 + Mi1/(6*hi)*(xv-xi)**3
           +(yi/hi-Mi*hi/6)*(xi1-xv) +(yi1/hi-Mi1*hi/6)*(xv-xi))

def seg_fn(i, M, x_arr):
    x=np.atleast_1d(x_arr)
    hi=h[i]; xi,xi1=xs[i],xs[i+1]; Mi,Mi1=M[i],M[i+1]; yi,yi1=ys[i],ys[i+1]
    return (Mi/(6*hi)*(xi1-x)**3 + Mi1/(6*hi)*(x-xi)**3
           +(yi/hi-Mi*hi/6)*(xi1-x) +(yi1/hi-Mi1*hi/6)*(x-xi))

seg_xr = [np.linspace(xs[i],xs[i+1],200) for i in range(n-1)]
res_par = spline_eval(X_EVAL, M_par)
res_nat = spline_eval(X_EVAL, M_nat)

# ── Figura ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.color":TXT_PRI,"axes.labelcolor":TXT_SEC,
    "xtick.color":TXT_SEC,"ytick.color":TXT_SEC,"font.size":9,
})

fig = plt.figure(figsize=(14,8), facecolor=BG_FIG)
fig.text(0.5, 0.975,
         "Spline Cúbico  —  Extremos Parabólicos vs Natural",
         ha="center", va="top", fontsize=12, color=ACCENT)

gs = fig.add_gridspec(1,2, width_ratios=[1.45,1],
                      left=0.06,right=0.97,bottom=0.20,top=0.94,wspace=0.06)
ax  = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])

ax.set_facecolor(BG_AX); ax.set_xlim(0.7,5.4); ax.set_ylim(7,18)
ax.set_xlabel("x",fontsize=11); ax.set_ylabel("S(x)",fontsize=11)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)
ax.tick_params(colors=TXT_SEC,labelsize=9)
ax.grid(True,color=GRID_CLR,lw=0.6,zorder=0)
ax.axhline(0,color=SPINE_CLR,lw=0.5,zorder=1)
for xi in xs: ax.axvline(xi,color=SPINE_CLR,lw=0.5,ls=":",alpha=0.5,zorder=1)
ax.axvline(X_EVAL,color="#FF5252",lw=1.1,ls="--",alpha=0.8,zorder=2)
ax.text(X_EVAL+0.04,7.4,f"x={X_EVAL}",fontsize=8.5,color="#FF5252")

ax.scatter(xs,ys,zorder=7,s=95,color=ACCENT,edgecolors="white",linewidths=1.8)
for xi,yi in zip(xs,ys):
    ax.annotate(f"({xi:.0f},{yi})",(xi,yi),textcoords="offset points",
                xytext=(6,7),fontsize=8.5,color=ACCENT)

# Líneas parabólico (sólidas)
seg_par = [ax.plot([],[],color=SEG_PAR[i],lw=2.6,zorder=5)[0] for i in range(n-1)]
# Líneas natural (punteadas, más tenues)
seg_nat = [ax.plot([],[],color=SEG_NAT[i],lw=1.5,ls="--",alpha=0.5,zorder=4)[0] for i in range(n-1)]

# Puntos de evaluación
dot_par = ax.scatter([],[],s=140,color="#4FC3F7",zorder=8,
                     edgecolors="white",linewidths=2,marker="*")
dot_nat = ax.scatter([],[],s=100,color="#FFB74D",zorder=7,
                     edgecolors="white",linewidths=1.5,marker="D")

ax.legend(
    handles=[
        mpatches.Patch(color=SEG_PAR[0],label="Tramos parabólico (sólido)"),
        mpatches.Patch(color=SEG_NAT[0],alpha=0.5,label="Tramos natural (punteado)"),
        mpatches.Patch(color="#4FC3F7",label=f"S_par({X_EVAL})={res_par:.4f}"),
        mpatches.Patch(color="#FFB74D",label=f"S_nat({X_EVAL})={res_nat:.4f}"),
    ],
    loc="upper right",fontsize=8,framealpha=0.35,
    facecolor=BG_AX,edgecolor=SPINE_CLR,labelcolor=TXT_PRI)

status_txt = ax.text(0.02,0.97,"",transform=ax.transAxes,
                     fontsize=10,va="top",color=ACCENT)

axt.set_facecolor(BG_TABLE)
for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
axt.set_axis_off()

# ── Tabla ──────────────────────────────────────────────────────────────────
def draw_table(done=False):
    axt.cla(); axt.set_facecolor(BG_TABLE)
    for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
    axt.set_xlim(0,1); axt.set_ylim(0,1); axt.set_axis_off()

    if not done:
        axt.text(0.5,0.52,"Presiona  ▶  Animar\npara ver el spline",
                 ha="center",va="center",fontsize=11,color=TXT_SEC,
                 linespacing=2.2,transform=axt.transAxes)
        fig.canvas.draw_idle(); return

    axt.text(0.5,0.975,"Extremos parabólicos vs Natural",
             ha="center",va="top",fontsize=9.5,color=ACCENT,transform=axt.transAxes)

    r_h=0.054; y=0.90

    def sec_hdr(y_c,txt):
        rect=mpatches.FancyBboxPatch((0.01,y_c-r_h*0.46),0.98,r_h*0.92,
            boxstyle="round,pad=0.003",facecolor="#1A3A5C",edgecolor="none",
            transform=axt.transAxes,zorder=1,clip_on=False)
        axt.add_patch(rect)
        axt.text(0.5,y_c,txt,ha="center",va="center",fontsize=7.8,color=ACCENT,
                 transform=axt.transAxes,zorder=3,clip_on=False)

    def row(y_c,lbl,val,clr=TXT_PRI,bold=False,bg=None):
        if bg:
            rect=mpatches.FancyBboxPatch((0.02,y_c-r_h*0.46),0.96,r_h*0.92,
                boxstyle="round,pad=0.003",facecolor=bg,edgecolor="none",
                transform=axt.transAxes,zorder=1,clip_on=False)
            axt.add_patch(rect)
        axt.text(0.04,y_c,lbl,ha="left",va="center",fontsize=7.2,color=TXT_SEC,
                 transform=axt.transAxes,zorder=3,clip_on=False)
        axt.text(0.96,y_c,val,ha="right",va="center",
                 fontsize=8.0 if bold else 7.5,color=clr,
                 fontweight="bold" if bold else "normal",
                 transform=axt.transAxes,zorder=3,clip_on=False)

    def hline(y_c):
        axt.plot([0.02,0.98],[y_c]*2,color=SPINE_CLR,lw=0.5,ls="--",
                 transform=axt.transAxes,zorder=4)

    # 1. Condición de extremo
    sec_hdr(y,"1 · Condición de extremo parabólico"); y-=r_h*1.2
    row(y,"S'' constante en extremos:","M₀=M₁  y  M₃=M₄")
    y-=r_h
    row(y,"Consecuencia:","Tramos [1,2] y [4,5] son parábolas")
    y-=r_h
    hline(y-r_h*0.2); y-=r_h*0.7

    # 2. Sistema
    sec_hdr(y,"2 · Sistema 3×3  (h=1 uniforme)"); y-=r_h*1.2
    system=[("Nodo i=1:","5M₁ + M₂         = −30"),
            ("Nodo i=2:","M₁ + 4M₂ + M₃   =   0"),
            ("Nodo i=3:","       M₂ + 5M₃   =  42")]
    for lbl,val in system:
        row(y,lbl,val); y-=r_h
    hline(y-r_h*0.2); y-=r_h*0.7

    # 3. Momentos (comparación)
    sec_hdr(y,"3 · Momentos  Mᵢ"); y-=r_h*1.2
    # Tabla comparativa
    col_w=[0.06,0.46,0.48]
    hdr=["i","Parabólico","Natural"]
    xc=0.02
    for txt,cw in zip(hdr,col_w):
        axt.text(xc+cw/2,y,txt,ha="center",va="center",fontsize=7,color=TXT_SEC,
                 transform=axt.transAxes,zorder=3,clip_on=False); xc+=cw
    y-=r_h*0.85
    axt.plot([0.02,0.98],[y+r_h*0.15]*2,color=SPINE_CLR,lw=0.6,
             transform=axt.transAxes,zorder=4)
    for mi,(mp,mn) in enumerate(zip(M_par,M_nat)):
        is_end=(mi==0 or mi==4)
        bg="#0A2A50" if not is_end else None
        clr_p="#FF7043" if is_end else "#4FC3F7"
        clr_n="#FFB74D"
        rect=mpatches.FancyBboxPatch((0.02,y-r_h*0.44),0.96,r_h*0.88,
            boxstyle="round,pad=0.002",facecolor=bg or "none",edgecolor="none",
            transform=axt.transAxes,zorder=1,clip_on=False)
        if bg: axt.add_patch(rect)
        xc=0.02
        for txt,cw,clr in zip([str(mi),f"{mp:.4f}",f"{mn:.4f}"],col_w,[TXT_SEC,clr_p,clr_n]):
            axt.text(xc+cw/2,y,txt,ha="center",va="center",fontsize=7.8,color=clr,
                     fontweight="bold" if is_end and cw==col_w[1] else "normal",
                     transform=axt.transAxes,zorder=3,clip_on=False); xc+=cw
        y-=r_h
    hline(y-r_h*0.2); y-=r_h*0.7

    # 4. Evaluación
    sec_hdr(y,f"4 · Evaluación en  x = {X_EVAL}"); y-=r_h*1.2
    row(y,"Tramo activo:","[3, 4]  →  i = 2"); y-=r_h
    row(y,"Parabólico:",f"{res_par:.6f}",clr="#4FC3F7",bold=True,bg="#0A2A50"); y-=r_h
    row(y,"Natural:",   f"{res_nat:.6f}",clr="#FFB74D",bold=True,bg="#0A2A50"); y-=r_h
    hline(y-r_h*0.2); y-=r_h*0.7

    # Caja final
    y_box=y-r_h*0.2
    box=mpatches.FancyBboxPatch((0.03,y_box-0.062),0.94,0.105,
        boxstyle="round,pad=0.01",facecolor="#0D47A1",edgecolor=ACCENT,linewidth=1.2,
        transform=axt.transAxes,zorder=4,clip_on=False)
    axt.add_patch(box)
    axt.text(0.5,y_box+0.022,f"S_par(3.4)  =  {res_par:.6f}",
             ha="center",va="center",fontsize=10,color="#4FC3F7",fontweight="bold",
             transform=axt.transAxes,zorder=5,clip_on=False)
    axt.text(0.5,y_box-0.018,f"S_nat(3.4)  =  {res_nat:.6f}",
             ha="center",va="center",fontsize=10,color="#FFB74D",fontweight="bold",
             transform=axt.transAxes,zorder=5,clip_on=False)
    fig.canvas.draw_idle()

# ── Animación ──────────────────────────────────────────────────────────────
N_FRAMES=75; MSG_FRAMES=30
state={"running":False,"anim":None}
BUILD=[
    "Condición extremo parabólico: M₀=M₁, M₃=M₄...",
    "Derivando sistema 3×3...",
    "Resolviendo: M = [−5.8667, −5.8667, −0.6667, 8.5333, 8.5333]",
    "Trazando S₀ parabólico [1,2] (tramo parabólico)...",
    "Trazando S₁ [2,3]...",
    "Trazando S₂ [3,4]  ← evaluar x=3.4...",
    "Trazando S₃ parabólico [4,5] (tramo parabólico)...",
    "Superponiendo spline natural (punteado) para comparar...",
    f"Evaluando ambos en x={X_EVAL}...",
]

def clear_all():
    for ln in seg_par+seg_nat: ln.set_data([],[])
    dot_par.set_offsets(np.empty((0,2)))
    dot_nat.set_offsets(np.empty((0,2)))
    status_txt.set_text("")
    draw_table(done=False)

def run_animation(event=None):
    if state["running"]: return
    clear_all(); state["running"]=True

    n_msg=3; n_cur=4; n_nat=1; n_fin=1
    total=(n_msg)*MSG_FRAMES + n_cur*(MSG_FRAMES+N_FRAMES) + n_nat*(MSG_FRAMES+N_FRAMES) + n_fin*MSG_FRAMES

    def update(frame):
        # Bloques: 3 msg, 4 curvas par, 1 bloque nat (todas), 1 final
        bk=[n_msg*MSG_FRAMES]
        for _ in range(n_cur): bk.append(bk[-1]+MSG_FRAMES+N_FRAMES)
        bk.append(bk[-1]+MSG_FRAMES+N_FRAMES)  # natural
        bk.append(bk[-1]+MSG_FRAMES)

        if frame < bk[0]:
            ph=frame//MSG_FRAMES; status_txt.set_text(BUILD[ph])
        elif frame < bk[4]:
            # Tramos parabólicos
            f2=frame-bk[0]; pl=MSG_FRAMES+N_FRAMES; ph2=f2//pl; loc=f2%pl
            if ph2<4:
                status_txt.set_text(BUILD[3+ph2])
                if loc>=MSG_FRAMES:
                    cf=loc-MSG_FRAMES
                    n_pts=max(2,int((cf+1)/N_FRAMES*len(seg_xr[ph2])))
                    seg_par[ph2].set_data(seg_xr[ph2][:n_pts],
                                         seg_fn(ph2,M_par,seg_xr[ph2][:n_pts]))
        elif frame < bk[5]:
            # Natural overlay
            f3=frame-bk[4]; loc=f3%(MSG_FRAMES+N_FRAMES)
            status_txt.set_text(BUILD[7])
            cf=max(0,loc-MSG_FRAMES)
            for si in range(4):
                n_pts=max(2,int((cf+1)/N_FRAMES*len(seg_xr[si])))
                seg_nat[si].set_data(seg_xr[si][:n_pts],
                                     seg_fn(si,M_nat,seg_xr[si][:n_pts]))
        else:
            status_txt.set_text(BUILD[8])
            dot_par.set_offsets([[X_EVAL,res_par]])
            dot_nat.set_offsets([[X_EVAL,res_nat]])

        if frame==total-1:
            status_txt.set_text(
                f"✓  Parabólico: S({X_EVAL})={res_par:.5f}   "
                f"Natural: S({X_EVAL})={res_nat:.5f}")
            draw_table(done=True); state["running"]=False
        return tuple(seg_par+seg_nat)

    state["anim"]=FuncAnimation(fig,update,frames=total,
                                 interval=18,blit=False,repeat=False)
    fig.canvas.draw_idle()

def reset_animation(event=None):
    if state["anim"]:
        state["anim"].event_source.stop(); state["anim"]=None
    state["running"]=False; clear_all(); fig.canvas.draw_idle()

ax_play=fig.add_axes([0.06,0.075,0.16,0.07])
ax_reset=fig.add_axes([0.24,0.075,0.16,0.07])
for wax in [ax_play,ax_reset]:
    wax.set_facecolor(BG_FIG)
    for sp in wax.spines.values(): sp.set_color(SPINE_CLR)
btn_play=Button(ax_play,"▶   Animar",color="#0D2B55",hovercolor="#1A3A6E")
btn_reset=Button(ax_reset,"↺   Reset",color="#1A1A3E",hovercolor="#2A2A5E")
for b in [btn_play,btn_reset]:
    b.label.set_color(TXT_PRI); b.label.set_fontsize(10)
btn_play.on_clicked(run_animation)
btn_reset.on_clicked(reset_animation)

draw_table(done=False)
plt.show()