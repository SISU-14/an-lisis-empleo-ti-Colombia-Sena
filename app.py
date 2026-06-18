"""
Segmentador de Empleo Digital Colombia - SENA Funza
Autores: Daniel Ibanez, Johan Moreno, Miguel Portilla
Instructor: Carlos Andres Figueredo Rodriguez
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from constants import (
    TOP_TECHS, SALARIO_MEDIANA, TECH_MULTIPLIERS, MODALIDAD_MULTIPLIERS,
    CONTRATO_MULTIPLIERS, CLUSTER_COLORS, CLUSTER_NAMES, CLUSTER_INFO,
    FEATURE_COLUMNS_DEFAULT, PKL_FILES,
)
from utils import (
    cargar_modelos, construir_vector, calcular_salario,
    clasificar_por_reglas, generar_distribucion_salarial,
)

# =========================================================
# CONFIGURACION DE PAGINA
# =========================================================
st.set_page_config(
    page_title="Segmentador Empleo Digital - SENA",
    page_icon="\U0001f3af",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# ESTILOS CSS
# =========================================================
def _load_css(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()

_css_path = os.path.join(os.path.dirname(__file__), "styles.css")
st.markdown(f"<style>{_load_css(_css_path)}</style>", unsafe_allow_html=True)

# =========================================================
# PARTICLES CANVAS (JavaScript inyectado)
# =========================================================
_PARTICLES_JS = """
<script>
(function() {
    if (document.getElementById("particles-canvas")) return;
    const canvas = document.createElement("canvas");
    canvas.id = "particles-canvas";
    document.body.insertBefore(canvas, document.body.firstChild);
    const ctx = canvas.getContext("2d");
    let W, H;
    function resize() { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; }
    resize();
    window.addEventListener("resize", resize);
    const COUNT = 55, DIST = 140;
    const SYMBOLS = ["{", "}", "<", ">", "/", ";", "0", "1", "\\u03BB", "\\u03A3", "#", "%", "&", "*", "+", "=", "?", "!", "[", "]"];
    const particles = [];
    class Particle {
        constructor() { this.reset(); }
        reset() {
            this.x = Math.random() * W; this.y = Math.random() * H;
            this.vx = (Math.random() - 0.5) * 0.4; this.vy = (Math.random() - 0.5) * 0.4;
            this.r = Math.random() * 2 + 1; this.o = Math.random() * 0.4 + 0.1;
            this.sym = SYMBOLS[Math.floor(Math.random() * SYMBOLS.length)];
            this.isSym = Math.random() > 0.7; this.ph = Math.random() * Math.PI * 2;
        }
        update() { this.x += this.vx; this.y += this.vy; this.ph += 0.02;
            if (this.x < 0 || this.x > W) this.vx *= -1; if (this.y < 0 || this.y > H) this.vy *= -1; }
        draw() { const p = Math.sin(this.ph) * 0.15 + this.o;
            if (this.isSym) { ctx.font = (this.r * 5 + 8) + "px monospace"; ctx.fillStyle = "rgba(0,242,254," + (p*0.5) + ")"; ctx.fillText(this.sym, this.x, this.y); }
            else { ctx.beginPath(); ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2); ctx.fillStyle = "rgba(0,242,254," + p + ")"; ctx.fill(); } }
    }
    for (let i = 0; i < COUNT; i++) particles.push(new Particle());
    function drawLines() { for (let i = 0; i < particles.length; i++) for (let j = i + 1; j < particles.length; j++) { const dx = particles[i].x - particles[j].x, dy = particles[i].y - particles[j].y, d = Math.sqrt(dx * dx + dy * dy); if (d < DIST) { const a = (1 - d / DIST) * 0.12; ctx.beginPath(); ctx.moveTo(particles[i].x, particles[i].y); ctx.lineTo(particles[j].x, particles[j].y); ctx.strokeStyle = "rgba(0,242,254," + a + ")"; ctx.lineWidth = 0.5; ctx.stroke(); } } }
    function animate() { ctx.clearRect(0, 0, W, H); particles.forEach(function(p) { p.update(); p.draw(); }); drawLines(); requestAnimationFrame(animate); }
    animate();
})();
</script>
"""
st.markdown(_PARTICLES_JS, unsafe_allow_html=True)

# =========================================================
# CARGA DE MODELOS
# =========================================================
modelos, errores_carga = cargar_modelos(PKL_FILES)
modelo_ok = len(errores_carga) == 0
kmeans_model = modelos.get("kmeans")
pca_model = modelos.get("pca")
scaler_model = modelos.get("scaler")
feature_cols = modelos.get("features", FEATURE_COLUMNS_DEFAULT)
pca_variance = modelos.get("pca_variance", None)

if modelo_ok and kmeans_model is not None:
    _expected = len(feature_cols)
    _actual = kmeans_model.cluster_centers_.shape[1]
    if _expected != _actual:
        errores_carga.append(
            f"Inconsistencia de features: el modelo K-Means espera {_actual} columnas "
            f"pero feature_cols tiene {_expected}. Regenera los .pkl desde el notebook."
        )
        modelo_ok = False

# =========================================================
# NAVEGACION
# =========================================================
PAGINAS = ["🏠 Inicio", "🔎 Test de Perfil", "⚖️ Comparador", "📊 Visualizaciones", "📖 Metodologia"]

if "pagina" not in st.session_state:
    st.session_state.pagina = "🏠 Inicio"

pagina_actual = st.session_state.pagina

# Barra de navegacion visual (HTML)
_nav_html = f"""
<div class="top-nav">
    <div class="nav-logo">SENA FUNZA</div>
    <div class="nav-links">
        {"".join(f'<span class="nav-link {"active" if p == pagina_actual else ""}">{p}</span>' for p in PAGINAS)}
    </div>
    <div class="nav-hamburger"><span></span><span></span><span></span></div>
</div>
"""
st.markdown(_nav_html, unsafe_allow_html=True)

# Botones de navegacion funcionales (Streamlit native)
st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
nav_cols = st.columns(len(PAGINAS))
for i, pagina in enumerate(PAGINAS):
    with nav_cols[i]:
        btn_type = "primary" if pagina == pagina_actual else "secondary"
        if st.button(pagina, key=f"nav_btn_{pagina}", type=btn_type, use_container_width=True):
            st.session_state.pagina = pagina
            st.rerun()

st.markdown("<hr style='border:none;border-top:1px solid #1f293d;margin:16px 0;'>", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("<div class='sidebar-logo'>SENA FUNZA</div>", unsafe_allow_html=True)
    st.markdown("**Analisis de Empleo Digital**")
    st.markdown("---")
    st.markdown("**Proyecto**")
    st.markdown("Segmentacion de 5.000 ofertas de empleo digital en Colombia mediante K-Means + PCA.")
    st.markdown("---")
    st.markdown("**Autores**")
    st.markdown("- Daniel Ibanez\\n- Johan Moreno\\n- Miguel Portilla")
    st.markdown("**Instructor**")
    st.markdown("Carlos A. Figueredo R.")
    st.markdown("---")
    st.markdown("**Archivos del modelo**")
    for key, fname in PKL_FILES.items():
        exists = os.path.exists(fname)
        icon = "\u2705" if exists else ("\u26a0" if key == "pca_variance" else "\u274c")
        st.markdown(f"{icon} `{fname}`")

# =========================================================
# HELPER: desglose salarial
# =========================================================
def _html_salary_breakdown(detalle, sel_experiencia, sel_modalidad, sel_contrato):
    salario_final = detalle["final"]
    salario_base = detalle["base"]

    filas = f"""
    <div class="salary-row base">
      <span>Salario Base de Referencia ({sel_experiencia})</span>
      <span>${salario_base:,.0f} COP</span>
    </div>
    """

    tech_bonuses = detalle["tech_bonuses"]
    if tech_bonuses:
        pills = "".join(f'<span class="tech-pill">{b["tech"]} (+{b["pct"]:.0%})</span>' for b in tech_bonuses)
        limite_msg = (
            " <span style='font-size:0.75rem;color:#ff9f1c;font-family:monospace'>"
            "// Top max. +40% aplicado</span>" if detalle["excede_max_tech"] else ""
        )
        filas += f"""
        <div class="salary-row bonus">
          <span>Bono de Habilidades {limite_msg}<br/>{pills}</span>
          <span>+ ${detalle["tech_bonus_val"]:,.0f} COP</span>
        </div>
        """
    else:
        filas += """
        <div class="salary-row" style="color: #8b949e;">
          <span>Bono de Habilidades (Ninguna seleccionada)</span>
          <span>$0 COP</span>
        </div>
        """

    mod_mul = detalle["mod_mul"]
    if mod_mul != 0.0:
        cls = "bonus" if mod_mul > 0 else "penalty"
        sign = "+" if mod_mul > 0 else "-"
        filas += f"""
        <div class="salary-row {cls}">
          <span>Ajuste de Modalidad ({sel_modalidad}: {sign}{abs(mod_mul):.0%})</span>
          <span>{sign} ${abs(detalle["mod_val"]):,d} COP</span>
        </div>
        """
    else:
        filas += f"""
        <div class="salary-row" style="color: #8b949e;">
          <span>Ajuste de Modalidad ({sel_modalidad}: 0%)</span>
          <span>$0 COP</span>
        </div>
        """

    cont_mul = detalle["cont_mul"]
    if cont_mul != 0.0:
        cls = "bonus" if cont_mul > 0 else "penalty"
        sign = "+" if cont_mul > 0 else "-"
        filas += f"""
        <div class="salary-row {cls}">
          <span>Ajuste de Contrato ({sel_contrato}: {sign}{abs(cont_mul):.0%})</span>
          <span>{sign} ${abs(detalle["cont_val"]):,d} COP</span>
        </div>
        """
    else:
        filas += f"""
        <div class="salary-row" style="color: #8b949e;">
          <span>Ajuste de Contrato ({sel_contrato}: 0%)</span>
          <span>$0 COP</span>
        </div>
        """

    filas += f"""
    <div class="salary-row total">
      <span>Salario Estimado Final de tu Perfil</span>
      <span>${salario_final:,.0f} COP/mes</span>
    </div>
    """

    return f"""
    <div class="salary-breakdown-card">
      <div class="salary-breakdown-header">
        <div class="salary-breakdown-title">Desglose Detallado del Salario Estimado</div>
        <div class="salary-breakdown-total">${salario_final:,.0f} COP</div>
      </div>
      {filas}
    </div>
    """


# =========================================================
# PAGINA: INICIO
# =========================================================
if pagina_actual == "🏠 Inicio":
    st.markdown(
        '<div class="title-container">'
        '<div class="main-title"><span class="main-title-text">'
        '<span class="title-prompt">&gt;_</span>Segmentador de Empleo Digital<br>'
        'Colombia &middot; SENA Funza</span></div>'
        '<div class="main-subtitle">Daniel Ibanez &nbsp;&middot;&nbsp; Johan Moreno &nbsp;&middot;&nbsp; '
        'Miguel Portilla</div>'
        '<div class="accent-bar"></div>'
        '<div class="terminal-line"><span class="comment">// Powered by K-Means + PCA &middot; Python &middot; Streamlit</span></div>'
        '</div>',
        unsafe_allow_html=True
    )

    if modelo_ok:
        st.markdown(
            '<div class="status-ok">Modelo cargado correctamente &mdash; Pipeline K-Means + PCA listo</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-err"><b>Archivos .pkl no encontrados</b><br>' +
            "<br>".join(errores_carga) +
            "<br><br>Ejecuta el notebook <code>Clustering_Empleo_Digital_Colombia.ipynb</code> "
            "para generarlos.</div>",
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-head code-style">Metricas del Modelo</div>', unsafe_allow_html=True)

    var_exp_pc1 = var_exp_pc2 = None
    if pca_model:
        var_exp_pc1 = pca_model.explained_variance_ratio_[0]
        var_exp_pc2 = pca_model.explained_variance_ratio_[1]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Clusteres (K)", "4", "K-Means")
    col2.metric("Ofertas analizadas", "5.000", "dataset original")
    col3.metric("Varianza PC1", f"{var_exp_pc1:.1%}" if var_exp_pc1 else "\u2014", "PCA")
    col4.metric("Varianza PC2", f"{var_exp_pc2:.1%}" if var_exp_pc2 else "\u2014", "PCA")
    col5.metric("Variables del modelo", str(len(feature_cols)), "29 features")

    st.markdown('<div class="section-head code-style">Los 4 Clusteres Identificados</div>', unsafe_allow_html=True)

    for c in range(4):
        info = CLUSTER_INFO[c]
        color = CLUSTER_COLORS[c]
        nombre = CLUSTER_NAMES[c]
        st.markdown(f"""
        <div class="code-editor-card" style="border-left: 4px solid {color}">
          <div class="editor-header">
            <div class="editor-dots">
              <span class="editor-dot red"></span>
              <span class="editor-dot yellow"></span>
              <span class="editor-dot green"></span>
            </div>
            <div class="editor-title" style="color:{color}">cluster_0{c}_profile.json</div>
          </div>
          <div class="editor-body">
            <div class="code-line"><span class="ln">1</span><span class="code-keyword">const</span> <span class="code-key">cluster</span> = <span class="code-symbol">{{</span></div>
            <div class="code-line"><span class="ln">2</span>  <span class="code-key">"id"</span><span class="code-symbol">:</span> <span class="code-number">{c}</span>,</div>
            <div class="code-line"><span class="ln">3</span>  <span class="code-key">"nombre"</span><span class="code-symbol">:</span> <span class="code-string">"{nombre.split(chr(0x2014))[1].strip()}"</span>,</div>
            <div class="code-line"><span class="ln">4</span>  <span class="code-key">"experiencia"</span><span class="code-symbol">:</span> <span class="code-string">"{info["experiencia"]}"</span>,</div>
            <div class="code-line"><span class="ln">5</span>  <span class="code-key">"salario_rango"</span><span class="code-symbol">:</span> <span class="code-string">"{info["salario"]} COP"</span>,</div>
            <div class="code-line"><span class="ln">6</span>  <span class="code-key">"salario_mediana"</span><span class="code-symbol">:</span> <span class="code-string">"${info["sal_median"]:,.0f} COP"</span>,</div>
            <div class="code-line"><span class="ln">7</span>  <span class="code-key">"tecnologias_clave"</span><span class="code-symbol">:</span> <span class="code-string">"{info["tecnologias"]}"</span>,</div>
            <div class="code-line"><span class="ln">8</span>  <span class="code-key">"modalidad"</span><span class="code-symbol">:</span> <span class="code-string">"{info["modalidad"]}"</span>,</div>
            <div class="code-line"><span class="ln">9</span>  <span class="code-key">"ofertas_registradas"</span><span class="code-symbol">:</span> <span class="code-number">{info["n_ofertas"]}</span></div>
            <div class="code-line"><span class="ln">10</span><span class="code-symbol">}};</span></div>
            <div class="rec-box">
              <div class="rec-title">Recomendacion Formativa SENA:</div>
              {info["recomendacion"]}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# PAGINA: TEST DE PERFIL
# =========================================================
elif pagina_actual == "🔎 Test de Perfil":
    st.markdown(
        '<div class="title-container">'
        '<div class="main-title"><span class="main-title-text">'
        '<span class="title-prompt">&gt;_</span>Test de Perfil</span></div>'
        '<div class="main-subtitle">Selecciona tus habilidades y descubre tu cluster de empleo digital</div>'
        '<div class="accent-bar"></div>'
        '<div class="terminal-line"><span class="comment">// funcion clasificar_perfil(techs, exp, mod, cont) -&gt; cluster</span></div>'
        '</div>',
        unsafe_allow_html=True
    )

    if not modelo_ok:
        st.markdown(
            '<div class="fallback-banner">'
            '<b>Modo Fallback activo</b> &mdash; Los archivos del modelo (.pkl) no estan disponibles. '
            "La clasificacion se realiza mediante reglas heuristicas. "
            "Los resultados son aproximados. Ejecuta el notebook para restaurar el modelo K-Means completo.</div>",
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-head code-style">Tecnologias que dominas</div>', unsafe_allow_html=True)
    cols_tech = st.columns(3)
    tech_checks = {}
    for i, tech in enumerate(TOP_TECHS):
        tech_checks[tech] = cols_tech[i % 3].checkbox(tech, key=f"cb_{tech}")

    st.markdown('<div class="section-head code-style">Informacion del Perfil</div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        sel_experiencia = st.selectbox("Nivel de experiencia", ["Junior", "Semi-Senior", "Senior"])
    with col_b:
        sel_modalidad = st.selectbox("Modalidad preferida", ["Remoto", "Hibrido", "Presencial"])
    with col_c:
        sel_contrato = st.selectbox("Tipo de contrato", ["Indefinido", "Temporal", "Contrato", "Freelance"])

    techs_sel = [t for t, v in tech_checks.items() if v]
    st.markdown(
        f"**Tecnologias seleccionadas ({len(techs_sel)}):** " + (", ".join(techs_sel) if techs_sel else "*Ninguna*")
    )

    if not techs_sel:
        st.warning(
            "No has seleccionado ninguna tecnologia. "
            "El analisis funcionara, pero el perfil tendra menor precision. "
            "Selecciona al menos una para un resultado mas representativo."
        )

    if st.button("Analizar mi perfil", type="primary", use_container_width=True):
        if modelo_ok:
            X_nuevo = construir_vector(techs_sel, sel_experiencia, sel_modalidad, sel_contrato, feature_cols)
            X_scaled = scaler_model.transform(X_nuevo)
            cluster_id = int(kmeans_model.predict(X_scaled)[0])
            distancias = kmeans_model.transform(X_scaled)[0]
            afinidades = 1 / (1 + distancias)
            afinidades_norm = afinidades / afinidades.sum() * 100
        else:
            cluster_id = clasificar_por_reglas(techs_sel, sel_experiencia)
            afinidades_norm = None

        info = CLUSTER_INFO[cluster_id]
        color = CLUSTER_COLORS[cluster_id]
        nombre = CLUSTER_NAMES[cluster_id]

        salario_base = SALARIO_MEDIANA[sel_experiencia]
        salario_final, detalle = calcular_salario(salario_base, techs_sel, sel_modalidad, sel_contrato)

        tus_techs_str = ", ".join(f'"{t}"' for t in techs_sel) if techs_sel else ""
        st.markdown("---")
        st.markdown('<div class="section-head code-style">Resultado</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="code-editor-card" style="border-left: 4px solid {color}">
          <div class="editor-header">
            <div class="editor-dots">
              <span class="editor-dot red"></span>
              <span class="editor-dot yellow"></span>
              <span class="editor-dot green"></span>
            </div>
            <div class="editor-title" style="color:{color}">profile_classifier_output.log</div>
          </div>
          <div class="editor-body">
            <div class="code-line"><span class="ln">1</span><span class="code-comment">// --- ANALISIS DE PERFIL COMPLETADO ---</span></div>
            <div class="code-line"><span class="ln">2</span><span class="code-keyword">struct</span> <span class="code-key">UserProfile</span> <span class="code-symbol">{{</span></div>
            <div class="code-line"><span class="ln">3</span>  <span class="code-key">experiencia_detectada</span><span class="code-symbol">:</span> <span class="code-string">"{sel_experiencia}"</span>,</div>
            <div class="code-line"><span class="ln">4</span>  <span class="code-key">modalidad_preferida</span><span class="code-symbol">:</span> <span class="code-string">"{sel_modalidad}"</span>,</div>
            <div class="code-line"><span class="ln">5</span>  <span class="code-key">contrato_seleccionado</span><span class="code-symbol">:</span> <span class="code-string">"{sel_contrato}"</span>,</div>
            <div class="code-line"><span class="ln">6</span>  <span class="code-key">skills_usuario</span><span class="code-symbol">:</span> <span class="code-symbol">[</span>{tus_techs_str}<span class="code-symbol">]</span></div>
            <div class="code-line"><span class="ln">7</span><span class="code-symbol">}};</span></div>
            <div class="code-line"><span class="ln">8</span></div>
            <div class="code-line"><span class="ln">9</span><span class="code-keyword">const</span> <span class="code-key">classificationResult</span> = <span class="code-symbol">{{</span></div>
            <div class="code-line"><span class="ln">10</span>  <span class="code-key">cluster_asignado</span><span class="code-symbol">:</span> <span class="code-string">"{nombre}"</span>,</div>
            <div class="code-line"><span class="ln">11</span>  <span class="code-key">experiencia_promedio</span><span class="code-symbol">:</span> <span class="code-string">"{info["experiencia"]}"</span>,</div>
            <div class="code-line"><span class="ln">12</span>  <span class="code-key">salario_promedio_rango</span><span class="code-symbol">:</span> <span class="code-string">"{info["salario"]} COP/mes"</span>,</div>
            <div class="code-line"><span class="ln">13</span>  <span class="code-key">salario_estimado_personalizado</span><span class="code-symbol">:</span> <span class="code-string">"${salario_final:,.0f} COP/mes"</span>,</div>
            <div class="code-line"><span class="ln">14</span>  <span class="code-key">stack_dominante_cluster</span><span class="code-symbol">:</span> <span class="code-string">"{info["tecnologias"]}"</span></div>
            <div class="code-line"><span class="ln">15</span><span class="code-symbol">}};</span></div>
            <div class="rec-box">
              <div class="rec-title">Recomendacion de Formacion SENA:</div>
              {info["recomendacion"]}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        salario_delta = salario_final - salario_base
        if salario_delta > 0:
            delta_str = f"+${salario_delta:,.0f} COP (habilidades/preferencias)"
        elif salario_delta < 0:
            delta_str = f"-${abs(salario_delta):,.0f} COP (ajuste presencial/temporal)"
        else:
            delta_str = "Sin variacion vs base"

        cm1, cm2, cm3 = st.columns(3)
        cm1.metric("Cluster asignado", nombre.split(chr(0x2014))[0].strip())
        cm2.metric("Salario estimado de tu perfil", f"${salario_final:,.0f} COP", delta=delta_str)
        cm3.metric("Tecnologias seleccionadas", len(techs_sel))

        st.markdown(_html_salary_breakdown(detalle, sel_experiencia, sel_modalidad, sel_contrato), unsafe_allow_html=True)

        r_min = info["sal_min"]
        r_max = info["sal_max"]
        fig_rango = go.Figure()
        fig_rango.add_trace(go.Bar(
            x=[r_max - r_min], y=["Salario"], base=r_min,
            orientation="h", marker_color=color, opacity=0.25,
            name="Rango del Cluster", hoverinfo="skip", showlegend=False
        ))
        fig_rango.add_trace(go.Scatter(
            x=[salario_final], y=["Salario"], mode="markers+text",
            marker=dict(size=20, color=color, symbol="diamond-dot", line=dict(width=3, color="#e6edf3")),
            text=[f"${salario_final:,.0f} COP"], textposition="top center",
            name="Tu Perfil", hovertemplate="<b>Tu Salario Estimado:</b><br>%{x:$,.0f} COP<extra></extra>",
            showlegend=False
        ))
        fig_rango.add_trace(go.Scatter(
            x=[r_min], y=["Salario"], mode="markers+text",
            marker=dict(size=12, color="#8b949e", symbol="line-ns-open"),
            text=[f"Min: ${r_min:,.0f}"], textposition="bottom center", hoverinfo="skip", showlegend=False
        ))
        fig_rango.add_trace(go.Scatter(
            x=[r_max], y=["Salario"], mode="markers+text",
            marker=dict(size=12, color="#8b949e", symbol="line-ns-open"),
            text=[f"Max: ${r_max:,.0f}"], textposition="bottom center", hoverinfo="skip", showlegend=False
        ))
        x_min = min(r_min * 0.82, salario_final * 0.82)
        x_max = max(r_max * 1.15, salario_final * 1.15)
        fig_rango.update_layout(
            title=dict(
                text=f"Tu Salario Estimado en el Rango del Cluster Asignado ({info['experiencia']})",
                font=dict(family="JetBrains Mono, Fira Code, monospace", color="#00f2fe", size=13)
            ),
            height=190, margin=dict(l=20, r=20, t=55, b=45),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Outfit, sans-serif", color="#e6edf3"),
            xaxis=dict(range=[x_min, x_max], gridcolor="#1f293d", zerolinecolor="#1f293d", tickformat="$,.0f"),
            yaxis=dict(showticklabels=False, gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_rango, use_container_width=True)

        if salario_final > r_max:
            st.markdown(
                '<div class="info-note" style="border-left: 3px solid #ff9f1c; '
                'background: rgba(255, 159, 28, 0.05); border-radius: 8px;">'
                "<b>Nota:</b> El salario estimado supera el rango salarial tipico de este cluster "
                "en el dataset. Esto es normal si posees una combinacion de multiples habilidades "
                "de alta demanda o modalidades con prima (como Freelance), las cuales representan "
                "nichos muy competitivos y especializados.</div>",
                unsafe_allow_html=True
            )

        if afinidades_norm is not None:
            st.markdown('<div class="section-head code-style">Afinidad con cada cluster</div>', unsafe_allow_html=True)
            afinidad_html = ""
            for i in range(4):
                afinidad_html += f"""
                <div class="afinidad-row">
                  <div class="afinidad-label">{CLUSTER_NAMES[i].split(chr(0x2014))[1].strip()}</div>
                  <div class="afinidad-bar-bg">
                    <div class="afinidad-bar-fill" style="width:{afinidades_norm[i]:.1f}%;background:{CLUSTER_COLORS[i]}"></div>
                  </div>
                  <div class="afinidad-pct">{afinidades_norm[i]:.1f}%</div>
                </div>"""
            st.markdown(afinidad_html, unsafe_allow_html=True)

            fig_af = go.Figure(go.Bar(
                x=[afinidades_norm[i] for i in range(4)],
                y=[CLUSTER_NAMES[i] for i in range(4)],
                orientation="h",
                marker_color=[CLUSTER_COLORS[i] for i in range(4)],
                text=[f"{afinidades_norm[i]:.1f}%" for i in range(4)],
                textposition="outside",
            ))
            fig_af.update_layout(
                title=dict(text="Afinidad (%) con cada cluster", font=dict(family="JetBrains Mono, Fira Code, monospace", color="#00f2fe")),
                xaxis_title="Afinidad (%)",
                height=280, margin=dict(l=0, r=80, t=40, b=20),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Outfit, sans-serif", color="#e6edf3"),
                xaxis=dict(range=[0, max(afinidades_norm) * 1.2], gridcolor="#1f293d", zerolinecolor="#1f293d"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_af, use_container_width=True)

        st.markdown('<div class="section-head code-style">Simulador - Que pasa si aprendo una nueva tech?</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-note">Selecciona una tecnologia que aun no dominas y veras el impacto '
            "inmediato en tu salario estimado y si cambia tu cluster asignado.</div>",
            unsafe_allow_html=True
        )
        techs_disponibles = [t for t in TOP_TECHS if t not in techs_sel]
        if techs_disponibles:
            tech_nueva = st.selectbox("Tecnologia a aprender", techs_disponibles, key="sim_tech")
            techs_sim = techs_sel + [tech_nueva]

            if modelo_ok:
                X_sim = construir_vector(techs_sim, sel_experiencia, sel_modalidad, sel_contrato, feature_cols)
                X_sim_sc = scaler_model.transform(X_sim)
                cluster_sim = int(kmeans_model.predict(X_sim_sc)[0])
            else:
                cluster_sim = clasificar_por_reglas(techs_sim, sel_experiencia)

            sal_sim, _ = calcular_salario(salario_base, techs_sim, sel_modalidad, sel_contrato)
            bonus_nueva = TECH_MULTIPLIERS.get(tech_nueva, 0)
            sal_delta_sim = sal_sim - salario_final

            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Tecnologia simulada", tech_nueva, f"+{bonus_nueva:.0%} multiplicador")
            col_s2.metric(
                "Nuevo salario estimado", f"${sal_sim:,.0f} COP",
                delta=f"+${sal_delta_sim:,.0f} COP" if sal_delta_sim > 0 else f"${sal_delta_sim:,.0f} COP"
            )
            col_s3.metric(
                "Cluster resultante", CLUSTER_NAMES[cluster_sim].split(chr(0x2014))[0].strip(),
                delta="Sube de cluster" if cluster_sim < cluster_id else
                      ("Sin cambio" if cluster_sim == cluster_id else "Baja de cluster")
            )

            fig_sim = go.Figure()
            fig_sim.add_trace(go.Bar(
                x=["Base referencia", "Perfil actual", f"+ {tech_nueva}"],
                y=[salario_base, salario_final, sal_sim],
                marker_color=["#484f58", color, CLUSTER_COLORS[cluster_sim]],
                text=[f"${salario_base:,.0f}", f"${salario_final:,.0f}", f"${sal_sim:,.0f}"],
                textposition="outside",
                hovertemplate="%{x}<br><b>%{y:$,.0f} COP</b><extra></extra>",
            ))
            fig_sim.update_layout(
                title=dict(text="Comparacion salarial: Base -> Perfil actual -> Con nueva tech",
                           font=dict(family="JetBrains Mono, Fira Code, monospace", color="#00f2fe", size=12)),
                height=290, margin=dict(l=20, r=20, t=45, b=20),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Outfit, sans-serif", color="#e6edf3"),
                yaxis=dict(tickformat="$,.0f", gridcolor="#1f293d", zerolinecolor="#1f293d"),
                xaxis=dict(gridcolor="rgba(0,0,0,0)"), showlegend=False,
            )
            st.plotly_chart(fig_sim, use_container_width=True)
        else:
            st.info("Ya tienes todas las tecnologias disponibles seleccionadas.")


# =========================================================
# PAGINA: COMPARADOR
# =========================================================
elif pagina_actual == "⚖️ Comparador":
    st.markdown(
        '<div class="title-container">'
        '<div class="main-title"><span class="main-title-text">'
        '<span class="title-prompt">&gt;_</span>Comparador de Perfiles</span></div>'
        '<div class="main-subtitle">Ingresa dos perfiles y compara cluster, salario y afinidades lado a lado</div>'
        '<div class="accent-bar"></div>'
        '<div class="terminal-line"><span class="comment">// compare(profile_a, profile_b) -&gt; diff_report</span></div>'
        '</div>',
        unsafe_allow_html=True
    )

    if not modelo_ok:
        st.markdown(
            '<div class="fallback-banner">'
            "<b>Modo Fallback activo</b> &mdash; Clasificacion por reglas heuristicas. "
            "El grafico radar de afinidades no esta disponible sin el modelo K-Means.</div>",
            unsafe_allow_html=True
        )

    col_izq, col_der = st.columns(2)

    def render_perfil_inputs(col, prefix, label):
        with col:
            st.markdown(f'<div class="section-head code-style">{label}</div>', unsafe_allow_html=True)
            exp = st.selectbox("Experiencia", ["Junior", "Semi-Senior", "Senior"], key=f"{prefix}_exp")
            mod = st.selectbox("Modalidad", ["Remoto", "Hibrido", "Presencial"], key=f"{prefix}_mod")
            cont = st.selectbox("Contrato", ["Indefinido", "Temporal", "Contrato", "Freelance"], key=f"{prefix}_cont")
            st.markdown("**Tecnologias:**")
            cols_t = st.columns(2)
            checks = {}
            for i, tech in enumerate(TOP_TECHS):
                checks[tech] = cols_t[i % 2].checkbox(tech, key=f"{prefix}_{tech}")
            techs = [t for t, v in checks.items() if v]
        return exp, mod, cont, techs

    exp1, mod1, cont1, techs1 = render_perfil_inputs(col_izq, "p1", "Perfil 1")
    exp2, mod2, cont2, techs2 = render_perfil_inputs(col_der, "p2", "Perfil 2")

    if st.button("Comparar perfiles", type="primary", use_container_width=True):

        def analizar(techs_s, exp, mod, cont):
            base = SALARIO_MEDIANA[exp]
            sal, _ = calcular_salario(base, techs_s, mod, cont)
            if modelo_ok:
                X = construir_vector(techs_s, exp, mod, cont, feature_cols)
                Xsc = scaler_model.transform(X)
                cid = int(kmeans_model.predict(Xsc)[0])
                dists = kmeans_model.transform(Xsc)[0]
                afin = 1 / (1 + dists)
                afin_n = afin / afin.sum() * 100
            else:
                cid = clasificar_por_reglas(techs_s, exp)
                afin_n = None
            return cid, afin_n, sal

        cid1, afin1, sal1 = analizar(techs1, exp1, mod1, cont1)
        cid2, afin2, sal2 = analizar(techs2, exp2, mod2, cont2)

        st.markdown("---")
        st.markdown('<div class="section-head code-style">Resultado de la Comparacion</div>', unsafe_allow_html=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Cluster Perfil 1", CLUSTER_NAMES[cid1].split(chr(0x2014))[0].strip())
        mc2.metric("Cluster Perfil 2", CLUSTER_NAMES[cid2].split(chr(0x2014))[0].strip())
        mc3.metric("Salario Perfil 1", f"${sal1:,.0f} COP")
        mc4.metric("Salario Perfil 2", f"${sal2:,.0f} COP", delta=f"${sal2-sal1:+,.0f} COP vs P1")

        fig_comp_sal = go.Figure(go.Bar(
            x=["Perfil 1", "Perfil 2"], y=[sal1, sal2],
            marker_color=[CLUSTER_COLORS[cid1], CLUSTER_COLORS[cid2]],
            text=[f"${sal1:,.0f}", f"${sal2:,.0f}"], textposition="outside",
        ))
        fig_comp_sal.update_layout(
            title=dict(text="Salario estimado comparado", font=dict(family="JetBrains Mono, Fira Code, monospace", color="#00f2fe")),
            height=300, margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Outfit, sans-serif", color="#e6edf3"),
            yaxis=dict(tickformat="$,.0f", gridcolor="#1f293d", zerolinecolor="#1f293d"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"), showlegend=False,
        )
        st.plotly_chart(fig_comp_sal, use_container_width=True)

        if afin1 is not None and afin2 is not None:
            categorias = [CLUSTER_NAMES[i].split(chr(0x2014))[1].strip() for i in range(4)]
            categorias_cierre = categorias + [categorias[0]]
            afin1_cierre = list(afin1) + [afin1[0]]
            afin2_cierre = list(afin2) + [afin2[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=afin1_cierre, theta=categorias_cierre, fill="toself", name="Perfil 1",
                line=dict(color=CLUSTER_COLORS[cid1], width=2),
                fillcolor=f"rgba({int(CLUSTER_COLORS[cid1][1:3],16)},{int(CLUSTER_COLORS[cid1][3:5],16)},{int(CLUSTER_COLORS[cid1][5:7],16)},0.15)"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=afin2_cierre, theta=categorias_cierre, fill="toself", name="Perfil 2",
                line=dict(color=CLUSTER_COLORS[cid2], width=2),
                fillcolor=f"rgba({int(CLUSTER_COLORS[cid2][1:3],16)},{int(CLUSTER_COLORS[cid2][3:5],16)},{int(CLUSTER_COLORS[cid2][5:7],16)},0.15)"
            ))
            fig_radar.update_layout(
                polar=dict(bgcolor="#121826",
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="#1f293d", color="#c9d1d9"),
                    angularaxis=dict(gridcolor="#1f293d", color="#c9d1d9"),
                ),
                title=dict(text="Afinidad por cluster - radar comparativo",
                           font=dict(family="JetBrains Mono, Fira Code, monospace", color="#00f2fe")),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Outfit, sans-serif", color="#e6edf3"),
                height=400, legend=dict(font=dict(color="#c9d1d9")),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        solo_p1 = set(techs1) - set(techs2)
        solo_p2 = set(techs2) - set(techs1)
        comunes = set(techs1) & set(techs2)
        td1, td2, td3 = st.columns(3)
        with td1:
            st.markdown(f"**Solo Perfil 1 ({len(solo_p1)})**")
            for t in sorted(solo_p1): st.markdown(f"- `{t}`")
        with td2:
            st.markdown(f"**Comunes ({len(comunes)})**")
            for t in sorted(comunes): st.markdown(f"- `{t}`")
        with td3:
            st.markdown(f"**Solo Perfil 2 ({len(solo_p2)})**")
            for t in sorted(solo_p2): st.markdown(f"- `{t}`")


# =========================================================
# PAGINA: VISUALIZACIONES
# =========================================================
elif pagina_actual == "📊 Visualizaciones":
    st.markdown(
        '<div class="title-container">'
        '<div class="main-title"><span class="main-title-text">'
        '<span class="title-prompt">&gt;_</span>Visualizaciones del Modelo</span></div>'
        '<div class="main-subtitle">Analisis grafico de los clusters sin reentrenamiento</div>'
        '<div class="accent-bar"></div>'
        '<div class="terminal-line"><span class="comment">// import matplotlib.pyplot as plt &middot; import plotly.graph_objects as go</span></div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-head code-style">1. Proyeccion PCA de los Centroides K-Means</div>', unsafe_allow_html=True)

    if not modelo_ok:
        st.markdown(
            '<div class="fallback-banner">'
            "<b>Grafico PCA no disponible</b> &mdash; Los archivos del modelo (.pkl) no estan cargados. "
            "Ejecuta el notebook para generar los modelos y ver esta visualizacion.</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="info-note">'
            "Este grafico muestra los <b>4 centroides</b> del modelo K-Means proyectados al "
            "espacio PCA de 2 componentes. No se muestran los 5.000 puntos individuales "
            "porque la app carga unicamente los modelos .pkl sin el CSV original.</div>",
            unsafe_allow_html=True
        )
        centroides_scaled = kmeans_model.cluster_centers_
        centroides_pca = pca_model.transform(centroides_scaled)
        var_exp = pca_model.explained_variance_ratio_

        fig_pca = go.Figure()
        for i in range(4):
            fig_pca.add_trace(go.Scatter(
                x=[centroides_pca[i, 0]], y=[centroides_pca[i, 1]],
                mode="markers+text",
                marker=dict(size=28, color=CLUSTER_COLORS[i], symbol="star", line=dict(width=2, color="#080b11")),
                text=[CLUSTER_NAMES[i].split(chr(0x2014))[0].strip()], textposition="top center",
                name=CLUSTER_NAMES[i],
                hovertemplate=(
                    f"<b>{CLUSTER_NAMES[i]}</b><br>"
                    f"PC1: {centroides_pca[i, 0]:.3f}<br>"
                    f"PC2: {centroides_pca[i, 1]:.3f}<br>"
                    f"Salario mediana: ${CLUSTER_INFO[i]['sal_median']:,.0f}<br>"
                    f"Ofertas: {CLUSTER_INFO[i]['n_ofertas']:,}<extra></extra>"
                )
            ))
        fig_pca.update_layout(
            title=dict(
                text=f"Centroides K-Means en espacio PCA &middot; PC1={var_exp[0]:.1%} PC2={var_exp[1]:.1%}",
                font=dict(family="JetBrains Mono, Fira Code, monospace", color="#00f2fe")
            ),
            xaxis_title=f"Componente Principal 1 ({var_exp[0]:.1%})",
            yaxis_title=f"Componente Principal 2 ({var_exp[1]:.1%})",
            height=450, margin=dict(l=40, r=40, t=60, b=120),
            plot_bgcolor="#121826", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Outfit, sans-serif", color="#e6edf3"),
            xaxis=dict(gridcolor="#1f293d", zerolinecolor="#1f293d"),
            yaxis=dict(gridcolor="#1f293d", zerolinecolor="#1f293d"),
            legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5,
                        font=dict(color="#c9d1d9", size=11)),
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown('<div class="section-head code-style">2. Varianza Explicada por PCA</div>', unsafe_allow_html=True)

    if not modelo_ok:
        st.markdown(
            '<div class="fallback-banner">'
            "<b>Varianza PCA no disponible</b> &mdash; Se requieren los archivos .pkl.</div>",
            unsafe_allow_html=True
        )
    else:
        if pca_variance is not None:
            var_ratio = pca_variance
            var_cum = np.cumsum(var_ratio)
            n_comp = len(var_ratio)
            nota_var = f"Varianza total de los {n_comp} componentes (generada desde <code>pca_full_variance.pkl</code>)."
        else:
            var_ratio = pca_model.explained_variance_ratio_
            var_cum = np.cumsum(var_ratio)
            n_comp = len(var_ratio)
            nota_var = (
                "Solo se muestran PC1 y PC2 porque el PCA fue guardado con "
                "<code>n_components=2</code>. Para la curva completa, regenera los modelos "
                "desde el notebook (seccion 13) &mdash; creara <code>pca_full_variance.pkl</code>."
            )

        st.markdown(f'<div class="info-note">{nota_var}</div>', unsafe_allow_html=True)

        fig_var = make_subplots(specs=[[{"secondary_y": True}]])
        fig_var.add_trace(
            go.Bar(x=[f"PC{i+1}" for i in range(n_comp)], y=var_ratio * 100,
                   name="Varianza individual", marker_color="#00d2ff"), secondary_y=False
        )
        fig_var.add_trace(
            go.Scatter(x=[f"PC{i+1}" for i in range(n_comp)], y=var_cum * 100,
                       name="Varianza acumulada", mode="lines+markers",
                       marker=dict(color="#e040fb"), line=dict(color="#e040fb", width=2)),
            secondary_y=True
        )
        if pca_variance is not None:
            for level, c in [(80, "#00e676"), (90, "#ff9f1c")]:
                fig_var.add_hline(y=level, line_dash="dash", line_color=c,
                                  annotation_text=f"{level}%", annotation_font=dict(color=c), secondary_y=True)
        fig_var.update_layout(
            title=dict(text="Varianza Explicada por Componente PCA",
                       font=dict(family="JetBrains Mono, Fira Code, monospace", color="#00f2fe")),
            height=400, margin=dict(l=40, r=40, t=60, b=80),
            plot_bgcolor="#121826", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Outfit, sans-serif", color="#e6edf3"),
            xaxis=dict(gridcolor="#1f293d", zerolinecolor="#1f293d"),
            yaxis=dict(gridcolor="#1f293d", zerolinecolor="#1f293d"),
            yaxis2=dict(gridcolor="rgba(0,0,0,0)"),
            legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5,
                        font=dict(color="#c9d1d9", size=11)),
        )
        fig_var.update_yaxes(title_text="Varianza individual (%)", secondary_y=False)
        fig_var.update_yaxes(title_text="Varianza acumulada (%)", secondary_y=True)
        st.plotly_chart(fig_var, use_container_width=True)

    st.markdown('<div class="section-head code-style">3. Distribucion de Ofertas por Cluster</div>', unsafe_allow_html=True)
    df_dist = pd.DataFrame({
        "Cluster": [CLUSTER_NAMES[i] for i in range(4)],
        "Ofertas": [CLUSTER_INFO[i]["n_ofertas"] for i in range(4)],
    })
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        fig_bar = px.bar(df_dist, x="Ofertas", y="Cluster", orientation="h", color="Cluster",
                         color_discrete_map={CLUSTER_NAMES[i]: CLUSTER_COLORS[i] for i in range(4)},
                         text="Ofertas", title="Numero de ofertas por cluster")
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            title=dict(text="Numero de ofertas por cluster",
                       font=dict(family="JetBrains Mono, Fira Code, monospace", color="#00f2fe")),
            height=320, showlegend=False, plot_bgcolor="#121826", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Outfit, sans-serif", color="#e6edf3"),
            xaxis=dict(gridcolor="#1f293d", zerolinecolor="#1f293d"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"), margin=dict(l=0, r=40, t=40, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_p2:
        fig_pie = px.pie(df_dist, values="Ofertas", names="Cluster", color="Cluster",
                         color_discrete_map={CLUSTER_NAMES[i]: CLUSTER_COLORS[i] for i in range(4)})
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(
            title=dict(text="Proporcion por cluster",
                       font=dict(family="JetBrains Mono, Fira Code, monospace", color="#00f2fe")),
            height=320, showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Outfit, sans-serif", color="#e6edf3"),
            margin=dict(l=0, r=0, t=40, b=20)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<div class="section-head code-style">4. Rango Salarial por Cluster</div>', unsafe_allow_html=True)
    sal_min_list = [CLUSTER_INFO[i]["sal_min"] for i in range(4)]
    sal_med_list = [CLUSTER_INFO[i]["sal_median"] for i in range(4)]
    sal_max_list = [CLUSTER_INFO[i]["sal_max"] for i in range(4)]

    fig_sal = go.Figure()
    for i in range(4):
        fig_sal.add_trace(go.Bar(
            x=[CLUSTER_NAMES[i]], y=[sal_max_list[i] - sal_min_list[i]], base=[sal_min_list[i]],
            marker_color=CLUSTER_COLORS[i], opacity=0.45, showlegend=False,
        ))
        fig_sal.add_trace(go.Scatter(
            x=[CLUSTER_NAMES[i]], y=[sal_med_list[i]], mode="markers",
            marker=dict(color=CLUSTER_COLORS[i], size=14, symbol="diamond", line=dict(width=2, color="#080b11")),
            name=f"{CLUSTER_NAMES[i]} (mediana)", showlegend=False,
        ))
    fig_sal.update_layout(
        title=dict(text="Rango salarial (barra) y mediana (diamante)",
                   font=dict(family="JetBrains Mono, Fira Code, monospace", color="#00f2fe")),
        yaxis=dict(title="Salario (COP)", tickformat="$,.0f", gridcolor="#1f293d", zerolinecolor="#1f293d"),
        barmode="overlay", height=420, plot_bgcolor="#121826", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", color="#e6edf3"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_sal, use_container_width=True)

    st.markdown('<div class="section-head code-style">5. Perfil Tecnologico por Cluster</div>', unsafe_allow_html=True)
    TECH_RATES = {
        0: {"Java": 0.46, "R": 0.46, "Tableau": 0.30, "Power BI": 0.29, "React": 0.28,
            "Python": 0.25, "SQL": 0.24, "AWS": 0.22, "Excel": 0.21, "Pandas": 0.20},
        1: {"R": 0.48, "Python": 0.30, "Tableau": 0.28, "SQL": 0.27, "Pandas": 0.27,
            "Excel": 0.26, "TensorFlow": 0.25, "Power BI": 0.24, "AWS": 0.23, "JavaScript": 0.21},
        2: {"JavaScript": 1.0, "Java": 1.0, "R": 0.42, "Pandas": 0.28, "Tableau": 0.27,
            "SQL": 0.26, "React": 0.25, "Node.js": 0.24, "Python": 0.22, "Excel": 0.21},
        3: {"R": 0.50, "React": 0.29, "Pandas": 0.29, "Power BI": 0.28, "SQL": 0.28,
            "Excel": 0.27, "Python": 0.26, "Tableau": 0.25, "TensorFlow": 0.24, "JavaScript": 0.23},
    }
    fig_techs = make_subplots(rows=2, cols=2, subplot_titles=[CLUSTER_NAMES[i] for i in range(4)],
                              horizontal_spacing=0.10, vertical_spacing=0.14)
    for i, (r, c) in enumerate([(1, 1), (1, 2), (2, 1), (2, 2)]):
        techs_r = list(TECH_RATES[i].keys())
        rates = list(TECH_RATES[i].values())
        fig_techs.add_trace(
            go.Bar(x=rates, y=techs_r, orientation="h", marker_color=CLUSTER_COLORS[i],
                   text=[f"{v:.0%}" for v in rates], textposition="outside",
                   name=CLUSTER_NAMES[i], showlegend=False),
            row=r, col=c
        )
        fig_techs.update_xaxes(range=[0, 1.25], tickformat=".0%", row=r, col=c,
                               gridcolor="#1f293d", zerolinecolor="#1f293d")
        fig_techs.update_yaxes(row=r, col=c, gridcolor="rgba(0,0,0,0)")

    fig_techs.update_layout(
        title=dict(text="% de ofertas que requieren cada tecnologia, por cluster",
                   font=dict(family="JetBrains Mono, Fira Code, monospace", color="#00f2fe")),
        height=650, plot_bgcolor="#121826", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", color="#e6edf3"),
    )
    fig_techs.update_annotations(font=dict(family="JetBrains Mono, Fira Code, monospace", size=10, color="#c9d1d9"))
    fig_techs.update_xaxes(gridcolor="#1f293d", zerolinecolor="#1f293d")
    fig_techs.update_yaxes(gridcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_techs, use_container_width=True)

    st.markdown('<div class="section-head code-style">6. Distribucion Salarial Simulada por Cluster</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-note">Distribucion salarial simulada a partir de los rangos y medianas conocidos '
        "de cada cluster (distribucion beta escalada). Util para visualizar la dispersion intra-cluster.</div>",
        unsafe_allow_html=True
    )
    fig_hist = go.Figure()
    for i in range(4):
        muestras = generar_distribucion_salarial(i)
        info_c = CLUSTER_INFO[i]
        fig_hist.add_trace(go.Histogram(
            x=muestras, name=CLUSTER_NAMES[i].split(chr(0x2014))[1].strip(),
            marker_color=CLUSTER_COLORS[i], opacity=0.65, nbinsx=40,
            hovertemplate="Rango: %{x:$,.0f}<br>Ofertas: %{y}<extra>" +
                          CLUSTER_NAMES[i].split(chr(0x2014))[1].strip() + "</extra>",
        ))
        fig_hist.add_vline(
            x=info_c["sal_median"], line_dash="dash", line_color=CLUSTER_COLORS[i], line_width=1.5,
            annotation_text=f"Med {i}", annotation_font_color=CLUSTER_COLORS[i], annotation_font_size=10,
        )

    fig_hist.update_layout(
        barmode="overlay",
        title=dict(text="Distribucion salarial simulada con medianas (lineas punteadas)",
                   font=dict(family="JetBrains Mono, Fira Code, monospace", color="#00f2fe")),
        xaxis=dict(title="Salario (COP)", tickformat="$,.0f", gridcolor="#1f293d", zerolinecolor="#1f293d"),
        yaxis=dict(title="Frecuencia simulada", gridcolor="#1f293d", zerolinecolor="#1f293d"),
        height=430, plot_bgcolor="#121826", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", color="#e6edf3"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5,
                    font=dict(color="#c9d1d9", size=11)),
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# =========================================================
# PAGINA: METODOLOGIA
# =========================================================
elif pagina_actual == "📖 Metodologia":
    st.markdown(
        '<div class="title-container">'
        '<div class="main-title"><span class="main-title-text">'
        '<span class="title-prompt">&gt;_</span>Metodologia</span></div>'
        '<div class="main-subtitle">Marco metodologico &middot; SENA Funza &middot; '
        "Analisis de Empleo Digital Colombia</div>"
        '<div class="accent-bar"></div>'
        '<div class="terminal-line"><span class="comment">// from sklearn.cluster import KMeans &middot; from sklearn.decomposition import PCA</span></div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-head">Que es K-Means y por que se eligio K=4?</div>', unsafe_allow_html=True)
    st.markdown("""
K-Means es un algoritmo de **aprendizaje no supervisado** que agrupa observaciones en *K* clusters,
minimizando la suma de distancias cuadradas desde cada punto hasta el centroide de su cluster (inercia).

**Por que K=4 y no K=3:**
Aunque el coeficiente de silueta es maximo en K=3 (0.0890), se eligio K=4 por cuatro razones:
1. **Interpretabilidad**: genera segmentos alineados con los 4 niveles del mercado laboral colombiano.
2. **Gradiente salarial**: con K=3 los niveles Junior y Junior-Intermedio quedan fusionados, ocultando una diferencia de $1.4M COP en medianas salariales.
3. **Validez del codo**: la caida de inercia entre K=3 y K=4 es mayor que entre K=4 y K=5.
4. **Alineacion con el proyecto**: el documento metodologico del SENA identifica 4 perfiles de formacion diferenciados.
    """)

    st.markdown('<div class="section-head">Simulador Interactivo &mdash; Curva del Codo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-note">Arrastra el slider para explorar como cambia la inercia del modelo '
        "al variar el numero de clusters K. Los valores estan precalculados sobre el dataset de 5.000 ofertas.</div>",
        unsafe_allow_html=True
    )
    _INERCIAS = {2: 4_312, 3: 3_185, 4: 2_098, 5: 1_824, 6: 1_612, 7: 1_471, 8: 1_358}
    _SILUETAS = {2: 0.063, 3: 0.089, 4: 0.081, 5: 0.074, 6: 0.068, 7: 0.061, 8: 0.055}

    k_sel = st.slider(
        "Numero de clusters K", min_value=2, max_value=8, value=4, step=1,
        help="K=4 es el valor elegido en el proyecto. Observa la caida de inercia."
    )
    _fig_codo = go.Figure()
    _fig_codo.add_trace(go.Scatter(
        x=list(_INERCIAS.keys()), y=list(_INERCIAS.values()), mode="lines+markers",
        line=dict(color="#00f2fe", width=2),
        marker=dict(
            size=[18 if k == k_sel else 10 for k in _INERCIAS],
            color=["#00e676" if k == k_sel else "#00f2fe" for k in _INERCIAS],
            line=dict(width=2, color="#080b11")
        ),
        name="Inercia", hovertemplate="K=%{x}<br>Inercia=%{y:,.0f}<extra></extra>",
    ))
    _fig_codo.add_vline(
        x=k_sel, line_dash="dot", line_color="#00e676", line_width=1.5,
        annotation_text=f"K={k_sel} seleccionado", annotation_font_color="#00e676", annotation_font_size=11,
    )
    _fig_codo.update_layout(
        title=dict(
            text=f"Curva del Codo &mdash; Inercia por K | K={k_sel}: inercia={_INERCIAS[k_sel]:,}, silueta={_SILUETAS[k_sel]:.3f}",
            font=dict(family="JetBrains Mono, Fira Code, monospace", color="#00f2fe", size=12)
        ),
        xaxis=dict(title="Numero de clusters (K)", tickmode="linear", gridcolor="#1f293d", zerolinecolor="#1f293d"),
        yaxis=dict(title="Inercia (suma de distancias^2)", gridcolor="#1f293d", zerolinecolor="#1f293d"),
        height=320, margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", color="#e6edf3"), showlegend=False,
    )
    st.plotly_chart(_fig_codo, use_container_width=True)
    if k_sel == 4:
        st.success("K=4 &mdash; Punto elegido en el proyecto. Buen equilibrio entre inercia e interpretabilidad.")
    elif k_sel < 4:
        st.info(f"K={k_sel}: clusters mas grandes y heterogeneos. Se pierden matices.")
    else:
        st.info(f"K={k_sel}: mayor granularidad, pero los clusters son mas dificiles de interpretar.")

    st.markdown('<div class="section-head">Para que se uso PCA?</div>', unsafe_allow_html=True)
    st.markdown("""
**PCA (Analisis de Componentes Principales)** reduce dimensionalidad. En este proyecto se aplica
**despues** del clustering, exclusivamente para **visualizacion**:

1. La matriz de 29 variables no es directamente visualizable en 2D.
2. PCA proyecta en **2 componentes principales** (PC1 y PC2).
3. Permite ver graficamente la separacion entre los 4 clusters.

> **Importante:** el K-Means opera sobre las 29 variables escaladas originales. PCA es solo para visualizar.
    """)

    st.markdown('<div class="section-head">Interpretacion de los Clusters</div>', unsafe_allow_html=True)
    for c in range(4):
        info = CLUSTER_INFO[c]
        color = CLUSTER_COLORS[c]
        st.markdown(f"""
        <div style="border-left:4px solid {color};padding:14px 18px;margin:12px 0;
                    background:#121826;border-radius:0 8px 8px 0;border:1px solid #1f293d;border-left-width:4px">
          <b style="color:{color};font-family:'JetBrains Mono','Fira Code',monospace;font-size:0.95rem">{CLUSTER_NAMES[c]}</b><br>
          <div style="font-size:0.88rem;color:#c9d1d9;margin-top:6px;line-height:1.4">
          Representa perfiles con <b>{info["experiencia"]}</b>, rango salarial de
          <b>{info["salario"]} COP/mes</b> y stack frecuente: <b>{info["tecnologias"]}</b>.<br>
          Modalidad predominante: <b>{info["modalidad"]}</b>.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-head">Pipeline del Proyecto</div>', unsafe_allow_html=True)
    st.code("""
Dataset bruto (5.000 registros, CSV sucio)
     |
     V
Limpieza de datos
  |-- Normalizacion de categorias (experiencia, modalidad, ciudad, tipo_contrato)
  |-- Imputacion de salarios nulos (mediana por nivel de experiencia)
     |
     V
Feature Engineering
  |-- 15 variables binarias por tecnologia
  |-- One-Hot Encoding: experiencia, modalidad, tipo_contrato
  |-- Variables numericas: num_lenguajes, salario, vacantes, mes_publicacion
     |
     V
StandardScaler (29 variables) -> K-Means (K=4) -> PCA (2 componentes)
     |
     V
Exportacion .pkl -> cargado en esta app Streamlit (sin reentrenamiento)
    """)

    st.markdown('<div class="section-head">Variables del Modelo (29 columnas)</div>', unsafe_allow_html=True)
    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        st.markdown("**Numericas (4)**")
        for col in ["num_lenguajes", "salario", "vacantes", "mes_publicacion"]:
            st.markdown(f"- `{col}`")
    with col_v2:
        st.markdown("**Tecnologias (15)**")
        for t in TOP_TECHS:
            st.markdown(f"- `tiene_{t.lower().replace(' ', '_').replace('.', '')}`")
    with col_v3:
        st.markdown("**One-Hot (10)**")
        for col in ["exp_Junior", "exp_Semi-Senior", "exp_Senior",
                    "mod_Hibrido", "mod_Presencial", "mod_Remoto",
                    "contrato_Contrato", "contrato_Freelance",
                    "contrato_Indefinido", "contrato_Temporal"]:
            st.markdown(f"- `{col}`")


