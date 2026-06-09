# constants.py – Centraliza todas las configuraciones y literals del proyecto

# Tecnologías que pueden seleccionar los usuarios
TOP_TECHS = [
    'Python', 'SQL', 'JavaScript', 'React', 'AWS',
    'Azure', 'Docker', 'TensorFlow', 'Power BI', 'Tableau',
    'Node.js', 'Excel', 'Pandas', 'R', 'Java'
]

# Salario base (mediana) por nivel de experiencia
SALARIO_MEDIANA = {
    'Junior':      3_228_080,
    'Semi-Senior': 5_321_101,
    'Senior':      9_085_076,
}

# Multiplicadores por tecnología (bonificaciones)
TECH_MULTIPLIERS = {
    'TensorFlow': 0.10,
    'AWS':        0.08,
    'Azure':      0.06,
    'Python':     0.06,
    'Docker':     0.05,
    'React':      0.05,
    'Node.js':    0.05,
    'Java':       0.05,
    'SQL':        0.04,
    'Tableau':    0.04,
    'Power BI':   0.04,
    'JavaScript': 0.03,
    'Pandas':     0.03,
    'R':          0.03,
    'Excel':      0.00,
}

# Multiplicadores por modalidad de trabajo
MODALIDAD_MULTIPLIERS = {
    'Remoto':     0.08,
    'Híbrido':    0.00,
    'Presencial': -0.05,
}

# Multiplicadores por tipo de contrato
CONTRATO_MULTIPLIERS = {
    'Freelance':   0.15,
    'Contrato':    0.05,
    'Indefinido':  0.00,
    'Temporal':   -0.10,
}

# Paleta de colores por clúster (usada en UI)
# Contraste WCAG: 0=#00d2ff(~8:1 AAA) 1=#ff9f1c(~6:1 AA) 2=#00e676(~7:1 AA) 3=#c026d3(~5.5:1 AA)
# #c026d3 tiene mayor luminosidad que #d500f9 y #e040fb — mejor visibilidad sobre fondos oscuros de Plotly (#121826)
CLUSTER_COLORS = {
    0: '#00d2ff',
    1: '#ff9f1c',
    2: '#00e676',
    3: '#c026d3',  # magenta con luminosidad óptima para fondos oscuros (#121826)
}

# Nombres amigables por clúster
CLUSTER_NAMES = {
    0: 'Clúster A — Senior / Alta Especialización',
    1: 'Clúster B — Semi-Senior / Perfil Intermedio',
    2: 'Clúster C — Junior-Intermedio / Stack Fullstack',
    3: 'Clúster D — Junior / Perfil de Entrada',
}

# Información resumida por clúster (usada en cards)
CLUSTER_INFO = {
    0: {
        'nombre':        'Senior / Alta Especialización',
        'experiencia':   'Senior (5+ años)',
        'salario':       '$7.800.000 – $10.300.000',
        'tecnologias':   'Java, R, Tableau, Power BI, React',
        'modalidad':     'Remoto / Híbrido predominante',
        'recomendacion': 'Profundizar en arquitecturas cloud (AWS/Azure), IA generativa y certificaciones de liderazgo técnico.',
        'n_ofertas':     992,
        'sal_median':    9_085_076,
        'sal_min':       7_800_000,
        'sal_max':       10_300_000,
    },
    1: {
        'nombre':        'Semi-Senior / Perfil Intermedio',
        'experiencia':   'Semi-Senior (2-5 años)',
        'salario':       '$3.800.000 – $6.800.000',
        'tecnologias':   'R, Python, Tableau, SQL, Pandas',
        'modalidad':     'Remoto / Híbrido',
        'recomendacion': 'Certificación en Power BI, SQL avanzado y proyectos con TensorFlow o Scikit-learn.',
        'n_ofertas':     1_392,
        'sal_median':    5_321_101,
        'sal_min':       3_800_000,
        'sal_max':       6_800_000,
    },
    2: {
        'nombre':        'Junior-Intermedio / Stack Fullstack',
        'experiencia':   'Junior a Semi-Senior',
        'salario':       '$1.800.000 – $6.800.000',
        'tecnologias':   'JavaScript, Java, R, Pandas, Tableau',
        'modalidad':     'Remoto / Híbrido / Presencial',
        'recomendacion': 'Fortalecer React, Node.js y fundamentos de bases de datos (SQL/NoSQL).',
        'n_ofertas':     1_097,
        'sal_median':    4_285_769,
        'sal_min':       1_800_000,
        'sal_max':       6_800_000,
    },
    3: {
        'nombre':        'Junior / Perfil de Entrada',
        'experiencia':   'Junior (0-2 años)',
        'salario':       '$1.800.000 – $4.600.000',
        'tecnologias':   'R, React, Pandas, Power BI, SQL',
        'modalidad':     'Remoto / Híbrido / Presencial',
        'recomendacion': 'Iniciar con Python (Pandas, NumPy), SQL básico y Power BI. Construir portafolio en GitHub.',
        'n_ofertas':     1_519,
        'sal_median':    3_211_474,
        'sal_min':       1_800_000,
        'sal_max':       4_600_000,
    },
}

# Lista de columnas usadas por el modelo (puede cambiarse con PCA completa)
FEATURE_COLUMNS_DEFAULT = [
    'num_lenguajes', 'salario', 'vacantes', 'mes_publicacion',
    'tiene_python', 'tiene_sql', 'tiene_javascript', 'tiene_react',
    'tiene_aws', 'tiene_azure', 'tiene_docker', 'tiene_tensorflow',
    'tiene_power_bi', 'tiene_tableau', 'tiene_nodejs', 'tiene_excel',
    'tiene_pandas', 'tiene_r', 'tiene_java',
    'exp_Junior', 'exp_Semi-Senior', 'exp_Senior',
    'mod_Híbrido', 'mod_Presencial', 'mod_Remoto',
    'contrato_Contrato', 'contrato_Freelance', 'contrato_Indefinido', 'contrato_Temporal',
]

# Mapeo de nombres de archivos .pkl
PKL_FILES = {
    'kmeans':        'kmeans_empleo_digital.pkl',
    'pca':           'pca_empleo_digital.pkl',
    'scaler':        'scaler_empleo_digital.pkl',
    'features':      'feature_columns.pkl',
    'pca_variance':  'pca_full_variance.pkl',   # opcional
}

# CSS styles can be imported as a raw string from utils if needed
