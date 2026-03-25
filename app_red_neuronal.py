# -*- coding: utf-8 -*-
"""
Aplicación de Reconstrucción de Trayectorias de Béisbol con Redes Neuronales
Basado en: "Robust baseball pitch reconstruction using artificial neural networks"
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import threading
import warnings
warnings.filterwarnings('ignore')

# Colores del tema
COLORES = {
    'fondo_principal': '#1a1a2e',
    'fondo_secundario': '#16213e',
    'fondo_panel': '#0f3460',
    'acento': '#e94560',
    'acento_secundario': '#533483',
    'texto': '#eaeaea',
    'texto_secundario': '#a0a0a0',
    'exito': '#00d4aa',
    'advertencia': '#ffc107',
    'grafico_linea1': '#00d4aa',
    'grafico_linea2': '#e94560',
    'grafico_linea3': '#ffc107',
    'grafico_fondo': '#0f3460'
}


class SimuladorBeisbol:
    """Simulador de trayectorias de béisbol basado en física"""
    
    def __init__(self):
        self.g = 9.81  # Aceleración de gravedad (m/s²)
        self.B = 0.00041  # Factor de escala magnus
        self.v_d = 35.0  # m/s
        self.delta = 5.0  # m/s
        
    def fuerza_arrastre(self, v_vec):
        """Calcula la aceleración por arrastre"""
        v = np.linalg.norm(v_vec)
        if v < 0.01:
            return np.zeros(3)
        coef = 0.0039 + 0.0058 / (1 + np.exp((v - self.v_d) / self.delta))
        return -coef * v * v_vec
    
    def fuerza_magnus(self, omega_vec, v_vec):
        """Calcula la aceleración por efecto Magnus"""
        return self.B * np.cross(omega_vec, v_vec)
    
    def fuerza_gravedad(self):
        """Retorna la aceleración por gravedad"""
        return np.array([0, 0, -self.g])
    
    def derivadas(self, estado, omega_vec):
        """Calcula las derivadas para integración RK4"""
        pos = estado[:3]
        vel = estado[3:]
        
        a_total = (self.fuerza_arrastre(vel) + 
                   self.fuerza_magnus(omega_vec, vel) + 
                   self.fuerza_gravedad())
        
        return np.concatenate([vel, a_total])
    
    def runge_kutta_4(self, estado, omega_vec, dt):
        """Integración Runge-Kutta de 4to orden"""
        k1 = self.derivadas(estado, omega_vec)
        k2 = self.derivadas(estado + 0.5 * dt * k1, omega_vec)
        k3 = self.derivadas(estado + 0.5 * dt * k2, omega_vec)
        k4 = self.derivadas(estado + dt * k3, omega_vec)
        
        return estado + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def simular_lanzamiento(self, v0_mph, spin_rpm, phi_deg=45, alpha_deg=1, 
                            x0=np.zeros(3), t_final=0.5, dt=0.002083):
        """
        Simula una trayectoria completa de lanzamiento
        
        Args:
            v0_mph: Velocidad inicial en mph
            spin_rpm: Tasa de rotación en rpm
            phi_deg: Ángulo del eje de rotación (grados)
            alpha_deg: Ángulo de lanzamiento (grados)
            x0: Posición inicial
            t_final: Tiempo final de simulación
            dt: Paso de tiempo
            
        Returns:
            t_array: Array de tiempos
            trayectoria: Array de posiciones (N x 3)
        """
        # Convertir unidades
        v0 = v0_mph * 0.44704  # mph a m/s
        omega = spin_rpm * 2 * np.pi / 60  # rpm a rad/s
        phi = np.radians(phi_deg)
        alpha = np.radians(alpha_deg)
        
        # Velocidad inicial
        vel0 = np.array([v0 * np.cos(alpha), 0, v0 * np.sin(alpha)])
        
        # Eje de rotación (sin componente en x)
        omega_vec = omega * np.array([0, np.sin(phi), np.cos(phi)])
        
        # Estado inicial [x, y, z, vx, vy, vz]
        estado = np.concatenate([x0, vel0])
        
        # Integración temporal
        n_pasos = int(t_final / dt) + 1
        t_array = np.linspace(0, t_final, n_pasos)
        trayectoria = np.zeros((n_pasos, 3))
        trayectoria[0] = x0
        
        for i in range(1, n_pasos):
            estado = self.runge_kutta_4(estado, omega_vec, dt)
            trayectoria[i] = estado[:3]
            
        return t_array, trayectoria
    
    def obtener_tres_puntos(self, trayectoria, t_array, fps=60):
        """Obtiene 3 puntos espaciotemporales a 60 fps"""
        dt_frame = 1.0 / fps
        
        # Índices de los tres puntos (últimos tres frames a 60 fps)
        t3 = t_array[-1]
        t2 = t3 - dt_frame
        t1 = t2 - dt_frame
        
        idx1 = np.argmin(np.abs(t_array - t1))
        idx2 = np.argmin(np.abs(t_array - t2))
        idx3 = np.argmin(np.abs(t_array - t3))
        
        return np.array([trayectoria[idx1], trayectoria[idx2], trayectoria[idx3]])
    
    def generar_datos_entrenamiento(self, n_muestras, v_min=70, v_max=100, 
                                     spin_min=800, spin_max=3300, callback=None):
        """Genera datos sintéticos para entrenamiento"""
        X = []  # Tres puntos (entrada)
        y = []  # Velocidad y spin (salida)
        
        for i in range(n_muestras):
            # Muestrear aleatoriamente
            v0 = np.random.uniform(v_min, v_max)
            spin = np.random.uniform(spin_min, spin_max)
            
            # Simular trayectoria
            t_array, trayectoria = self.simular_lanzamiento(v0, spin)
            
            # Obtener tres puntos
            tres_puntos = self.obtener_tres_puntos(trayectoria, t_array)
            
            X.append(tres_puntos.flatten())
            y.append([v0, spin])
            
            if callback and (i + 1) % max(1, n_muestras // 100) == 0:
                callback(i + 1, n_muestras)
        
        return np.array(X), np.array(y)


class RedNeuronal:
    """Red Neuronal para reconstrucción de trayectorias"""
    
    def __init__(self, usar_tensorflow=True):
        self.modelo = None
        self.historial = None
        self.escalador_X = None
        self.escalador_y = None
        self.usar_tensorflow = usar_tensorflow
        
        # Intentar importar TensorFlow
        try:
            import tensorflow as tf
            self.tf = tf
            tf.get_logger().setLevel('ERROR')
        except ImportError:
            self.usar_tensorflow = False
            print("TensorFlow no disponible, usando implementación básica")
    
    def normalizar_datos(self, X, y):
        """Normaliza los datos de entrada y salida"""
        self.X_min = X.min(axis=0)
        self.X_max = X.max(axis=0)
        self.y_min = y.min(axis=0)
        self.y_max = y.max(axis=0)
        
        X_norm = (X - self.X_min) / (self.X_max - self.X_min + 1e-8)
        y_norm = (y - self.y_min) / (self.y_max - self.y_min + 1e-8)
        
        return X_norm, y_norm
    
    def desnormalizar_y(self, y_norm):
        """Desnormaliza la salida"""
        return y_norm * (self.y_max - self.y_min) + self.y_min
    
    def construir_modelo(self, capas_ocultas=3, neuronas=128):
        """Construye el modelo de red neuronal"""
        if self.usar_tensorflow:
            from tensorflow import keras
            
            modelo = keras.Sequential([
                keras.layers.Input(shape=(9,)),
            ])
            
            for _ in range(capas_ocultas):
                modelo.add(keras.layers.Dense(neuronas, activation='relu'))
            
            modelo.add(keras.layers.Dense(2, activation='sigmoid'))
            
            modelo.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            self.modelo = modelo
        else:
            # Implementación básica sin TensorFlow
            self.pesos = []
            self.sesgos = []
            
            capas = [9] + [neuronas] * capas_ocultas + [2]
            
            for i in range(len(capas) - 1):
                w = np.random.randn(capas[i], capas[i+1]) * 0.1
                b = np.zeros(capas[i+1])
                self.pesos.append(w)
                self.sesgos.append(b)
    
    def entrenar(self, X, y, epochs=100, batch_size=20, ruido=0, 
                 validacion=0.2, callback=None):
        """Entrena la red neuronal"""
        # Inyectar ruido si se especifica
        if ruido > 0:
            X = X + np.random.uniform(-ruido/1000, ruido/1000, X.shape)
        
        # Normalizar datos
        X_norm, y_norm = self.normalizar_datos(X, y)
        
        # Dividir en entrenamiento y validación
        n_val = int(len(X) * validacion)
        indices = np.random.permutation(len(X))
        
        X_train = X_norm[indices[n_val:]]
        y_train = y_norm[indices[n_val:]]
        X_val = X_norm[indices[:n_val]]
        y_val = y_norm[indices[:n_val]]
        
        if self.usar_tensorflow:
            # Callback personalizado para progreso
            class ProgresoCallback(self.tf.keras.callbacks.Callback):
                def __init__(self, callback_fn, total_epochs):
                    super().__init__()
                    self.callback_fn = callback_fn
                    self.total_epochs = total_epochs
                    
                def on_epoch_end(self, epoch, logs=None):
                    if self.callback_fn:
                        self.callback_fn(epoch + 1, self.total_epochs, 
                                        logs.get('loss', 0), 
                                        logs.get('val_loss', 0))
            
            callbacks = []
            if callback:
                callbacks.append(ProgresoCallback(callback, epochs))
            
            self.historial = self.modelo.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                verbose=0,
                callbacks=callbacks
            )
            
            return {
                'loss': self.historial.history['loss'],
                'val_loss': self.historial.history['val_loss']
            }
        else:
            # Entrenamiento básico (descenso de gradiente)
            perdidas_train = []
            perdidas_val = []
            lr = 0.01
            
            for epoch in range(epochs):
                # Forward pass
                a = X_train
                activaciones = [a]
                
                for i, (w, b) in enumerate(zip(self.pesos, self.sesgos)):
                    z = np.dot(a, w) + b
                    if i < len(self.pesos) - 1:
                        a = np.maximum(0, z)  # ReLU
                    else:
                        a = 1 / (1 + np.exp(-z))  # Sigmoid
                    activaciones.append(a)
                
                # Calcular pérdida
                loss = np.mean((a - y_train) ** 2)
                perdidas_train.append(loss)
                
                # Validación
                a_val = X_val
                for i, (w, b) in enumerate(zip(self.pesos, self.sesgos)):
                    z = np.dot(a_val, w) + b
                    if i < len(self.pesos) - 1:
                        a_val = np.maximum(0, z)
                    else:
                        a_val = 1 / (1 + np.exp(-z))
                
                val_loss = np.mean((a_val - y_val) ** 2)
                perdidas_val.append(val_loss)
                
                # Backpropagation simplificado
                delta = (a - y_train) * a * (1 - a)
                
                for i in range(len(self.pesos) - 1, -1, -1):
                    grad_w = np.dot(activaciones[i].T, delta) / len(X_train)
                    grad_b = np.mean(delta, axis=0)
                    
                    self.pesos[i] -= lr * grad_w
                    self.sesgos[i] -= lr * grad_b
                    
                    if i > 0:
                        delta = np.dot(delta, self.pesos[i].T)
                        delta = delta * (activaciones[i] > 0)
                
                if callback:
                    callback(epoch + 1, epochs, loss, val_loss)
            
            return {'loss': perdidas_train, 'val_loss': perdidas_val}
    
    def predecir(self, X):
        """Realiza predicciones"""
        X_norm = (X - self.X_min) / (self.X_max - self.X_min + 1e-8)
        
        if self.usar_tensorflow:
            y_norm = self.modelo.predict(X_norm, verbose=0)
        else:
            a = X_norm
            for i, (w, b) in enumerate(zip(self.pesos, self.sesgos)):
                z = np.dot(a, w) + b
                if i < len(self.pesos) - 1:
                    a = np.maximum(0, z)
                else:
                    a = 1 / (1 + np.exp(-z))
            y_norm = a
        
        return self.desnormalizar_y(y_norm)


class AplicacionPrincipal:
    """Aplicación principal con interfaz gráfica"""
    
    def __init__(self):
        self.ventana = tk.Tk()
        self.ventana.title("🎯 Reconstrucción de Trayectorias de Béisbol - RNA")
        self.ventana.geometry("1400x900")
        self.ventana.configure(bg=COLORES['fondo_principal'])
        
        # Configurar estilo
        self.configurar_estilo()
        
        # Componentes
        self.simulador = SimuladorBeisbol()
        self.red_neuronal = None
        self.datos_X = None
        self.datos_y = None
        self.historial_entrenamiento = None
        
        # Crear interfaz
        self.crear_interfaz()
        
    def configurar_estilo(self):
        """Configura el estilo de los widgets"""
        estilo = ttk.Style()
        estilo.theme_use('clam')
        
        # Configurar colores para ttk widgets
        estilo.configure('TFrame', background=COLORES['fondo_principal'])
        estilo.configure('TLabel', 
                        background=COLORES['fondo_principal'], 
                        foreground=COLORES['texto'],
                        font=('Segoe UI', 10))
        estilo.configure('TLabelframe', 
                        background=COLORES['fondo_secundario'],
                        foreground=COLORES['texto'])
        estilo.configure('TLabelframe.Label', 
                        background=COLORES['fondo_secundario'],
                        foreground=COLORES['acento'],
                        font=('Segoe UI', 11, 'bold'))
        estilo.configure('TButton',
                        background=COLORES['acento'],
                        foreground='white',
                        font=('Segoe UI', 10, 'bold'),
                        padding=10)
        estilo.map('TButton',
                  background=[('active', COLORES['acento_secundario'])])
        estilo.configure('TEntry',
                        fieldbackground=COLORES['fondo_panel'],
                        foreground=COLORES['texto'])
        estilo.configure('TProgressbar',
                        background=COLORES['acento'],
                        troughcolor=COLORES['fondo_panel'])
        estilo.configure('Horizontal.TScale',
                        background=COLORES['fondo_principal'],
                        troughcolor=COLORES['fondo_panel'])
        
    def crear_interfaz(self):
        """Crea la interfaz gráfica completa"""
        # Frame principal
        frame_principal = tk.Frame(self.ventana, bg=COLORES['fondo_principal'])
        frame_principal.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Panel izquierdo (controles)
        panel_izq = tk.Frame(frame_principal, bg=COLORES['fondo_secundario'], 
                            width=400)
        panel_izq.pack(side='left', fill='y', padx=(0, 10))
        panel_izq.pack_propagate(False)
        
        # Panel derecho (gráficos)
        panel_der = tk.Frame(frame_principal, bg=COLORES['fondo_principal'])
        panel_der.pack(side='right', fill='both', expand=True)
        
        # === TÍTULO ===
        titulo = tk.Label(panel_izq, 
                         text="🎯 Red Neuronal\nReconstrucción de Béisbol",
                         bg=COLORES['fondo_secundario'],
                         fg=COLORES['acento'],
                         font=('Segoe UI', 16, 'bold'),
                         justify='center')
        titulo.pack(pady=15)
        
        # Botón para ver teoría
        btn_teoria = tk.Button(panel_izq, text="📚 Ver Teoría del Artículo",
                              bg=COLORES['acento_secundario'], fg='white',
                              font=('Segoe UI', 10, 'bold'),
                              activebackground=COLORES['acento'],
                              cursor='hand2',
                              command=self.mostrar_teoria)
        btn_teoria.pack(pady=(0, 10), padx=10, fill='x')
        
        # === SECCIÓN: PARÁMETROS DE SIMULACIÓN ===
        self.crear_seccion_simulacion(panel_izq)
        
        # === SECCIÓN: CONFIGURACIÓN DE RED ===
        self.crear_seccion_red(panel_izq)
        
        # === SECCIÓN: ENTRENAMIENTO ===
        self.crear_seccion_entrenamiento(panel_izq)
        
        # === SECCIÓN: PREDICCIÓN ===
        self.crear_seccion_prediccion(panel_izq)
        
        # === CONSOLA ===
        self.crear_consola(panel_izq)
        
        # === ÁREA DE GRÁFICOS ===
        self.crear_area_graficos(panel_der)
        
    def crear_seccion_simulacion(self, parent):
        """Crea la sección de parámetros de simulación"""
        frame = tk.LabelFrame(parent, text=" 📊 Parámetros de Simulación ",
                             bg=COLORES['fondo_secundario'],
                             fg=COLORES['acento'],
                             font=('Segoe UI', 11, 'bold'))
        frame.pack(fill='x', padx=10, pady=5)
        
        # Número de muestras
        row1 = tk.Frame(frame, bg=COLORES['fondo_secundario'])
        row1.pack(fill='x', padx=10, pady=5)
        
        tk.Label(row1, text="Número de muestras:", 
                bg=COLORES['fondo_secundario'], fg=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='left')
        
        self.var_muestras = tk.StringVar(value="1000")
        entry_muestras = tk.Entry(row1, textvariable=self.var_muestras,
                                 bg=COLORES['fondo_panel'], fg=COLORES['texto'],
                                 insertbackground=COLORES['texto'], width=10,
                                 font=('Segoe UI', 10))
        entry_muestras.pack(side='right')
        
        # Rango de velocidad
        row2 = tk.Frame(frame, bg=COLORES['fondo_secundario'])
        row2.pack(fill='x', padx=10, pady=5)
        
        tk.Label(row2, text="Velocidad (mph):", 
                bg=COLORES['fondo_secundario'], fg=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='left')
        
        self.var_v_max = tk.StringVar(value="100")
        self.var_v_min = tk.StringVar(value="70")
        
        tk.Entry(row2, textvariable=self.var_v_max, width=6,
                bg=COLORES['fondo_panel'], fg=COLORES['texto'],
                insertbackground=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='right')
        tk.Label(row2, text=" - ", bg=COLORES['fondo_secundario'], 
                fg=COLORES['texto']).pack(side='right')
        tk.Entry(row2, textvariable=self.var_v_min, width=6,
                bg=COLORES['fondo_panel'], fg=COLORES['texto'],
                insertbackground=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='right')
        
        # Rango de spin
        row3 = tk.Frame(frame, bg=COLORES['fondo_secundario'])
        row3.pack(fill='x', padx=10, pady=5)
        
        tk.Label(row3, text="Spin (rpm):", 
                bg=COLORES['fondo_secundario'], fg=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='left')
        
        self.var_spin_max = tk.StringVar(value="3300")
        self.var_spin_min = tk.StringVar(value="800")
        
        tk.Entry(row3, textvariable=self.var_spin_max, width=6,
                bg=COLORES['fondo_panel'], fg=COLORES['texto'],
                insertbackground=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='right')
        tk.Label(row3, text=" - ", bg=COLORES['fondo_secundario'], 
                fg=COLORES['texto']).pack(side='right')
        tk.Entry(row3, textvariable=self.var_spin_min, width=6,
                bg=COLORES['fondo_panel'], fg=COLORES['texto'],
                insertbackground=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='right')
        
        # Botón generar datos
        btn_generar = tk.Button(frame, text="⚙️ Generar Datos Sintéticos",
                               bg=COLORES['acento'], fg='white',
                               font=('Segoe UI', 10, 'bold'),
                               activebackground=COLORES['acento_secundario'],
                               cursor='hand2',
                               command=self.generar_datos)
        btn_generar.pack(pady=10, padx=10, fill='x')
        
    def crear_seccion_red(self, parent):
        """Crea la sección de configuración de red"""
        frame = tk.LabelFrame(parent, text=" 🧠 Configuración de Red ",
                             bg=COLORES['fondo_secundario'],
                             fg=COLORES['acento'],
                             font=('Segoe UI', 11, 'bold'))
        frame.pack(fill='x', padx=10, pady=5)
        
        # Capas ocultas
        row1 = tk.Frame(frame, bg=COLORES['fondo_secundario'])
        row1.pack(fill='x', padx=10, pady=5)
        
        tk.Label(row1, text="Capas ocultas:", 
                bg=COLORES['fondo_secundario'], fg=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='left')
        
        self.var_capas = tk.StringVar(value="3")
        tk.Entry(row1, textvariable=self.var_capas, width=6,
                bg=COLORES['fondo_panel'], fg=COLORES['texto'],
                insertbackground=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='right')
        
        # Neuronas por capa
        row2 = tk.Frame(frame, bg=COLORES['fondo_secundario'])
        row2.pack(fill='x', padx=10, pady=5)
        
        tk.Label(row2, text="Neuronas/capa:", 
                bg=COLORES['fondo_secundario'], fg=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='left')
        
        self.var_neuronas = tk.StringVar(value="128")
        tk.Entry(row2, textvariable=self.var_neuronas, width=6,
                bg=COLORES['fondo_panel'], fg=COLORES['texto'],
                insertbackground=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='right')
        
        # Ruido de entrenamiento
        row3 = tk.Frame(frame, bg=COLORES['fondo_secundario'])
        row3.pack(fill='x', padx=10, pady=5)
        
        tk.Label(row3, text="Ruido (mm):", 
                bg=COLORES['fondo_secundario'], fg=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='left')
        
        self.var_ruido = tk.StringVar(value="0")
        tk.Entry(row3, textvariable=self.var_ruido, width=6,
                bg=COLORES['fondo_panel'], fg=COLORES['texto'],
                insertbackground=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='right')
        
    def crear_seccion_entrenamiento(self, parent):
        """Crea la sección de entrenamiento"""
        frame = tk.LabelFrame(parent, text=" 🎓 Entrenamiento ",
                             bg=COLORES['fondo_secundario'],
                             fg=COLORES['acento'],
                             font=('Segoe UI', 11, 'bold'))
        frame.pack(fill='x', padx=10, pady=5)
        
        # Épocas
        row1 = tk.Frame(frame, bg=COLORES['fondo_secundario'])
        row1.pack(fill='x', padx=10, pady=5)
        
        tk.Label(row1, text="Épocas:", 
                bg=COLORES['fondo_secundario'], fg=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='left')
        
        self.var_epochs = tk.StringVar(value="100")
        tk.Entry(row1, textvariable=self.var_epochs, width=6,
                bg=COLORES['fondo_panel'], fg=COLORES['texto'],
                insertbackground=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='right')
        
        # Batch size
        row2 = tk.Frame(frame, bg=COLORES['fondo_secundario'])
        row2.pack(fill='x', padx=10, pady=5)
        
        tk.Label(row2, text="Batch size:", 
                bg=COLORES['fondo_secundario'], fg=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='left')
        
        self.var_batch = tk.StringVar(value="20")
        tk.Entry(row2, textvariable=self.var_batch, width=6,
                bg=COLORES['fondo_panel'], fg=COLORES['texto'],
                insertbackground=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='right')
        
        # Barra de progreso
        self.progreso = ttk.Progressbar(frame, mode='determinate', length=200)
        self.progreso.pack(pady=5, padx=10, fill='x')
        
        self.lbl_progreso = tk.Label(frame, text="Listo para entrenar",
                                    bg=COLORES['fondo_secundario'],
                                    fg=COLORES['texto_secundario'],
                                    font=('Segoe UI', 9))
        self.lbl_progreso.pack()
        
        # Botón entrenar
        btn_entrenar = tk.Button(frame, text="🚀 Entrenar Red Neuronal",
                                bg=COLORES['exito'], fg='white',
                                font=('Segoe UI', 10, 'bold'),
                                activebackground=COLORES['acento_secundario'],
                                cursor='hand2',
                                command=self.entrenar_red)
        btn_entrenar.pack(pady=10, padx=10, fill='x')
        
    def crear_seccion_prediccion(self, parent):
        """Crea la sección de predicción"""
        frame = tk.LabelFrame(parent, text=" 🎯 Predicción ",
                             bg=COLORES['fondo_secundario'],
                             fg=COLORES['acento'],
                             font=('Segoe UI', 11, 'bold'))
        frame.pack(fill='x', padx=10, pady=5)
        
        # Velocidad real para prueba
        row1 = tk.Frame(frame, bg=COLORES['fondo_secundario'])
        row1.pack(fill='x', padx=10, pady=5)
        
        tk.Label(row1, text="Velocidad test (mph):", 
                bg=COLORES['fondo_secundario'], fg=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='left')
        
        self.var_v_test = tk.StringVar(value="90")
        tk.Entry(row1, textvariable=self.var_v_test, width=6,
                bg=COLORES['fondo_panel'], fg=COLORES['texto'],
                insertbackground=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='right')
        
        # Spin real para prueba
        row2 = tk.Frame(frame, bg=COLORES['fondo_secundario'])
        row2.pack(fill='x', padx=10, pady=5)
        
        tk.Label(row2, text="Spin test (rpm):", 
                bg=COLORES['fondo_secundario'], fg=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='left')
        
        self.var_spin_test = tk.StringVar(value="1500")
        tk.Entry(row2, textvariable=self.var_spin_test, width=6,
                bg=COLORES['fondo_panel'], fg=COLORES['texto'],
                insertbackground=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='right')
        
        # Ruido de prueba
        row3 = tk.Frame(frame, bg=COLORES['fondo_secundario'])
        row3.pack(fill='x', padx=10, pady=5)
        
        tk.Label(row3, text="Ruido test (mm):", 
                bg=COLORES['fondo_secundario'], fg=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='left')
        
        self.var_ruido_test = tk.StringVar(value="2")
        tk.Entry(row3, textvariable=self.var_ruido_test, width=6,
                bg=COLORES['fondo_panel'], fg=COLORES['texto'],
                insertbackground=COLORES['texto'],
                font=('Segoe UI', 10)).pack(side='right')
        
        # Botón predecir
        btn_predecir = tk.Button(frame, text="🔮 Predecir y Visualizar",
                                bg=COLORES['advertencia'], fg='black',
                                font=('Segoe UI', 10, 'bold'),
                                activebackground=COLORES['acento_secundario'],
                                cursor='hand2',
                                command=self.predecir_y_visualizar)
        btn_predecir.pack(pady=10, padx=10, fill='x')
        
    def crear_consola(self, parent):
        """Crea el área de consola"""
        frame = tk.LabelFrame(parent, text=" 📋 Consola ",
                             bg=COLORES['fondo_secundario'],
                             fg=COLORES['acento'],
                             font=('Segoe UI', 11, 'bold'))
        frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.consola = tk.Text(frame, height=8,
                              bg=COLORES['fondo_panel'],
                              fg=COLORES['texto'],
                              font=('Consolas', 9),
                              wrap='word')
        self.consola.pack(fill='both', expand=True, padx=5, pady=5)
        
        scrollbar = tk.Scrollbar(self.consola)
        scrollbar.pack(side='right', fill='y')
        self.consola.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.consola.yview)
        
        self.log("Sistema iniciado. ¡Bienvenido!")
        self.log("Basado en: 'Robust baseball pitch reconstruction using ANNs'")
        
    def crear_area_graficos(self, parent):
        """Crea el área de gráficos"""
        # Notebook para pestañas de gráficos
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill='both', expand=True)
        
        # === PESTAÑA 0: TEORÍA ===
        tab_teoria = tk.Frame(self.notebook, bg=COLORES['fondo_principal'])
        self.notebook.add(tab_teoria, text=" 📚 Teoría del Artículo ")
        self.crear_pestana_teoria(tab_teoria)
        
        # === PESTAÑA 1: Trayectorias 3D + 2D ===
        tab1 = tk.Frame(self.notebook, bg=COLORES['fondo_principal'])
        self.notebook.add(tab1, text=" 📈 Trayectorias 3D/2D ")
        
        # Frame para controles de vista
        frame_controles = tk.Frame(tab1, bg=COLORES['fondo_secundario'])
        frame_controles.pack(fill='x', padx=5, pady=5)
        
        tk.Label(frame_controles, text="🔄 Usa el mouse para rotar la vista 3D | ",
                bg=COLORES['fondo_secundario'], fg=COLORES['texto'],
                font=('Segoe UI', 9)).pack(side='left', padx=5)
        
        tk.Label(frame_controles, text="Elevación:",
                bg=COLORES['fondo_secundario'], fg=COLORES['texto'],
                font=('Segoe UI', 9)).pack(side='left')
        
        self.var_elev = tk.IntVar(value=20)
        scale_elev = tk.Scale(frame_controles, from_=0, to=90, orient='horizontal',
                             variable=self.var_elev, length=100,
                             bg=COLORES['fondo_secundario'], fg=COLORES['texto'],
                             highlightthickness=0, troughcolor=COLORES['fondo_panel'],
                             command=lambda e: self.actualizar_vista_3d())
        scale_elev.pack(side='left', padx=5)
        
        tk.Label(frame_controles, text="Azimut:",
                bg=COLORES['fondo_secundario'], fg=COLORES['texto'],
                font=('Segoe UI', 9)).pack(side='left')
        
        self.var_azim = tk.IntVar(value=-60)
        scale_azim = tk.Scale(frame_controles, from_=-180, to=180, orient='horizontal',
                             variable=self.var_azim, length=100,
                             bg=COLORES['fondo_secundario'], fg=COLORES['texto'],
                             highlightthickness=0, troughcolor=COLORES['fondo_panel'],
                             command=lambda e: self.actualizar_vista_3d())
        scale_azim.pack(side='left', padx=5)
        
        # Botón para resetear vista
        btn_reset = tk.Button(frame_controles, text="↺ Reset Vista",
                             bg=COLORES['acento'], fg='white',
                             font=('Segoe UI', 9),
                             command=self.resetear_vista_3d)
        btn_reset.pack(side='left', padx=10)
        
        # Figura con 2 subplots: 3D y 2D
        self.fig1 = Figure(figsize=(12, 5), facecolor=COLORES['grafico_fondo'])
        
        # Subplot 3D (izquierda)
        self.ax_tray = self.fig1.add_subplot(121, projection='3d')
        self.configurar_grafico_3d(self.ax_tray, "Vista 3D - Trayectorias")
        
        # Subplot 2D (derecha)
        self.ax_tray_2d = self.fig1.add_subplot(122)
        self.configurar_grafico_2d(self.ax_tray_2d, "Vista 2D - Deflexiones Y/Z",
                                   "Posición X (m)", "Deflexión (m)")
        
        self.fig1.tight_layout()
        
        self.canvas1 = FigureCanvasTkAgg(self.fig1, tab1)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(fill='both', expand=True)
        
        toolbar1 = NavigationToolbar2Tk(self.canvas1, tab1)
        toolbar1.update()
        
        # === PESTAÑA 2: Entrenamiento ===
        tab2 = tk.Frame(self.notebook, bg=COLORES['fondo_principal'])
        self.notebook.add(tab2, text=" 📊 Pérdida de Entrenamiento ")
        
        self.fig2 = Figure(figsize=(10, 6), facecolor=COLORES['grafico_fondo'])
        self.ax_loss = self.fig2.add_subplot(111)
        self.configurar_grafico_2d(self.ax_loss, "Pérdida Durante Entrenamiento",
                                   "Época", "MSE Loss")
        
        self.canvas2 = FigureCanvasTkAgg(self.fig2, tab2)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(fill='both', expand=True)
        
        toolbar2 = NavigationToolbar2Tk(self.canvas2, tab2)
        toolbar2.update()
        
        # === PESTAÑA 3: Comparación ===
        tab3 = tk.Frame(self.notebook, bg=COLORES['fondo_principal'])
        self.notebook.add(tab3, text=" 🎯 Comparación Real vs Predicción ")
        
        self.fig3 = Figure(figsize=(10, 6), facecolor=COLORES['grafico_fondo'])
        self.ax_comp = self.fig3.add_subplot(111)
        self.configurar_grafico_2d(self.ax_comp, "Comparación de Trayectorias",
                                   "Posición X (m)", "Posición Y/Z (m)")
        
        self.canvas3 = FigureCanvasTkAgg(self.fig3, tab3)
        self.canvas3.draw()
        self.canvas3.get_tk_widget().pack(fill='both', expand=True)
        
        toolbar3 = NavigationToolbar2Tk(self.canvas3, tab3)
        toolbar3.update()
        
        # === PESTAÑA 4: Análisis de Robustez ===
        tab4 = tk.Frame(self.notebook, bg=COLORES['fondo_principal'])
        self.notebook.add(tab4, text=" 📉 Análisis de Errores ")
        
        self.fig4 = Figure(figsize=(10, 6), facecolor=COLORES['grafico_fondo'])
        self.ax_err = self.fig4.add_subplot(111)
        self.configurar_grafico_2d(self.ax_err, "Error vs Ruido de Entrada",
                                   "Ruido Inyectado (mm)", "Error Euclidiano (mm)")
        
        self.canvas4 = FigureCanvasTkAgg(self.fig4, tab4)
        self.canvas4.draw()
        self.canvas4.get_tk_widget().pack(fill='both', expand=True)
        
        toolbar4 = NavigationToolbar2Tk(self.canvas4, tab4)
        toolbar4.update()
    
    def crear_pestana_teoria(self, parent):
        """Crea la pestaña con la teoría del artículo"""
        # Frame principal con scroll
        canvas_scroll = tk.Canvas(parent, bg=COLORES['fondo_principal'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient='vertical', command=canvas_scroll.yview)
        frame_contenido = tk.Frame(canvas_scroll, bg=COLORES['fondo_principal'])
        
        frame_contenido.bind('<Configure>', 
                             lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox('all')))
        
        canvas_scroll.create_window((0, 0), window=frame_contenido, anchor='nw')
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        
        # Habilitar scroll con rueda del mouse
        def scroll_mouse(event):
            canvas_scroll.yview_scroll(int(-1*(event.delta/120)), 'units')
        canvas_scroll.bind_all('<MouseWheel>', scroll_mouse)
        
        scrollbar.pack(side='right', fill='y')
        canvas_scroll.pack(side='left', fill='both', expand=True)
        
        # === CONTENIDO DE LA TEORÍA ===
        
        # Título principal
        titulo = tk.Label(frame_contenido, 
                         text="📖 TEORÍA: Reconstrucción de Trayectorias de Béisbol\ncon Redes Neuronales Artificiales",
                         bg=COLORES['fondo_principal'], fg=COLORES['acento'],
                         font=('Segoe UI', 18, 'bold'), justify='center')
        titulo.pack(pady=20, padx=20)
        
        # Sección 1: Introducción
        self.agregar_seccion_teoria(frame_contenido, "1. INTRODUCCIÓN",
            """El seguimiento de lanzamientos en tiempo real proporciona análisis valioso de 
velocidad, movimiento y rotación para la toma de decisiones en béisbol.

• Los sistemas de alta gama (como Hawk-Eye) son muy costosos
• Esta aplicación implementa una alternativa de BAJO COSTO usando RNA
• Utiliza el método de reconstrucción de TRES PUNTOS

La idea es democratizar el acceso a análisis cuantitativos en todos los niveles 
del béisbol (amateur, universitario, internacional).""")
        
        # Sección 2: Física del Béisbol
        self.agregar_seccion_teoria(frame_contenido, "2. FÍSICA DE LA TRAYECTORIA",
            """La trayectoria del béisbol está gobernada por tres fuerzas principales:

🔹 FUERZA DE ARRASTRE (Drag):
   fD = -(0.0039 + 0.0058/(1 + e^((v-vd)/Δ))) · v · v
   
   Donde vd = 35 m/s y Δ = 5 m/s

🔹 FUERZA DE MAGNUS:
   fM = B · (ω × v)
   
   Donde B = 0.00041 (factor de escala para béisbol)
   Esta fuerza causa el característico "movimiento" de los lanzamientos

🔹 GRAVEDAD:
   fG = -9.81 m/s² (dirección z)

⚡ ECUACIÓN DE MOVIMIENTO:
   a = fD + fM + fG
   
Se integra usando Runge-Kutta de 4to orden con Δt = 2.083 ms""")
        
        # Sección 3: Método de Tres Puntos
        self.agregar_seccion_teoria(frame_contenido, "3. MÉTODO DE TRES PUNTOS",
            """El algoritmo utiliza SOLO 3 posiciones espaciotemporales del béisbol:

   📍 x(t₁), x(t₂), x(t₃)  a 60 fps

Estos tres puntos representan la INFORMACIÓN MÍNIMA necesaria para 
determinar la trayectoria completa del lanzamiento.

✅ Ventajas:
   • Compatible con cámaras de teléfono (60 fps estándar)
   • Bajo requisito de memoria
   • Adecuado para sistemas de bajo costo

❌ El ruido en la detección de posición afecta la precisión:
   • Cámaras de alta fidelidad: ~2 mm de error
   • Cámaras de baja fidelidad: ~20-50 mm de error""")
        
        # Sección 4: Arquitectura RNA
        self.agregar_seccion_teoria(frame_contenido, "4. ARQUITECTURA DE LA RED NEURONAL",
            """La RNA tiene la siguiente estructura:

┌─────────────────────────────────────────────────────────┐
│  ENTRADA (9 neuronas)                                   │
│  [x₁, y₁, z₁, x₂, y₂, z₂, x₃, y₃, z₃]                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  CAPA OCULTA 1: 128 neuronas + ReLU                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  CAPA OCULTA 2: 128 neuronas + ReLU                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  CAPA OCULTA 3: 128 neuronas + ReLU                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  SALIDA (2 neuronas) + Sigmoid                         │
│  [velocidad_inicial, tasa_de_rotación]                 │
└─────────────────────────────────────────────────────────┘

🔧 Hiperparámetros:
   • Optimizador: ADAM
   • Función de pérdida: MSE (Error Cuadrático Medio)
   • Batch size: 20
   • Épocas: 100""")
        
        # Sección 5: Entrenamiento con Ruido
        self.agregar_seccion_teoria(frame_contenido, "5. ROBUSTEZ: ENTRENAMIENTO CON RUIDO",
            """Se entrenan DOS modelos de RNA:

🔇 MODELO 'SILENCIOSO' (Quiet):
   • Entrenado con datos LIMPIOS (sin ruido)
   • Mayor precisión con datos de alta fidelidad
   • Error aumenta rápidamente con ruido de entrada

📢 MODELO 'RUIDOSO' (Noisy):
   • Entrenado con ruido uniforme de ±25 mm
   • Menor precisión inicial, pero MAYOR ROBUSTEZ
   • Tolera mejor el ruido de cámaras de baja calidad

📊 HALLAZGOS CLAVE:
   • A partir de 26 mm de ruido, el modelo 'Ruidoso' SUPERA 
     a los métodos iterativos tradicionales
   • La pendiente de error del modelo 'Ruidoso' es ~3x menor
   • El modelo 'Ruidoso' es ideal para aplicaciones del mundo real""")
        
        # Sección 6: Métricas de Error
        self.agregar_seccion_teoria(frame_contenido, "6. MÉTRICAS DE EVALUACIÓN",
            """Se utilizan dos métricas principales:

📏 ERROR RELATIVO:
   ε = |x̂ - xT|₂ / xT²
   
   Donde x̂ es la posición predicha y xT la verdadera

📐 ERROR EUCLIDIANO (distancia L2):
   δ = ||x̂ - xT||_L2
   
   Medido en el punto MEDIO de la trayectoria (consistente 
   con la literatura)

📈 ANÁLISIS DE REGRESIÓN:
   Error = â · Ruido + b̂
   
   Modelo Silencioso: â ≈ 10.52, b̂ ≈ 31.98
   Modelo Ruidoso:    â ≈ 1.13,  b̂ ≈ 53.58
   
   ¡El modelo Ruidoso tiene 10x menor sensibilidad al ruido!""")
        
        # Sección 7: Conclusiones
        self.agregar_seccion_teoria(frame_contenido, "7. CONCLUSIONES",
            """✅ Las RNA pueden reconstruir trayectorias con ALTA PRECISIÓN

✅ La inyección de ruido en entrenamiento AUMENTA LA ROBUSTEZ

✅ El modelo supera métodos iterativos tradicionales con ruido > 26 mm

✅ Viable para sistemas de BAJO COSTO (teléfonos, cámaras económicas)

🚀 TRABAJO FUTURO:
   • Predecir eje de rotación y ángulo de lanzamiento
   • Integración con sistemas de visión por computadora
   • Extensión a otros deportes (cricket, tenis, golf)

═══════════════════════════════════════════════════════════
Basado en: "Robust baseball pitch reconstruction using 
artificial neural networks with noisy data"
DeBoskey, R.D., Hasti, V.R., Narayanaswamy, V.
J. Quant. Anal. Sports, 2026
═══════════════════════════════════════════════════════════""")
    
    def agregar_seccion_teoria(self, parent, titulo, contenido):
        """Agrega una sección de teoría con formato"""
        frame = tk.Frame(parent, bg=COLORES['fondo_secundario'], 
                        highlightbackground=COLORES['acento'],
                        highlightthickness=2)
        frame.pack(fill='x', padx=20, pady=10)
        
        # Título de sección
        lbl_titulo = tk.Label(frame, text=titulo,
                             bg=COLORES['fondo_secundario'],
                             fg=COLORES['exito'],
                             font=('Segoe UI', 14, 'bold'),
                             anchor='w')
        lbl_titulo.pack(fill='x', padx=15, pady=(10, 5))
        
        # Contenido
        lbl_contenido = tk.Label(frame, text=contenido,
                                bg=COLORES['fondo_secundario'],
                                fg=COLORES['texto'],
                                font=('Consolas', 10),
                                justify='left',
                                anchor='w')
        lbl_contenido.pack(fill='x', padx=15, pady=(5, 15))
    
    def actualizar_vista_3d(self):
        """Actualiza la vista 3D según los controles"""
        if hasattr(self, 'ax_tray'):
            self.ax_tray.view_init(elev=self.var_elev.get(), azim=self.var_azim.get())
            self.canvas1.draw()
    
    def resetear_vista_3d(self):
        """Resetea la vista 3D a valores por defecto"""
        self.var_elev.set(20)
        self.var_azim.set(-60)
        self.actualizar_vista_3d()
    
    def mostrar_teoria(self):
        """Muestra la pestaña de teoría"""
        self.notebook.select(0)  # La pestaña de teoría es la primera
        self.log("📚 Mostrando teoría del artículo")
        
    def configurar_grafico_3d(self, ax, titulo):
        """Configura un gráfico 3D"""
        ax.set_facecolor(COLORES['grafico_fondo'])
        ax.set_title(titulo, color=COLORES['texto'], fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)', color=COLORES['texto'])
        ax.set_ylabel('Y (m)', color=COLORES['texto'])
        ax.set_zlabel('Z (m)', color=COLORES['texto'])
        ax.tick_params(colors=COLORES['texto_secundario'])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
    def configurar_grafico_2d(self, ax, titulo, xlabel, ylabel):
        """Configura un gráfico 2D"""
        ax.set_facecolor(COLORES['grafico_fondo'])
        ax.set_title(titulo, color=COLORES['texto'], fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, color=COLORES['texto'])
        ax.set_ylabel(ylabel, color=COLORES['texto'])
        ax.tick_params(colors=COLORES['texto_secundario'])
        ax.spines['bottom'].set_color(COLORES['texto_secundario'])
        ax.spines['top'].set_color(COLORES['texto_secundario'])
        ax.spines['left'].set_color(COLORES['texto_secundario'])
        ax.spines['right'].set_color(COLORES['texto_secundario'])
        ax.grid(True, alpha=0.3, color=COLORES['texto_secundario'])
        
    def log(self, mensaje):
        """Añade un mensaje a la consola"""
        self.consola.insert('end', f"➤ {mensaje}\n")
        self.consola.see('end')
        self.ventana.update_idletasks()
        
    def generar_datos(self):
        """Genera datos sintéticos de entrenamiento"""
        try:
            n_muestras = int(self.var_muestras.get())
            v_min = float(self.var_v_min.get())
            v_max = float(self.var_v_max.get())
            spin_min = float(self.var_spin_min.get())
            spin_max = float(self.var_spin_max.get())
            
            self.log(f"Generando {n_muestras} muestras de entrenamiento...")
            self.progreso['value'] = 0
            
            def callback(actual, total):
                progreso = (actual / total) * 100
                self.progreso['value'] = progreso
                self.lbl_progreso.config(text=f"Generando: {actual}/{total}")
                self.ventana.update_idletasks()
            
            self.datos_X, self.datos_y = self.simulador.generar_datos_entrenamiento(
                n_muestras, v_min, v_max, spin_min, spin_max, callback
            )
            
            self.progreso['value'] = 100
            self.lbl_progreso.config(text="Datos generados ✓")
            self.log(f"✓ {n_muestras} trayectorias sintéticas generadas")
            self.log(f"  Velocidad: {v_min}-{v_max} mph, Spin: {spin_min}-{spin_max} rpm")
            
            # Graficar algunas trayectorias de ejemplo
            self.graficar_trayectorias_ejemplo()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar datos: {str(e)}")
            self.log(f"✗ Error: {str(e)}")
            
    def graficar_trayectorias_ejemplo(self):
        """Grafica algunas trayectorias de ejemplo en 3D y 2D"""
        # Limpiar ambos ejes
        self.ax_tray.clear()
        self.ax_tray_2d.clear()
        
        self.configurar_grafico_3d(self.ax_tray, "Vista 3D - Trayectorias")
        self.configurar_grafico_2d(self.ax_tray_2d, "Vista 2D - Deflexiones",
                                   "Posición X (m)", "Deflexión (m)")
        
        # Generar y graficar 5 trayectorias de ejemplo
        colores = [COLORES['grafico_linea1'], COLORES['grafico_linea2'], 
                   COLORES['grafico_linea3'], '#9b59b6', '#3498db']
        
        # Guardar trayectorias para referencia
        self.trayectorias_ejemplo = []
        
        for i in range(5):
            v0 = np.random.uniform(70, 100)
            spin = np.random.uniform(800, 3300)
            
            t, tray = self.simulador.simular_lanzamiento(v0, spin)
            self.trayectorias_ejemplo.append({'v0': v0, 'spin': spin, 'tray': tray})
            
            # Gráfico 3D
            self.ax_tray.plot3D(tray[:, 0], tray[:, 1], tray[:, 2],
                               color=colores[i], linewidth=2,
                               label=f'v={v0:.0f}mph, ω={spin:.0f}rpm')
            
            # Gráfico 2D - Deflexión Y (línea sólida)
            self.ax_tray_2d.plot(tray[:, 0], tray[:, 1],
                                color=colores[i], linewidth=2, linestyle='-',
                                label=f'Y: v={v0:.0f}, ω={spin:.0f}')
            
            # Gráfico 2D - Deflexión Z (línea punteada)
            self.ax_tray_2d.plot(tray[:, 0], tray[:, 2],
                                color=colores[i], linewidth=2, linestyle='--',
                                alpha=0.6)
        
        # Leyendas
        self.ax_tray.legend(loc='upper left', fontsize=7,
                           facecolor=COLORES['fondo_panel'],
                           edgecolor=COLORES['texto_secundario'],
                           labelcolor=COLORES['texto'])
        
        # Añadir nota en gráfico 2D
        self.ax_tray_2d.text(0.02, 0.98, 'Sólida: Y | Punteada: Z',
                            transform=self.ax_tray_2d.transAxes,
                            fontsize=8, color=COLORES['texto'],
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor=COLORES['fondo_panel'],
                                     edgecolor=COLORES['texto_secundario'], alpha=0.8))
        
        self.fig1.tight_layout()
        self.canvas1.draw()
        self.notebook.select(1)  # Seleccionar pestaña de trayectorias (ahora es índice 1)
        
    def entrenar_red(self):
        """Entrena la red neuronal"""
        if self.datos_X is None:
            messagebox.showwarning("Advertencia", 
                                  "Primero debe generar datos de entrenamiento")
            return
        
        try:
            capas = int(self.var_capas.get())
            neuronas = int(self.var_neuronas.get())
            epochs = int(self.var_epochs.get())
            batch_size = int(self.var_batch.get())
            ruido = float(self.var_ruido.get())
            
            self.log(f"Construyendo red: {capas} capas, {neuronas} neuronas")
            self.log(f"Entrenando con {epochs} épocas, batch={batch_size}, ruido={ruido}mm")
            
            # Crear red neuronal
            self.red_neuronal = RedNeuronal()
            self.red_neuronal.construir_modelo(capas, neuronas)
            
            self.progreso['value'] = 0
            
            def callback(epoch, total, loss, val_loss):
                progreso = (epoch / total) * 100
                self.progreso['value'] = progreso
                self.lbl_progreso.config(
                    text=f"Época {epoch}/{total} - Loss: {loss:.6f}, Val: {val_loss:.6f}"
                )
                self.ventana.update_idletasks()
            
            # Entrenar
            self.historial_entrenamiento = self.red_neuronal.entrenar(
                self.datos_X, self.datos_y,
                epochs=epochs,
                batch_size=batch_size,
                ruido=ruido,
                callback=callback
            )
            
            self.progreso['value'] = 100
            self.lbl_progreso.config(text="Entrenamiento completado ✓")
            
            final_loss = self.historial_entrenamiento['loss'][-1]
            final_val = self.historial_entrenamiento['val_loss'][-1]
            self.log(f"✓ Entrenamiento completado")
            self.log(f"  Loss final: {final_loss:.6f}, Val loss: {final_val:.6f}")
            
            # Graficar pérdida
            self.graficar_perdida()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en entrenamiento: {str(e)}")
            self.log(f"✗ Error: {str(e)}")
            
    def graficar_perdida(self):
        """Grafica la pérdida durante el entrenamiento"""
        self.ax_loss.clear()
        self.configurar_grafico_2d(self.ax_loss, "Pérdida Durante Entrenamiento",
                                   "Época", "MSE Loss (log)")
        
        epochs = range(1, len(self.historial_entrenamiento['loss']) + 1)
        
        self.ax_loss.semilogy(epochs, self.historial_entrenamiento['loss'],
                             color=COLORES['grafico_linea1'], linewidth=2,
                             label='Entrenamiento')
        self.ax_loss.semilogy(epochs, self.historial_entrenamiento['val_loss'],
                             color=COLORES['grafico_linea2'], linewidth=2,
                             label='Validación')
        
        self.ax_loss.legend(loc='upper right', fontsize=10,
                           facecolor=COLORES['fondo_panel'],
                           edgecolor=COLORES['texto_secundario'],
                           labelcolor=COLORES['texto'])
        
        self.canvas2.draw()
        self.notebook.select(1)
        
    def predecir_y_visualizar(self):
        """Realiza predicción y visualiza resultados"""
        if self.red_neuronal is None:
            messagebox.showwarning("Advertencia", 
                                  "Primero debe entrenar la red neuronal")
            return
        
        try:
            v_test = float(self.var_v_test.get())
            spin_test = float(self.var_spin_test.get())
            ruido_test = float(self.var_ruido_test.get())
            
            self.log(f"Probando con v={v_test}mph, spin={spin_test}rpm, ruido={ruido_test}mm")
            
            # Simular trayectoria real
            t_real, tray_real = self.simulador.simular_lanzamiento(v_test, spin_test)
            tres_puntos = self.simulador.obtener_tres_puntos(tray_real, t_real)
            
            # Añadir ruido
            ruido_m = ruido_test / 1000  # convertir a metros
            tres_puntos_ruido = tres_puntos + np.random.uniform(-ruido_m, ruido_m, 
                                                                  tres_puntos.shape)
            
            # Predecir
            X_test = tres_puntos_ruido.flatten().reshape(1, -1)
            prediccion = self.red_neuronal.predecir(X_test)[0]
            
            v_pred, spin_pred = prediccion
            
            # Reconstruir trayectoria con valores predichos
            t_pred, tray_pred = self.simulador.simular_lanzamiento(v_pred, spin_pred)
            
            # Calcular error euclidiano en punto medio
            idx_mid_real = len(tray_real) // 2
            idx_mid_pred = len(tray_pred) // 2
            
            error_euclidiano = np.linalg.norm(tray_real[idx_mid_real] - 
                                              tray_pred[idx_mid_pred]) * 1000  # mm
            
            self.log(f"✓ Predicción completada:")
            self.log(f"  Real:      v={v_test:.1f} mph, spin={spin_test:.0f} rpm")
            self.log(f"  Predicho:  v={v_pred:.1f} mph, spin={spin_pred:.0f} rpm")
            self.log(f"  Error:     Δv={abs(v_test-v_pred):.2f} mph, "
                    f"Δspin={abs(spin_test-spin_pred):.0f} rpm")
            self.log(f"  Error euclidiano: {error_euclidiano:.2f} mm")
            
            # Graficar comparación
            self.graficar_comparacion(tray_real, tray_pred, v_test, spin_test, 
                                      v_pred, spin_pred, error_euclidiano)
            
            # Realizar análisis de robustez
            self.analizar_robustez(v_test, spin_test)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en predicción: {str(e)}")
            self.log(f"✗ Error: {str(e)}")
            
    def graficar_comparacion(self, tray_real, tray_pred, v_real, spin_real,
                            v_pred, spin_pred, error):
        """Grafica la comparación entre trayectoria real y predicha"""
        self.ax_comp.clear()
        self.configurar_grafico_2d(self.ax_comp, 
                                   f"Comparación: Error Euclidiano = {error:.2f} mm",
                                   "Posición X (m)", "Deflexión (m)")
        
        # Graficar deflexión Y
        self.ax_comp.plot(tray_real[:, 0], tray_real[:, 1],
                         color=COLORES['grafico_linea1'], linewidth=2,
                         linestyle='-', label=f'Real Y (v={v_real:.0f}, ω={spin_real:.0f})')
        self.ax_comp.plot(tray_pred[:, 0], tray_pred[:, 1],
                         color=COLORES['grafico_linea2'], linewidth=2,
                         linestyle='--', label=f'Pred Y (v={v_pred:.0f}, ω={spin_pred:.0f})')
        
        # Graficar deflexión Z
        self.ax_comp.plot(tray_real[:, 0], tray_real[:, 2],
                         color=COLORES['grafico_linea1'], linewidth=2,
                         linestyle=':', alpha=0.7, label='Real Z')
        self.ax_comp.plot(tray_pred[:, 0], tray_pred[:, 2],
                         color=COLORES['grafico_linea2'], linewidth=2,
                         linestyle='-.', alpha=0.7, label='Pred Z')
        
        self.ax_comp.legend(loc='best', fontsize=9,
                           facecolor=COLORES['fondo_panel'],
                           edgecolor=COLORES['texto_secundario'],
                           labelcolor=COLORES['texto'])
        
        self.canvas3.draw()
        self.notebook.select(2)
        
    def analizar_robustez(self, v_test, spin_test):
        """Realiza análisis de robustez con diferentes niveles de ruido"""
        self.log("Realizando análisis de robustez...")
        
        niveles_ruido = np.arange(0, 52, 2)  # 0 a 50 mm en incrementos de 2
        errores_medios = []
        errores_max = []
        
        n_muestras = 20  # Muestras por nivel de ruido
        
        for ruido in niveles_ruido:
            errores = []
            ruido_m = ruido / 1000
            
            for _ in range(n_muestras):
                # Simular trayectoria
                t_real, tray_real = self.simulador.simular_lanzamiento(v_test, spin_test)
                tres_puntos = self.simulador.obtener_tres_puntos(tray_real, t_real)
                
                # Añadir ruido
                tres_puntos_ruido = tres_puntos + np.random.uniform(-ruido_m, ruido_m,
                                                                     tres_puntos.shape)
                
                # Predecir
                X_test = tres_puntos_ruido.flatten().reshape(1, -1)
                prediccion = self.red_neuronal.predecir(X_test)[0]
                
                # Reconstruir
                t_pred, tray_pred = self.simulador.simular_lanzamiento(
                    prediccion[0], prediccion[1]
                )
                
                # Calcular error en punto medio
                idx_mid = len(tray_real) // 2
                error = np.linalg.norm(tray_real[idx_mid] - tray_pred[idx_mid]) * 1000
                errores.append(error)
            
            errores_medios.append(np.mean(errores))
            errores_max.append(np.max(errores))
        
        # Graficar
        self.ax_err.clear()
        self.configurar_grafico_2d(self.ax_err, "Análisis de Robustez",
                                   "Ruido Inyectado (mm)", "Error Euclidiano (mm)")
        
        self.ax_err.plot(niveles_ruido, errores_medios,
                        color=COLORES['grafico_linea1'], linewidth=2,
                        marker='o', markersize=4, label='Error Medio')
        self.ax_err.plot(niveles_ruido, errores_max,
                        color=COLORES['grafico_linea2'], linewidth=2,
                        marker='s', markersize=4, label='Error Máximo')
        
        # Regresión lineal
        coef_medio = np.polyfit(niveles_ruido, errores_medios, 1)
        coef_max = np.polyfit(niveles_ruido, errores_max, 1)
        
        self.ax_err.plot(niveles_ruido, np.polyval(coef_medio, niveles_ruido),
                        color=COLORES['grafico_linea1'], linewidth=1,
                        linestyle='--', alpha=0.5,
                        label=f'Ajuste: y = {coef_medio[0]:.2f}x + {coef_medio[1]:.2f}')
        
        self.ax_err.legend(loc='upper left', fontsize=9,
                          facecolor=COLORES['fondo_panel'],
                          edgecolor=COLORES['texto_secundario'],
                          labelcolor=COLORES['texto'])
        
        self.canvas4.draw()
        self.notebook.select(3)
        
        self.log(f"✓ Análisis de robustez completado")
        self.log(f"  Pendiente error medio: {coef_medio[0]:.3f} mm/mm")
        self.log(f"  Pendiente error máx:   {coef_max[0]:.3f} mm/mm")
        
    def ejecutar(self):
        """Ejecuta la aplicación"""
        self.ventana.mainloop()


if __name__ == "__main__":
    app = AplicacionPrincipal()
    app.ejecutar()
