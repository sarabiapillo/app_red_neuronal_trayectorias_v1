# Red Neuronal para Reconstrucción de Lanzamientos de Béisbol ⚾🧠

Aplicación que implementa un método basado en redes neuronales artificiales para reconstruir parámetros de lanzamientos de béisbol (velocidad inicial y tasa de rotación) a partir de tres puntos de trayectoria.

## 📋 Requisitos

- Python 3.8 o superior
- Windows / Linux / macOS

## 🚀 Instalación Rápida

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/RedNeuronalBeisbol.git
cd RedNeuronalBeisbol
```

### 2. Instalar dependencias
```bash
pip install numpy matplotlib tensorflow
```

> **Nota:** TensorFlow es opcional. Si no está instalado, la app usará NumPy automáticamente.

### 3. Ejecutar la aplicación
```bash
python app_red_neuronal.py
```

## 📦 Dependencias

| Paquete | Versión | Obligatorio |
|---------|---------|-------------|
| numpy | >= 1.20 | ✅ Sí |
| matplotlib | >= 3.5 | ✅ Sí |
| tensorflow | >= 2.10 | ❌ Opcional |

## 🎯 Características

- ✅ Simulación física con modelo de arrastre de Giordano
- ✅ Red neuronal con arquitectura 9 → 128 → 128 → 128 → 2
- ✅ Visualización 3D y 2D de trayectorias
- ✅ Controles de rotación para vista 3D
- ✅ Análisis de robustez ante ruido en mediciones
- ✅ Documentación teórica integrada

## 🖥️ Uso

1. **Ver Trayectoria Ejemplo**: Simula y visualiza una trayectoria con los parámetros actuales
2. **Entrenar Red Neuronal**: Genera datos sintéticos y entrena el modelo
3. **Predecir y Visualizar**: Compara trayectoria real vs predicción
4. **Análisis de Robustez**: Evalúa el rendimiento con diferentes niveles de ruido
5. **Ver Teoría**: Accede a la documentación teórica del artículo

## 📸 Interfaz

La aplicación incluye:
- Panel de control con parámetros ajustables
- Vista 3D con controles de elevación y azimut
- Vista 2D lateral
- Pestaña de entrenamiento con gráfico de pérdida
- Pestaña de comparación real vs predicción
- Pestaña de análisis de robustez
- Pestaña de teoría del artículo

## 📚 Basado en

- **Artículo**: "Three-Point Pitch Reconstruction Using Artificial Neural Networks"
- **Física**: Modelo de arrastre de Giordano (2013)
- **Integrador**: Runge-Kutta 4° orden

## ⚠️ Solución de Problemas

### Error de Tkinter / Tcl
Si estás usando un entorno virtual y aparece error de Tcl:
```bash
# Opción 1: Desactivar el venv
deactivate
python app_red_neuronal.py

# Opción 2: Usar Python del sistema directamente
C:/Users/TU_USUARIO/AppData/Local/Programs/Python/Python313/python.exe app_red_neuronal.py
```

### Dependencias faltantes
```bash
pip install numpy matplotlib
```

## 📄 Licencia

MIT License - Uso libre para fines educativos e investigación.
