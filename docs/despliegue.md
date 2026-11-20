# Despliegue

| Parte | Plataforma | Por qué |
|---|---|---|
| Frontend (`frontend/`) | Vercel | Es HTML, CSS y JavaScript estático, sin compilación. |
| API (`src/api/`) | Render (Docker, plan gratis) | CatBoost, pandas y scikit-learn (~350 MB) más el modelo (74 MB) superan el límite de 250 MB de las funciones de Vercel. Render mantiene el proceso vivo, así que el historial de predicciones del dashboard se conserva mientras el servicio esté despierto. La API usa ~320 MB de RAM, dentro de los 512 MB del plan gratis. |

## 1. API en Render

La API se despliega desde un repositorio aparte que contiene solo lo necesario para ejecutarla (código de `src/`, modelos entrenados y manifiesto del dataset). Se genera con:

```bash
python -m src.ml.train            # si no existen los modelos en models/
bash deploy/api/build_deploy.sh   # arma build/api-deploy/ (repositorio git propio)
```

Primera vez:

1. Crear en GitHub un repositorio vacío (sin README), por ejemplo `ctr-api`.
2. Subir la carpeta:
   ```bash
   cd build/api-deploy
   git remote add origin https://github.com/<usuario>/ctr-api.git
   git push -u origin main
   ```
3. En render.com: **New → Blueprint**, elegir el repositorio y aceptar. Render lee `render.yaml` (servicio web Docker, plan *Free*, chequeo en `/health`).

La API queda en `https://ctr-api-xxxx.onrender.com` (documentación en `/docs`).

Para actualizarla: volver a ejecutar `build_deploy.sh`, y en `build/api-deploy` hacer `git add -A && git commit -m "..." && git push`. Render redespliega solo.

**Plan gratis:** el servicio se duerme tras 15 minutos sin peticiones y la primera tarda cerca de un minuto en responder. Antes de una presentación conviene abrir `/health` unos minutos antes.

Archivos: `deploy/api/Dockerfile`, `deploy/api/render.yaml` y `requirements-api.txt` (versiones fijadas: `model.joblib` exige la misma versión de scikit-learn con la que se entrenó).

## 2. Frontend en Vercel

1. Poner la URL de la API en `frontend/index.html`:
   ```html
   <meta name="ctr-api-base" content="https://ctr-api-xxxx.onrender.com">
   ```
   En `localhost` la interfaz sigue usando `http://127.0.0.1:8000`.
2. Publicar la carpeta:
   ```bash
   cd frontend
   npx vercel --prod   # la primera vez pide iniciar sesión y crear el proyecto
   ```
   Si se conecta el repositorio desde vercel.com, configurar **Root Directory = `frontend`** y *Framework Preset* = **Other**, sin comando de build.

La API acepta peticiones de cualquier origen (`CTR_CORS_ORIGINS="*"`). Para restringirla, cambiar esa variable en Render por el dominio de Vercel.
