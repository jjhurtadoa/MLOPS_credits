# Docker

> Containerización de servicios con Docker Compose

## 🎯 Arquitectura

```
docker-compose.yml
├── api (FastAPI)         : 8000
├── prometheus (Monitoring) :9090
└── grafana (Dashboards)   :3000
```

---

## 🚀 Comandos Esenciales

### **Levantar todos los servicios**
```bash
docker compose up -d
```

### **Ver logs**
```bash
# Todos los servicios
docker compose logs -f

# Solo API
docker compose logs -f api
```

### **Ver estado**
```bash
docker compose ps
```

### **Reiniciar servicio**
```bash
docker compose restart api
```

### **Rebuild (después de cambios en código)**
```bash
docker compose build api
docker compose up -d api
```

### **Apagar todo**
```bash
docker compose down
```

### **Limpiar volúmenes (reset completo)**
```bash
docker compose down -v
```

---

## 📂 Estructura de servicios

### **Service:  `api`**

**Archivo:** `Dockerfile`

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY api/ ./api/
COPY artifacts/ ./artifacts/
CMD ["gunicorn", "api.main:app", "-w", "2", "-k", "uvicorn.workers. UvicornWorker", "--bind", "0.0.0.0:8000"]
```

**Puertos:** `8000:8000`  
**Health check:** `curl http://localhost:8000/health`

---

### **Service: `prometheus`**

**Imagen:** `prom/prometheus:latest`  
**Config:** `prometheus/prometheus.yml`  
**Puertos:** `9090:9090`  
**Volúmenes:** 
- `./prometheus:/etc/prometheus`
- `prometheus_data:/prometheus`

---

### **Service: `grafana`**

**Imagen:** `grafana/grafana:latest`  
**Puertos:** `3000:3000`  
**Credenciales:** `admin/admin`  
**Volúmenes:**
- `grafana_data:/var/lib/grafana`
- `./grafana/provisioning:/etc/grafana/provisioning`

---

## 🔧 Configuración de red

```yaml
networks:
  mlops:
    driver: bridge
```

**Comunicación entre contenedores:**
- API → expone métricas en `api: 8000/metrics`
- Prometheus → scrape de `api:8000`
- Grafana → lee de `prometheus:9090`

---

## ⚠️ Troubleshooting

### **API no inicia**
```bash
# Ver logs detallados
docker compose logs api

# Causas comunes:
# - Modelo no encontrado (Git LFS)
# - Dependencias faltantes
# - Puerto 8000 ocupado
```

### **Prometheus no encuentra la API**
```bash
# Verificar red
docker network inspect mlops-credits_mlops

# Verificar targets en Prometheus UI
# http://localhost:9090/targets
```

### **Cambios en código no se reflejan**
```bash
# Rebuild forzado
docker compose build --no-cache api
docker compose up -d api
```

---

## 🔗 Referencias

- [Docker Compose Docs](https://docs.docker.com/compose/)
- [Multi-stage builds](https://docs.docker.com/develop/develop-images/multistage-build/)

---

[← Volver a Grafana](grafana.md) | [Índice](../index.md)