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