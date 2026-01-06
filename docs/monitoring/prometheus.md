# Prometheus - Monitoreo de Métricas

## 🎯 Propósito

Prometheus recolecta métricas de la API cada 15 segundos y las almacena en una time-series database.

---

## 🚀 Acceso

- **URL:** http://localhost:9090
- **Puerto:** 9090

---

## ✅ Verificar Targets

![Targets](../../images/prometheus-01-targets.png)

1. Click en **Status** → **Targets**
2. Verificar que `api: 8000/metrics` esté **UP** (verde)

---

## 📊 Queries Útiles

### Total de requests

```promql
http_requests_total
```

![Query Result](../../images/prometheus-02-graph.png)

### Requests por segundo

```promql
rate(http_requests_total[5m])
```

### Latencia promedio

```promql
rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])
```

### Percentil 95 de latencia

```promql
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

### Tasa de errores (5xx)

```promql
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) * 100
```

---

## 🔧 Configuración

**Archivo:** `prometheus/prometheus.yml`

```yaml
scrape_configs:
  - job_name: 'fastapi'
    scrape_interval: 15s
    static_configs:
      - targets: ['api:8000']
```

---

## 📖 Referencias

- [PromQL Cheat Sheet](https://promlabs.com/promql-cheat-sheet/)
- [Prometheus Docs](https://prometheus.io/docs/)

---

[← Volver al índice](../index.md) | [Siguiente:  Grafana →](grafana.md)