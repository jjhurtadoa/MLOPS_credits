# Grafana - Visualización de Métricas

## 🎯 Propósito

Grafana lee métricas de Prometheus y las visualiza en dashboards interactivos.

---

## 🚀 Acceso

- **URL:** http://localhost:3000
- **Usuario:** `admin`
- **Contraseña:** `admin`

---

## 📊 Dashboard Principal

![Grafana Dashboard](../../images/grafana-01-dashboard.png)

### Paneles incluidos

1. **Total Requests** - Contador total
2. **Requests per Second** - Gráfica de tiempo
3. **Average Response Time** - Latencia promedio
4. **Requests by Endpoint** - Distribución de tráfico

---

## 🎨 Crear Panel Personalizado

### 1. Agregar panel

Click en **"Add"** → **"Visualization"**

### 2. Seleccionar datasource

Seleccionar **"Prometheus"**

### 3. Query PromQL

```promql
rate(http_requests_total{path="/predict"}[5m])
```

### 4. Configurar visualización

- **Type:** Time series
- **Title:** "Predictions per Second"
- **Unit:** ops/sec

### 5. Aplicar

Click **"Apply"**

---

## 📸 Ejemplo de Panel de Latencia

![Grafana Detail](../../images/grafana-02-detail.png)

**Query:**
```promql
histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, path))
```

---

## 🔔 Alertas (Mejora futura)

```yaml
# Alerta si latencia > 1s
alert:  HighLatency
expr: http_request_duration_seconds_sum / http_request_duration_seconds_count > 1
for:  5m
annotations:
  summary:  "Latencia alta detectada"
```

---

## 📖 Referencias

- [Grafana Dashboards](https://grafana.com/grafana/dashboards/)
- [PromQL for Grafana](https://prometheus.io/docs/prometheus/latest/querying/basics/)

---

[← Volver a Prometheus](prometheus.md) | [Volver al índice](../index.md)